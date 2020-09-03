import copy
from logging import Logger

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.activations import gelu
from transformers import BertPreTrainedModel, BertModel

try:
    from . import layers
    from .metric import Average
    from .global_logger import get_child_logger
    from .modeling_bi_graph import BiGraphSelfAttention
except ImportError:
    import layers
    from metric import Average
    from global_logger import get_child_logger
    from modeling_bi_graph import BiGraphSelfAttention

logger: Logger = get_child_logger(__name__)

ACT2FN = {'gelu': gelu, "relu": F.relu, "tanh": F.tanh}


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config,
                 cls_n_head=None,
                 cls_d_head=None,
                 graph_residual=None,
                 mask_self=None,
                 graph_add_layers=None):
        super(BertForUtteranceEncoding, self).__init__(config)

        config.cls_n_head = cls_n_head
        config.cls_d_head = cls_d_head
        config.graph_residual = graph_residual
        config.mask_self = mask_self
        self.config = config
        self.bert = BertModel(config)

        if graph_add_layers is not None:
            for x in graph_add_layers:
                self.bert.encoder.layer[x].attention.self = BiGraphSelfAttention(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, **kwargs):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, **kwargs)


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device):
        super(BeliefTracker, self).__init__()

        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.max_slot_length = args.max_slot_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.max_turns = args.max_turn_length
        self.device = device

        # values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir)
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        # NBT
        self.cls_index = None
        if args.graph_add_layers is not None:
            self.graph_add_layers = [int(t) for t in args.graph_add_layers.split(',')]
        else:
            self.graph_add_layers = None

        # Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir,
                                                                          cls_n_head=args.cls_n_head,
                                                                          cls_d_head=args.cls_d_head,
                                                                          graph_residual=args.graph_residual,
                                                                          mask_self=args.mask_self,
                                                                          graph_add_layers=self.graph_add_layers)
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob if args.dropout is None else args.dropout
        logger.info(f'Dropout prob: {self.hidden_dropout_prob}')
        if args.fix_bert:
            logger.info('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        logger.info(f'Value vectors embedding type: {args.value_embedding_type}')
        self.value_embedding_type = args.value_embedding_type
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # Copy mechanism
        self.use_copy = args.use_copy
        if self.use_copy:
            self.hard_copy = args.hard_copy
            copy_config = copy.deepcopy(self.sv_encoder.config)
            copy_config.num_attention_heads = self.attn_head
            self.copy_attention = layers.Attention(copy_config, add_output=False, use_residual=False,
                                                   add_layer_norm=False)
            self.copy_gate = nn.Linear(self.bert_output_dim * 2, 1)
            self.copyLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=copy_config.layer_norm_eps)

        if args.cls_type == 0:
            self.classifier = nn.Linear(self.bert_output_dim, 3)
        elif args.cls_type == 1:
            self.classifier = nn.Sequential(nn.Linear(self.bert_output_dim, self.bert_output_dim),
                                            nn.Tanh(),
                                            nn.Linear(self.bert_output_dim, 3))
        if args.distance_metric == 'product':
            self.hidden_output = nn.Linear(self.bert_output_dim, self.bert_output_dim, bias=False)
        else:
            self.hidden_output = nn.Linear(self.bert_output_dim, self.bert_output_dim)

        # Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        elif self.distance_metric == 'product':
            self.metric = layers.ProductSimilarity(self.bert_output_dim)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        logger.info(f'Classification loss weight: {args.cls_loss_weight}')
        self.cls_loss_weight = args.cls_loss_weight

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.efficient = args.efficient

        # Metric
        self.metrics = {
            "cls_loss": {
                "train": Average(),
                "eval": Average(),
            },
            "matching_loss": {
                "train": Average(),
                "eval": Average(),
            }
        }

    def initialize_slot_value_lookup(self, label_ids, slot_ids, slot_token_type_ids=None):

        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0  # (slot_dim, slot_len)
        slot_mask = slot_mask.to(device=self.device, dtype=torch.long)
        slot_dim, slot_len = slot_mask.size()
        slot_to_slot_mask = torch.zeros(slot_dim, slot_dim, slot_len, dtype=torch.long, device=self.device)
        for i in range(slot_dim):
            slot_to_slot_mask[i, i] = slot_mask[i]
        slot_to_slot_mask = slot_to_slot_mask.unsqueeze(1).expand(-1, slot_len, -1, -1).reshape(
            slot_dim * slot_len, slot_dim * slot_len)
        seq_to_slot_mask = slot_mask.unsqueeze(0).expand(self.max_seq_length, -1, -1).reshape(
            self.max_seq_length, slot_dim * slot_len)

        # [seq_len + slot_dim * slot_len, slot_dim * slot_len]
        seq_to_slot_mask = torch.cat([seq_to_slot_mask, slot_to_slot_mask], dim=0)

        utt_pos_ids = torch.arange(self.max_seq_length, device=self.device, dtype=torch.long)
        slot_pos_ids = self.max_seq_length + torch.arange(slot_len, device=self.device, dtype=torch.long)
        slot_pos_ids = slot_pos_ids.unsqueeze(0).repeat(slot_dim, 1).reshape(slot_dim * slot_len)
        pos_ids = torch.cat([utt_pos_ids, slot_pos_ids], dim=-1)
        # print(pos_ids)

        self.register_buffer("slot_ids", slot_ids.reshape(slot_dim * slot_len))
        if slot_token_type_ids is None:
            slot_token_type_ids = slot_ids.new_zeros(slot_ids.size())
        self.register_buffer("slot_token_type_ids", slot_token_type_ids.reshape(slot_dim * slot_len))
        self.register_buffer("slot_mask", slot_mask)
        self.register_buffer("seq_to_slot_mask", seq_to_slot_mask)
        self.register_buffer("slot_to_slot_mask", slot_to_slot_mask)
        self.register_buffer("pos_ids", pos_ids)
        self.register_buffer("slot_pos_ids", slot_pos_ids)

        turn_ids = torch.arange(self.max_turns, dtype=torch.long, device=self.device)
        casual_mask = (turn_ids[None, :] <= turn_ids[:, None]).float()
        casual_mask = (1 - casual_mask) * -10000.0
        self.register_buffer("casual_mask", casual_mask)

        cls_index = [0] + [self.max_seq_length + off for off in range(0, slot_dim * slot_len, slot_len)]
        self.cls_index = cls_index

        max_value_num = 0
        value_list = []

        with torch.no_grad():
            for s, label_id in enumerate(label_ids):
                label_mask = label_id > 0
                hid_label = self.sv_encoder(input_ids=label_id.view(-1, self.max_label_length),
                                            attention_mask=label_mask.view(-1, self.max_label_length))[0]
                if self.value_embedding_type == 'cls':
                    hid_label = hid_label[:, 0, :]
                else:
                    hid_label = hid_label[:, 1:-1].mean(dim=1)
                hid_label = hid_label.detach()
                max_value_num = max(max_value_num, hid_label.size(0))
                value_list.append(hid_label)
        del self.sv_encoder
        torch.cuda.empty_cache()

        value_tensor = torch.zeros((slot_ids.size(0), max_value_num, self.bert_output_dim),
                                   dtype=value_list[0].dtype, device=self.device)
        value_mask = torch.zeros((slot_ids.size(0), max_value_num), dtype=torch.long, device=self.device)
        for s, value in enumerate(value_list):
            value_num = value.size(0)
            value_tensor[s, :value_num] = value
            value_mask[s, :value_num] = value.new_ones(value.size()[:-1], dtype=torch.long)
        value_tensor = value_tensor.half()
        self.register_buffer("defined_values", value_tensor)
        # self.register_buffer("value_mask", value_mask)
        self.register_buffer("value_mask", ((1 - value_mask).to(dtype=value_tensor.dtype) * -10000.0)[:, None, None, :])

        print("Complete initialization of slot and value lookup")

    def forward(self, input_ids, token_type_ids, attention_mask, answer_type_ids, labels, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)
        total_slot_len = slot_dim * self.max_slot_length

        # Utterance encoding
        input_ids = input_ids.view(-1, self.max_seq_length)
        token_type_ids = token_type_ids.view(-1, self.max_seq_length)
        attention_mask = attention_mask.view(-1, self.max_seq_length)

        input_ids = torch.cat([
            input_ids, self.slot_ids.unsqueeze(0).expand(bs, -1)
        ], dim=1)
        token_type_ids = torch.cat([
            token_type_ids, self.slot_token_type_ids.unsqueeze(0).expand(bs, -1)
        ], dim=1)
        attention_mask = torch.cat([
            attention_mask.unsqueeze(1).expand(
                -1, slot_dim * self.max_slot_length + self.max_seq_length, -1),
            self.seq_to_slot_mask.unsqueeze(0).expand(bs, -1, -1)
        ], dim=-1)
        # position_ids = self.slot_ids.unsqueeze(0).expand(bs, -1)
        position_ids = self.pos_ids.unsqueeze(0).expand(bs, -1)

        for x in self.graph_add_layers:
            self.utterance_encoder.bert.encoder.layer[x].attention.self.set_slot_dim(
                self.cls_index, ts, self.casual_mask[:ts, :ts])

        outputs = self.utterance_encoder(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask,
                                         position_ids=position_ids)

        seq_hidden = outputs[0]
        hidden = seq_hidden[:, self.cls_index[1:]]
        assert hidden.size(1) == slot_dim

        hidden = hidden.transpose(0, 1).reshape(slot_dim, ds, ts, -1)
        # hidden = hidden.view(ds, slot_dim, ts, -1)

        # if self.use_copy:
        #     hidden = hidden.transpose(1, 2).reshape(bs, slot_dim, -1)
        #     copy_hidden = self.copy_attention(hidden, hidden, hidden,
        #                                       attention_mask=self.diagonal_mask, hard=self.hard_copy)
        #     gate = torch.sigmoid((self.copy_gate(hidden, copy_hidden)))
        #     hidden = gate * copy_hidden + (1 - gate) * hidden
        #     hidden = self.copyLayerNorm(self.dropout(hidden)).view(ds, ts, slot_dim, -1).transpose(1, 2)

        # hidden = hidden.transpose(0, 1).contiguous()

        loss = 0

        answer_type_logits = self.classifier(hidden)
        _, answer_type_pred = answer_type_logits.max(dim=-1)

        hidden = self.hidden_output(hidden)

        # Label (slot-value) encoding
        loss_slot = [0.] * slot_dim
        pred_slot = []
        output = []
        type_loss = 0.
        matching_loss = 0.

        hid_label = self.defined_values
        hid_mask = self.value_mask
        num_slot_labels = hid_label.size(1)

        if self.distance_metric == 'product':
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(slot_dim * ds * ts,
                                                                                          num_slot_labels, -1)
            _hidden = hidden.reshape(slot_dim * ds * ts, 1, -1)
            if self.efficient and _hidden.requires_grad:
                _dist = torch.utils.checkpoint.checkpoint(self.metric, _hidden, _hid_label)
                _dist = _dist.squeeze(1).reshape(slot_dim, ds, ts, num_slot_labels)
            else:
                _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(slot_dim, ds, ts, num_slot_labels)
        else:
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(
                slot_dim * ds * ts * num_slot_labels, -1)
            _hidden = hidden.unsqueeze(3).repeat(1, 1, 1, num_slot_labels, 1).reshape(
                slot_dim * ds * ts * num_slot_labels, -1)
            if self.efficient and _hidden.requires_grad:
                _dist = torch.utils.checkpoint.checkpoint(
                    self.metric, _hid_label, _hidden).view(slot_dim, ds, ts, num_slot_labels)
            else:
                _dist = self.metric(_hid_label, _hidden).view(slot_dim, ds, ts, num_slot_labels)

        if self.distance_metric == 'euclidean':
            _dist = -_dist

        _dist = _dist + hid_mask

        _, pred_slot = torch.max(_dist, -1)

        if labels is not None:
            # No undefined value fix
            _loss = self.nll(_dist.reshape(-1, num_slot_labels), labels.permute(2, 0, 1).reshape(-1)) / ds
            matching_loss += _loss.item()
            loss += _loss
            # loss_slot = _loss.detach()

        if answer_type_ids is not None:
            cls_loss = self.nll(answer_type_logits.reshape(-1, 3), answer_type_ids.permute(2, 0, 1).reshape(-1)) / ds
            # loss_slot += cls_loss.detach()
            loss += self.cls_loss_weight * cls_loss
            type_loss += cls_loss.item()

        if labels is None:
            return _dist.detach()

        self.update_metric("cls_loss", type_loss)
        self.update_metric("matching_loss", matching_loss)

        # calculate joint accuracy
        # pred_slot = torch.cat(pred_slot, 2)  # (ds, ts, slot_dim)
        pred_slot = pred_slot.permute(1, 2, 0).contiguous().detach()
        answer_type_pred = answer_type_pred.permute(1, 2, 0).detach()  # (ds, ts, slot_dim)
        # 1 for `none` and `do not care`
        classified_mask = ((answer_type_ids != 2) * (answer_type_ids != -1)).view(-1, slot_dim)
        # mask classifying values as 1
        value_accuracy = (pred_slot == labels).view(-1, slot_dim).masked_fill(classified_mask, 1)
        answer_type_accuracy = (answer_type_pred == answer_type_ids).view(-1, slot_dim)
        # For `none` and `do not care`, value_accuracy is always 1, the final accuracy depends on slot_gate_accuracy
        # For `ptr`, the final accuracy is 1 if and only if value_accuracy == 1 and slot_gate_accuracy == 1
        accuracy = value_accuracy * answer_type_accuracy
        # Slot accuracy
        slot_data_num = torch.sum(answer_type_ids.view(-1, slot_dim) > -1, dim=0).float()
        acc_slot = accuracy.sum(dim=0).float() / slot_data_num
        type_acc_slot = answer_type_accuracy.sum(dim=0).float() / slot_data_num
        # Joint accuracy
        valid_turn = torch.sum(answer_type_ids[:, :, 0].view(-1) > -1, dim=0).float()
        acc = sum(torch.sum(accuracy, dim=1) // slot_dim).float() / valid_turn
        type_acc = sum(torch.sum(answer_type_accuracy, dim=1) // slot_dim).float() / valid_turn

        if n_gpu == 1:
            return loss, loss_slot, acc, type_acc, acc_slot, type_acc_slot, torch.cat(
                [answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1)
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), type_acc.unsqueeze(0), \
                   acc_slot.unsqueeze(0), type_acc_slot.unsqueeze(0), \
                   torch.cat([answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1).unsqueeze(0)

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)

    def update_metric(self, metric_name, *args, **kwargs):
        state = 'train' if self.training else 'eval'
        self.metrics[metric_name][state](*args, **kwargs)

    def get_metric(self, reset=False):
        state = 'train' if self.training else 'eval'
        metrics = {}
        for k, v in self.metrics.items():
            if state in v:
                metrics[k] = v[state].get_metric(reset=reset)
        return metrics

    def get_gate_metric(self, reset=False):
        metric = torch.cat(self.gate_metric, dim=1).squeeze(-1)
        if reset:
            self.gate_metric.clear()
        return metric
