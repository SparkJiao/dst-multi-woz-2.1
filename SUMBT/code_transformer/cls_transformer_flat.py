from logging import Logger

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import DistilBertPreTrainedModel
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertEncoder
from transformers.activations import gelu
import copy

try:
    from . import layers
    from .metric import Average
    from .global_logger import get_child_logger
    from .transformer import PreLNTransformer
except ImportError:
    import layers
    from metric import Average
    from transformer import PreLNTransformer
    from global_logger import get_child_logger

logger: Logger = get_child_logger(__name__)

ACT2FN = {'gelu': gelu, "relu": F.relu, "tanh": F.tanh}


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        return self.bert(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask, head_mask=head_mask,
                         inputs_embeds=inputs_embeds)


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.max_slot_length = args.max_slot_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        # Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir)
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob if args.dropout is None else args.dropout
        logger.info(f'Dropout prob: {self.hidden_dropout_prob}')
        if args.fix_bert:
            logger.info('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        # values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir)
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        logger.info(f'Value vectors embedding type: {args.value_embedding_type}')
        self.value_embedding_type = args.value_embedding_type
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        self.add_query_attn = args.add_query_attn
        if args.add_query_attn:
            query_attn_config = copy.deepcopy(self.sv_encoder.config)
            query_attn_config.num_attention_heads = self.attn_head
            self.query_attn = layers.Attention(query_attn_config, add_output=False, use_residual=False, add_layer_norm=False)
            self.query_output = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)
            self.queryLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=query_attn_config.layer_norm_eps)            

        # NBT
        nbt_config = copy.deepcopy(self.sv_encoder.config)
        nbt_config.num_hidden_layers = args.num_layers
        nbt_config.intermediate_size = args.intermediate_size
        nbt_config.num_attention_heads = self.attn_head
        self.nbt_config = nbt_config

        self.position_embedding = self.utterance_encoder.bert.embeddings.position_embeddings
        self.positionLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=nbt_config.layer_norm_eps)
        self.transformer = BertEncoder(nbt_config)

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
        slot_dim, slot_len = slot_mask.size()
        slot_to_slot_mask = torch.zeros(slot_dim, slot_dim, slot_len)
        for i in range(slot_dim):
            slot_to_slot_mask[i, i] = slot_mask[i]
        slot_to_slot_mask = slot_to_slot_mask.unsqueeze(1).expand(-1, slot_len, -1, -1).reshape(slot_dim * slot_len, slot_dim * slot_len)
        seq_to_slot_mask = torch.zeros((self.max_seq_length, slot_len * slot_dim), dtype=torch.long, device=self.device)

        utt_pos_ids = torch.arange(self.max_seq_length, device=self.device, dtype=torch.long)
        slot_pos_ids = self.max_seq_length + torch.arange(self.max_slot_length, device=self.device, dtype=torch.long)
        slot_pos_ids = slot_pos_ids.unsqueeze(0).repeat(slot_dim, 1).reshape(slot_dim * self.max_slot_length)
        pos_ids = torch.cat([utt_pos_ids, slot_pos_ids], dim=-1)
        
        self.register_buffer("slot_ids", slot_ids)
        if slot_token_type_ids is None:
            slot_token_type_ids = slot_ids.new_zeros(slot_ids.size())
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask.to(dtype=torch.long))
        self.register_buffer("slot_to_slot_mask", slot_to_slot_mask.to(dtype=torch.long, device=self.device))
        self.register_buffer("seq_to_slot_mask", seq_to_slot_mask)
        self.register_buffer("pos_ids", pos_ids)

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
                self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                self.value_lookup[s].padding_idx = -1
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

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0:  # padding
                    break
                elif len[i, j, 1] > 0:  # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids, token_type_ids, attention_mask, answer_type_ids, labels, n_gpu=1, target_slot=None, transfer_labels=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)

        seq_attention_mask = attention_mask
        attention_mask = attention_mask.view(-1, 1, self.max_seq_length).to(dtype=torch.long)
        flat_attention_mask = attention_mask.view(-1, 1, self.max_seq_length).expand(-1, self.max_seq_length, -1)  # (bs, seq_len, seq_len)
        # (bs, seq_len, seq_len + slot_len)
        flat_attention_mask = torch.cat([flat_attention_mask, self.seq_to_slot_mask.unsqueeze(0).expand(bs, -1, -1)], dim=-1)
        # (bs, slot_len, seq_len + slot_len)
        slot_flat_mask = torch.cat([attention_mask.expand(-1, self.max_slot_length * slot_dim, -1),
                                    self.slot_to_slot_mask.unsqueeze(0).expand(bs, -1, -1)], dim=-1)
        attention_mask = torch.cat([flat_attention_mask, slot_flat_mask], dim=1)
        input_ids = torch.cat([input_ids.view(-1, self.max_seq_length), self.slot_ids.unsqueeze(0).expand(
            bs, -1, -1).reshape(bs, slot_dim * self.max_slot_length)], dim=1)
        token_type_ids = torch.cat([token_type_ids.view(-1, self.max_seq_length), self.slot_token_type_ids.unsqueeze(0).expand(
            bs, -1, -1).reshape(bs, -1)], dim=1)
        pos_ids = self.pos_ids.unsqueeze(0).expand(bs, -1).contiguous()

        # Utterance encoding
        output = self.utterance_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=pos_ids)
        slot_hidden = output[0][:, self.max_seq_length:]
        seq_hidden = output[0][:, :self.max_seq_length]
        slot_hidden = slot_hidden.view(bs, slot_dim, self.max_slot_length, self.bert_output_dim)[:, :, 0]
        slot_h = slot_hidden.view(ds, ts, slot_dim, -1).transpose(1, 2).reshape(ds * slot_dim, ts, -1)

        if self.add_query_attn:
            queried_seq_h = self.query_attn(slot_hidden, seq_hidden, seq_hidden,
                                            attention_mask=self.utterance_encoder.get_extended_attention_mask(seq_attention_mask.view(bs, -1),
                                                                                                              (bs, slot_dim), self.device))[0]
            queried_seq_h = queried_seq_h.view(ds, ts, slot_dim, -1).transpose(1, 2).reshape(ds * slot_dim, ts, -1)
            slot_h = self.queryLayerNorm(self.dropout(self.query_output(torch.cat([slot_h, queried_seq_h], dim=-1))))

        turn_ids = torch.arange(ts, dtype=torch.long, device=self.device)
        casual_mask = turn_ids[None, :].repeat(ts, 1) <= turn_ids[:, None]
        casual_mask = casual_mask.unsqueeze(0).expand(ds * slot_dim, -1, -1)
        casual_mask = self.utterance_encoder.get_extended_attention_mask(casual_mask, (ds * slot_dim, ts), self.device)

        turn_embedding = self.position_embedding(turn_ids).unsqueeze(0).expand(ds * slot_dim, -1, -1)
        slot_h = self.dropout(self.positionLayerNorm(turn_embedding + slot_h))

        # (ds * slot_dim, ts, h)
        hidden = self.transformer(slot_h, casual_mask,
                                  head_mask=self.utterance_encoder.get_head_mask(None, self.nbt_config.num_hidden_layers))[0]
        hidden = hidden.view(ds, slot_dim, ts, -1).transpose(0, 1).contiguous()

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
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(slot_dim * ds * ts, num_slot_labels, -1)
            _hidden = hidden.reshape(slot_dim * ds * ts, 1, -1)
            if self.efficient and _hidden.requires_grad:
                _dist = torch.utils.checkpoint.checkpoint(self.metric, _hidden, _hid_label).squeeze(1).reshape(slot_dim, ds, ts, num_slot_labels)
            else:
                _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(slot_dim, ds, ts, num_slot_labels)
        else:
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(slot_dim * ds * ts * num_slot_labels, -1)
            _hidden = hidden.unsqueeze(3).repeat(1, 1, 1, num_slot_labels, 1).reshape(slot_dim * ds * ts * num_slot_labels, -1)
            if self.efficient and _hidden.requires_grad:
                _dist = torch.utils.checkpoint.checkpoint(self.metric, _hid_label, _hidden).view(slot_dim, ds, ts, num_slot_labels)
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

        # for s, slot_id in enumerate(target_slot):  # note: target_slots are successive
        #     # loss calculation
        #     hid_label = self.value_lookup[slot_id].weight
        #     num_slot_labels = hid_label.size(0)
        #
        #     if self.distance_metric == 'product':
        #         _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts, num_slot_labels, -1)
        #         _hidden = hidden[s].unsqueeze(2).view(ds * ts, 1, -1)
        #         _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(ds, ts, num_slot_labels)
        #     else:
        #         _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
        #         _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(
        #             ds * ts * num_slot_labels, -1)
        #         _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
        #
        #     if self.distance_metric == "euclidean":
        #         _dist = -_dist
        #     _, pred = torch.max(_dist, -1)
        #     pred_slot.append(pred.view(ds, ts, 1))
        #     output.append(_dist)
        #
        #     if labels is not None:
        #         """ FIX_UNDEFINED: Mask all undefined values """
        #         undefined_value_mask = (labels[:, :, s] == num_slot_labels)
        #         masked_slot_labels_for_loss = labels[:, :, s].masked_fill(undefined_value_mask, -1)
        #
        #         _loss = self.nll(_dist.view(ds * ts, -1), masked_slot_labels_for_loss.view(-1)) / (ds * 1.0)
        #         loss_slot.append(_loss.item())
        #         matching_loss += _loss.item()
        #         loss += _loss
        #
        #         if self.graph_value_sup > 0:
        #             graph_value_loss = nn.functional.nll_loss(masked_value_scores[s].view(bs, -1),
        #                                                       masked_slot_labels_for_loss.view(-1), ignore_index=-1,
        #                                                       reduction='sum') / ds
        #             graph_value_loss = self.graph_value_sup * graph_value_loss
        #             loss += graph_value_loss
        #             graph_loss += graph_value_loss.item()
        #
        #     if answer_type_ids is not None:
        #         cls_loss = self.nll(answer_type_logits[s].view(ds * ts, -1), answer_type_ids[:, :, s].view(-1)) / ds
        #         loss_slot[-1] += cls_loss.item()
        #         cls_loss = cls_loss * self.cls_loss_weight
        #         type_loss += cls_loss.item()
        #         loss += cls_loss
        #
        # if labels is None:
        #     return output

        self.update_metric("cls_loss", type_loss)
        self.update_metric("matching_loss", matching_loss)

        # calculate joint accuracy
        # pred_slot = torch.cat(pred_slot, 2)  # (ds, ts, slot_dim)
        pred_slot = pred_slot.permute(1, 2, 0).contiguous().detach()
        answer_type_pred = answer_type_pred.permute(1, 2, 0).detach()  # (ds, ts, slot_dim)
        classified_mask = ((answer_type_ids != 2) * (answer_type_ids != -1)).view(-1, slot_dim)  # 1 for `none` and `do not care`
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

        # accuracy = (pred_slot == labels).view(-1, slot_dim)
        # acc_slot = torch.sum(accuracy, 0).float() / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        # acc = sum(torch.sum(accuracy, 1) / slot_dim).float() / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, type_acc, acc_slot, type_acc_slot, torch.cat(
                [answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1)
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), type_acc.unsqueeze(0), acc_slot.unsqueeze(0), type_acc_slot.unsqueeze(
                0), torch.cat([answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1).unsqueeze(0)

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

    def get_value_scores(self, reset=False):
        value_scores = torch.cat(self.value_scores, dim=1)
        value_scores = value_scores.max(dim=-1)
        if reset:
            self.value_scores.clear()
        return value_scores

    def get_graph_scores(self, reset=False):
        graph_scores = torch.cat(self.graph_scores, dim=0)
        if reset:
            self.graph_scores.clear()
        return graph_scores
