import os.path
from logging import Logger

import torch
import torch.nn as nn
from allennlp.training.metrics import Average
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, gelu
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

try:
    from . import layers
    from .modeling_bert_extended_f import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from .global_logger import get_child_logger
    from .transformer import PreLNTransformer, bert_config2distill_bert_config
except ImportError:
    import layers
    from modeling_bert_extended_f import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from global_logger import get_child_logger
    from transformer import PreLNTransformer, bert_config2distill_bert_config

logger: Logger = get_child_logger(__name__)

ACT2FN = {'gelu': gelu, "relu": F.relu, "tanh": F.tanh}


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config, reduce_layers: int = 0, self_attention_type: int = 0):
        super(BertForUtteranceEncoding, self).__init__(config)
        logger.info(f'Reduce {reduce_layers} of BERT.')
        logger.info(f'Self Attention Type: {self_attention_type}')
        config.num_hidden_layers = config.num_hidden_layers - reduce_layers
        config.slot_attention_type = -1
        config.key_type = 0
        config.self_attention_type = self_attention_type
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids=None, output_all_encoded_layers=False,
                start_offset=0, all_attn_cache=None, **kwargs):
        return self.bert(input_ids, token_type_ids, attention_mask, position_ids=position_ids,
                         output_all_encoded_layers=output_all_encoded_layers,
                         start_offset=start_offset, all_attn_cache=all_attn_cache, **kwargs)


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
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'), reduce_layers=args.reduce_layers,
            self_attention_type=args.self_attention_type
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob if args.dropout is None else args.dropout
        logger.info(f'Dropout prob: {self.hidden_dropout_prob}')
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False
        if args.fix_bert:
            logger.info('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.bert.parameters():
                p.requires_grad = False

        # values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        logger.info(f'Value vectors embedding type: {args.value_embedding_type}')
        self.value_embedding_type = args.value_embedding_type
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # NBT
        nbt_config = bert_config2distill_bert_config(self.sv_encoder.config)
        nbt_config.n_layers = args.num_layers
        nbt_config.dropout = self.hidden_dropout_prob
        nbt_config.context_attn = True
        nbt_config.interact_attn = True
        nbt_config.matching_attn = True

        self.position_embedding = self.utterance_encoder.bert.embeddings.position_embeddings
        self.transformer = PreLNTransformer(nbt_config)
        self.final_layer_norm = nn.LayerNorm(self.bert_output_dim, eps=1e-12)

        self.linear_seq = layers.LinearSeq(self.bert_output_dim)

        self.mask_self = args.mask_self
        logger.info(f'If mask self during graph attention: {self.mask_self}')

        if args.slot_res is not None:
            self.slot_res = [int(x) for x in args.slot_res.split(":")]
            logger.info(f'Slot restriction: {self.slot_res}')
        else:
            self.slot_res = None

        if args.cls_type == 0:
            self.classifier = nn.Linear(self.bert_output_dim, 3)
        elif args.cls_type == 1:
            self.classifier = nn.Sequential(nn.Linear(self.bert_output_dim, self.bert_output_dim),
                                            nn.Tanh(),
                                            nn.Linear(self.bert_output_dim, 3))
        self.hidden_output = nn.Linear(self.bert_output_dim, self.bert_output_dim, bias=False)

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
        slot_mask = slot_ids > 0

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask.to(dtype=torch.long))

        if self.slot_res is not None:
            slot_dim = slot_ids.size(0)
            slot_res = torch.ones(slot_dim, slot_dim, device=self.device, dtype=torch.long)
            slot_res[:, self.slot_res] = torch.zeros(slot_dim, len(self.slot_res), device=self.device, dtype=torch.long)
            self.slot_res = slot_res

        max_value_num = 0
        value_list = []

        with torch.no_grad():
            for s, label_id in enumerate(label_ids):
                label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
                label_mask = label_id > 0
                hid_label, _, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                                  label_type_ids.view(-1, self.max_label_length),
                                                  label_mask.view(-1, self.max_label_length),
                                                  output_all_encoded_layers=False,
                                                  start_offset=0, all_attn_cache=None)
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
        self.register_buffer("defined_values", value_tensor)
        self.register_buffer("value_mask", value_mask)

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

        # Utterance encoding
        utt_h, _, all_attn_cache = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                                          token_type_ids.view(-1, self.max_seq_length),
                                                          attention_mask.view(-1, self.max_seq_length),
                                                          output_all_encoded_layers=False)

        # Domain-slot encoding
        slot_ids = self.slot_ids.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.max_slot_length)
        slot_mask = self.slot_mask.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.max_slot_length)
        slot_mask = torch.cat(
            [attention_mask.unsqueeze(0).expand(slot_dim, -1, -1, -1).reshape(-1, self.max_seq_length),
             slot_mask.to(dtype=attention_mask.dtype)], dim=-1)

        if self.slot_token_type_ids is not None:
            slot_token_type_ids = self.slot_token_type_ids.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.max_slot_length)
        else:
            slot_token_type_ids = None

        hidden, _, _ = self.utterance_encoder(slot_ids, token_type_ids=slot_token_type_ids, attention_mask=slot_mask,
                                              output_all_encoded_layers=False, all_attn_cache=all_attn_cache,
                                              start_offset=self.max_seq_length, slot_dim=slot_dim)
        hidden = self.linear_seq(hidden).view(slot_dim, ds, ts, -1).permute(1, 2, 0, 3).contiguous()

        turn_ids = torch.arange(ts, dtype=torch.long, device=hidden.device)
        casual_mask = turn_ids[None, :].repeat(ts, 1) <= turn_ids[:, None]
        casual_mask = casual_mask.unsqueeze(0).expand(ds * slot_dim, -1, -1)

        turn_embedding = self.position_embedding(turn_ids)[None, :, None, :].repeat(ds, 1, slot_dim, 1)
        hidden = hidden + turn_embedding

        if self.mask_self:
            interact_mask = torch.ones(bs, slot_dim, dtype=torch.long, device=hidden.device)
        else:
            interact_mask = (1 - torch.diag(torch.ones(slot_dim, dtype=torch.long, device=hidden.device))).unsqueeze(0).repeat(bs, 1, 1)
        hidden = self.transformer(hidden, utt_h, utt_mask=attention_mask.view(-1, self.max_seq_length),
                                  casual_mask=casual_mask, attn_mask=interact_mask)[0]
        hidden = self.final_layer_norm(hidden).permute(2, 0, 1, 3).contiguous()

        loss = 0

        answer_type_logits = self.classifier(hidden)
        _, answer_type_pred = answer_type_logits.max(dim=-1)

        hidden = self.hidden_output(hidden)

        # Label (slot-value) encoding
        loss_slot = []
        pred_slot = []
        output = []
        type_loss = 0.
        matching_loss = 0.
        graph_loss = 0.
        for s, slot_id in enumerate(target_slot):  # note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            if self.distance_metric == 'product':
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts, num_slot_labels, -1)
                _hidden = hidden[s].unsqueeze(2).view(ds * ts, 1, -1)
                _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(ds, ts, num_slot_labels)
            else:
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
                _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(
                    ds * ts * num_slot_labels, -1)
                _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                """ FIX_UNDEFINED: Mask all undefined values """
                undefined_value_mask = (labels[:, :, s] == num_slot_labels)
                masked_slot_labels_for_loss = labels[:, :, s].masked_fill(undefined_value_mask, -1)

                _loss = self.nll(_dist.view(ds * ts, -1), masked_slot_labels_for_loss.view(-1)) / (ds * 1.0)
                loss_slot.append(_loss.item())
                matching_loss += _loss.item()
                loss += _loss

            if answer_type_ids is not None:
                cls_loss = self.nll(answer_type_logits[s].view(ds * ts, -1), answer_type_ids[:, :, s].view(-1)) / ds
                loss_slot[-1] += cls_loss.item()
                cls_loss = cls_loss * self.cls_loss_weight
                type_loss += cls_loss.item()
                loss += cls_loss

        if labels is None:
            return output

        self.update_metric("cls_loss", type_loss)
        self.update_metric("matching_loss", matching_loss)

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)  # (ds, ts, slot_dim)
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
