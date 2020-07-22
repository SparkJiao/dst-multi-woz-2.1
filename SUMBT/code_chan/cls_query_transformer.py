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
except ImportError:
    import layers
    from metric import Average
    from global_logger import get_child_logger

logger: Logger = get_child_logger(__name__)

ACT2FN = {'gelu': gelu, "relu": F.relu, "tanh": F.tanh}


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        return self.bert(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                         attention_mask=attention_mask, head_mask=head_mask,
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

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        logger.info(f'Value vectors embedding type: {args.value_embedding_type}')
        self.value_embedding_type = args.value_embedding_type
        # self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        query_attn_config = copy.deepcopy(self.sv_encoder.config)
        query_attn_config.num_attention_heads = self.attn_head
        self.query_attn = layers.Attention(query_attn_config, add_output=True, use_residual=False, add_layer_norm=True)

        # NBT
        nbt_config = copy.deepcopy(self.sv_encoder.config)
        nbt_config.num_hidden_layers = args.num_layers
        nbt_config.intermediate_size = args.intermediate_size
        self.nbt_config = nbt_config

        self.position_embedding = self.utterance_encoder.bert.embeddings.position_embeddings
        self.positionLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=nbt_config.layer_norm_eps)
        self.transformer = BertEncoder(nbt_config)

        self.context_attn = layers.Attention(query_attn_config, add_output=True, use_residual=False, add_layer_norm=True)
        self.gate = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)

        self.clsf_update = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        self.update_linear = nn.Linear(self.bert_output_dim * 2, 2)

        # Matching
        self.linear = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        self.matchingLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=nbt_config.layer_norm_eps)

        # Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        elif self.distance_metric == 'product':
            self.metric = layers.ProductSimilarity(self.bert_output_dim)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Metric
        self.metrics = {
            "update_loss": {
                "train": Average(),
                "eval": Average(),
            },
            "matching_loss": {
                "train": Average(),
                "eval": Average(),
            },
            "update_acc": {
                "train": Average(),
                "eval": Average(),
            }
        }

    def initialize_slot_value_lookup(self, label_ids, slot_ids, slot_token_type_ids=None):

        self.sv_encoder.eval()

        # Slot encoding
        slot_mask = slot_ids > 0
        hid_slot = self.sv_encoder(input_ids=slot_ids.view(-1, self.max_slot_length),
                                   attention_mask=slot_mask.view(-1, self.max_slot_length))[0]
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

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
                # self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                # self.value_lookup[s].padding_idx = -1
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

    def forward(self, input_ids, token_type_ids, attention_mask, labels, update=None, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)

        hidden = self.utterance_encoder(input_ids=input_ids.view(-1, self.max_seq_length),
                                        token_type_ids=token_type_ids.view(-1, self.max_seq_length),
                                        attention_mask=attention_mask.view(-1, self.max_seq_length))[0]
        # hidden = hidden.mul(hidden, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden.size()))
        attention_mask = self.utterance_encoder.get_extended_attention_mask(attention_mask.view(bs, -1),
                                                                            (bs, slot_dim),
                                                                            device=hidden.device)

        slot_hidden = self.slot_lookup.weight[target_slot, :]
        expand_slot_hidden = slot_hidden.unsqueeze(0).expand(bs, -1, -1)

        hidden = self.query_attn(expand_slot_hidden, hidden, hidden, attention_mask)[0]
        hidden_curr = hidden.view(ds, ts, slot_dim, -1).transpose(1, 2).reshape(ds * slot_dim, ts, -1)

        turn_ids = torch.arange(ts, dtype=torch.long, device=hidden.device)
        casual_mask = turn_ids[None, :].repeat(ts, 1) <= turn_ids[:, None]
        casual_mask = casual_mask.unsqueeze(0).expand(ds * slot_dim, -1, -1)
        casual_mask = self.utterance_encoder.get_extended_attention_mask(casual_mask, (ds * slot_dim, ts),
                                                                         device=hidden.device)

        turn_embedding = self.position_embedding(turn_ids).unsqueeze(0).expand(ds * slot_dim, -1, -1)
        hidden = self.dropout(self.positionLayerNorm(turn_embedding + hidden_curr))

        # (ds * slot_dim, ts, h)
        hidden = self.transformer(hidden, casual_mask,
                                  head_mask=self.utterance_encoder.get_head_mask(None,
                                                                                 self.nbt_config.num_hidden_layers))[0]
        hidden = self.context_attn(slot_hidden.unsqueeze(0).expand(ds, -1, -1).reshape(ds * slot_dim, 1, -1),
                                   hidden, hidden, casual_mask)[0]
        g = torch.sigmoid(self.gate(torch.cat([hidden_curr, hidden], dim=-1)))
        hidden = g * hidden_curr + (1 - g) * hidden

        hidden = hidden.view(ds, slot_dim, ts, -1).transpose(0, 1).contiguous()

        loss = 0
        # print(update)
        if update is not None:
            hidden_update = torch.tanh(self.clsf_update(self.dropout(hidden)))
            prob_update = self.update_linear(torch.cat([hidden_update, torch.cat([torch.zeros_like(hidden_update[:, :, :1]).to(device=hidden_update.device), hidden_update[:, :, :-1]], dim=2)], dim=-1))
            loss_update = self.nll(prob_update.view(-1, 2), update.permute(2, 0, 1).reshape(-1))
            _, update_pred = torch.max(prob_update, -1)
            num_update_label = torch.sum(update != -1).float()
            acc_update = (update_pred == update.permute(2, 0, 1)).sum().float() / num_update_label
            loss = loss_update

            self.update_metric("update_loss", loss_update.detach(), num=ds)
            self.update_metric("update_acc", acc_update, num=num_update_label)
            # print(loss_update)

        # Label (slot-value) encoding
        loss_slot = [0.] * slot_dim
        # pred_slot = []
        # output = []
        type_loss = 0.
        matching_loss = 0.

        hid_label = self.defined_values
        num_slot_labels = hid_label.size(1)

        if self.distance_metric == 'product':
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(slot_dim * ds * ts,
                                                                                          num_slot_labels, -1)
            _hidden = hidden.reshape(slot_dim * ds * ts, 1, -1)
            _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(slot_dim, ds, ts, num_slot_labels)
        else:
            _hid_label = hid_label.unsqueeze(1).unsqueeze(1).repeat(1, ds, ts, 1, 1).view(
                slot_dim * ds * ts * num_slot_labels, -1)
            _hidden = hidden.unsqueeze(3).repeat(1, 1, 1, num_slot_labels, 1).reshape(
                slot_dim * ds * ts * num_slot_labels, -1)
            _dist = self.metric(_hid_label, _hidden).view(slot_dim, ds, ts, num_slot_labels)

        if self.distance_metric == 'euclidean':
            _dist = -_dist

        _, pred_slot = torch.max(_dist, -1)

        if labels is not None:
            # No undefined value fix
            _loss = self.nll(_dist.reshape(-1, num_slot_labels), labels.permute(2, 0, 1).reshape(-1))
            matching_loss += _loss.item()
            loss += _loss
            # loss_slot = _loss.detach()
        # print(loss)
        if labels is None:
            return _dist.detach()

        self.update_metric("matching_loss", matching_loss, num=ds)

        # calculate joint accuracy
        # pred_slot = torch.cat(pred_slot, 2)  # (ds, ts, slot_dim)
        pred_slot = pred_slot.permute(1, 2, 0).contiguous().detach()
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        # Slot accuracy
        slot_data_num = torch.sum(labels.view(-1, slot_dim) > -1, dim=0).float()
        acc_slot = accuracy.sum(dim=0).float() / slot_data_num
        # Joint accuracy
        valid_turn = torch.sum(labels[:, :, 0].view(-1) > -1, dim=0).float()
        acc = sum(torch.sum(accuracy, dim=1) // slot_dim).float() / valid_turn

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot

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
