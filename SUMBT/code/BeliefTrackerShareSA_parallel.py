import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from allennlp.nn.util import masked_log_softmax
from torch.nn.functional import nll_loss

try:
    from . import layers
    from .modeling_bert import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
except ImportError:
    import layers
    from modeling_bert import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config, reduce_layers: int = 0):
        super(BertForUtteranceEncoding, self).__init__(config)
        print(f'Reduce {reduce_layers} of BERT.')
        config.num_hidden_layers = config.num_hidden_layers - reduce_layers
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False,
                start_offset=0, all_attn_cache=None):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers,
                         start_offset=start_offset, all_attn_cache=all_attn_cache)


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

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'), reduce_layers=args.reduce_layers
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False
        if args.fix_bert:
            print('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.bert.parameters():
                p.requires_grad = False

        ### values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # NBT
        nbt_config = self.sv_encoder.config
        nbt_config.num_attention_heads = self.attn_head
        print(f"Dialog Self Attention add layer norm: {args.sa_add_layer_norm}")
        last_attention = self.utterance_encoder.bert.encoder.layer[-1].attention.self
        if args.override_attn:
            print("Override self attention from last layer of BERT")
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=args.sa_add_layer_norm,
                                                         self_attention=last_attention)
        else:
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=args.sa_add_layer_norm)
        if args.share_position_weight:
            print("Dialog self attention will share position embeddings with BERT")
            self.transformer.position_embeddings.weight = self.utterance_encoder.bert.embeddings.position_embeddings.weight

        self.do_cross_slot_attention = args.across_slot
        if args.across_slot:
            print("Add cross slot attention")
            print(f"Cross slot attention add layer norm: {args.ss_add_layer_norm}")
            self.slot_attention = SimpleSelfAttention(nbt_config, add_output=True,
                                                      add_layer_norm=args.ss_add_layer_norm)
        else:
            self.slot_attention = None

        ### Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        elif self.distance_metric == 'product':
            self.metric = layers.ProductSimilarity(self.bert_output_dim)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.parallel_loss = args.parallel_loss

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_mask", slot_mask)

        with torch.no_grad():
            for s, label_id in enumerate(label_ids):
                label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
                label_mask = label_id > 0
                hid_label, _, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                                  label_type_ids.view(-1, self.max_label_length),
                                                  label_mask.view(-1, self.max_label_length),
                                                  output_all_encoded_layers=False,
                                                  start_offset=0, all_attn_cache=None)
                hid_label = hid_label[:, 0, :]
                hid_label = hid_label.detach()
                self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                self.value_lookup[s].padding_idx = -1
        del self.sv_encoder
        torch.cuda.empty_cache()

        print("Complete initialization of slot and value lookup")

    def initialize_slot_value_lookup_parallel(self, label_ids, slot_ids):
        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_mask", slot_mask)

        values_list = []

        with torch.no_grad():
            for s, label_id in enumerate(label_ids):
                label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
                label_mask = label_id > 0
                hid_label, _, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                                  label_type_ids.view(-1, self.max_label_length),
                                                  label_mask.view(-1, self.max_label_length),
                                                  output_all_encoded_layers=False,
                                                  start_offset=0, all_attn_cache=None)
                hid_label = hid_label[:, 0, :]
                hid_label = hid_label.detach()
                # self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                # self.value_lookup[s].padding_idx = -1
                values_list.append(hid_label)
        del self.sv_encoder

        value_sizes = [values.size(0) for values in values_list]
        max_value_num = max(value_sizes)
        value_sizes = torch.tensor(value_sizes, dtype=torch.long)
        value_len_seq = torch.arange(max_value_num, dtype=torch.long).unsqueeze(0).expand(len(values_list), -1)
        value_lookup_mask = (value_len_seq < value_sizes.unsqueeze(1).expand(-1, max_value_num))

        self.register_buffer('value_sizes', value_sizes.to(device=self.device))
        self.register_buffer('value_lookup_mask', value_lookup_mask.to(device=self.device))

        # max_value_num = 0
        # for values in values_list:
        #     max_value_num = max(max_value_num, values.size(0))
        self.value_lookup = values_list[0].new_zeros(len(label_ids), max_value_num, self.bert_output_dim)
        # value_lookup_mask = self.value_lookup.new_ones((len(label_ids), max_value_num), dtype=torch.float)
        for s, values in enumerate(values_list):
            self.value_lookup[s][:values.size(0)] = values
            # value_lookup_mask[s][:values.size(0)] = torch.new_zeros(values.size(0))
            del values
        self.register_buffer('value_lookup_mask', value_lookup_mask)

        torch.cuda.empty_cache()

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

    def forward(self, input_ids, token_type_ids, attention_mask, labels, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)

        # Utterance encoding
        # token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)

        _, _, all_attn_cache = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                                      token_type_ids.view(-1, self.max_seq_length),
                                                      attention_mask.view(-1, self.max_seq_length),
                                                      output_all_encoded_layers=False)

        # Domain-slot encoding
        slot_ids = self.slot_ids.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.max_slot_length)
        slot_mask = self.slot_mask.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.max_slot_length)
        slot_mask = torch.cat(
            [attention_mask.unsqueeze(0).expand(slot_dim, -1, -1, -1).reshape(-1, self.max_seq_length),
             slot_mask], dim=-1)
        assert slot_ids.size(0) == slot_dim * bs
        for attn_cache in all_attn_cache:
            for k in attn_cache:
                attn_cache[k] = attn_cache[k].unsqueeze(0).expand(slot_dim, -1, -1, -1) \
                    .reshape(slot_dim * bs, self.max_seq_length, -1)
        hidden, _, _ = self.utterance_encoder(slot_ids, token_type_ids=None, attention_mask=slot_mask,
                                              output_all_encoded_layers=False, start_offset=self.max_seq_length,
                                              all_attn_cache=all_attn_cache)
        hidden = hidden[:, 0].view(slot_dim, ds, ts, -1)

        if self.do_cross_slot_attention:
            hidden = hidden.view(slot_dim, bs, -1).transpose(0, 1)
            hidden = self.slot_attention(hidden, None).transpose(0, 1).reshape(slot_dim, ds, ts, -1)
        hidden = hidden.view(slot_dim * ds, ts, self.bert_output_dim)

        # NBT
        hidden = self.transformer(hidden, None).view(slot_dim, ds, ts, -1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []

        if self.parallel_loss:
            max_label_num = self.value_lookup.size(1)
            _hidden = hidden.reshape(slot_dim * bs, 1, self.bert_output_dim)
            _value = self.value_lookup.unsqueeze(1).expand(-1, bs, -1, self.bert_output_dim).reshape(slot_dim * bs, -1, self.bert_output_dim)
            if self.distance_metric == 'product':
                _dist = self.metric(_hidden, _value).squeeze(1)
            else:
                _hidden = _hidden.expand(-1, max_label_num, -1)
                _dist = self.metric(_value, _hidden)

                if self.distance_metric == 'euclidean':
                    _dist = -_dist

            _, pred = torch.max(_dist, -1).reshape(slot_dim, ds, ts, max_label_num)

            if labels is not None:
                """ FIX_UNDEFINED: Mask all undefined values """
                # undefined_value_mask = (labels[:, :, s] == num_slot_labels)
                # masked_slot_labels_for_loss = labels[:, :, s].masked_fill(undefined_value_mask, -1)
                # TODO:
                #  1. mask undefined value
                #  2. mask logits of padding values
                pass


        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            if self.distance_metric == 'product':
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts, num_slot_labels, -1)
                _hidden = hidden[s].unsqueeze(2).view(ds * ts, 1, -1)
                _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(ds, ts, num_slot_labels)
            else:
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels,
                                                                                           -1)
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
                loss += _loss

        if labels is None:
            return output

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = torch.sum(accuracy, 0).float() \
                   / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
              / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot.unsqueeze(0)

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
