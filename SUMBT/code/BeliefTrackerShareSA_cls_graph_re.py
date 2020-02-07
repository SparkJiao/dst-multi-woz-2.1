import os.path
from logging import Logger

import torch
import torch.nn as nn
from allennlp.training.metrics import Average
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss

try:
    from . import layers
    from .modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from .global_logger import get_child_logger
except ImportError:
    import layers
    from modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from global_logger import get_child_logger

logger: Logger = get_child_logger(__name__)


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
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
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

        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # NBT
        nbt_config = self.sv_encoder.config
        nbt_config.num_attention_heads = self.attn_head
        # logger.info(f"Dialog Self Attention add layer norm: {args.sa_add_layer_norm}")
        # last_attention = self.utterance_encoder.bert.encoder.layer[-1].attention.self
        # if args.override_attn:
        #     logger.info("Override self attention from last layer of BERT")
        #     self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
        #                                                  add_layer_norm=args.sa_add_layer_norm,
        #                                                  self_attention=last_attention)
        # else:
        #     self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
        #                                                  add_layer_norm=args.sa_add_layer_norm)
        # if args.share_position_weight:
        #     logger.info("Dialog self attention will share position embeddings with BERT")
        #     self.transformer.position_embeddings.weight = self.utterance_encoder.bert.embeddings.position_embeddings.weight

        # logger.info(f"If extra neural belief tracker: {args.extra_nbt}")
        # self.extra_nbt = args.extra_nbt
        # if self.extra_nbt:
        #     self.belief_tracker = SimpleDialogSelfAttention(nbt_config, add_output=True,
        #                                                     add_layer_norm=args.sa_add_layer_norm)
        #     if args.share_position_weight:
        #         self.belief_tracker.position_embeddings.weight = self.utterance_encoder.bert.embeddings.position_embeddings.weight

        nbt_config.num_attention_heads = 1
        self.value_attention = layers.MultiHeadAttention(nbt_config)
        # self.value_project = layers.SimpleTransform(self.bert_output_dim)

        self.mask_self = args.mask_self
        logger.info(f'If mask self during graph attention: {self.mask_self}')

        self.graph_attention = layers.MultiHeadAttention(nbt_config)
        self.graph_project = layers.DynamicFusion(self.bert_output_dim, gate_type=1)

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
        logger.info(f"Graph scores supervision coherent: {args.graph_add_sup}")
        self.graph_add_sup = args.graph_add_sup
        logger.info(f"Graph value scores supervision coherent: {args.graph_value_sup}")
        self.graph_value_sup = args.graph_value_sup

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # logger.info(f'If parallel decoding: {args.parallel}')
        # self.parallel = args.parallel
        # self.parallel = False

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
        if self.graph_add_sup > 0:
            self.metrics["slot_negative_loss"] = {
                "train": Average(),
                "eval": Average()
            }
        if self.graph_value_sup > 0:
            self.metrics["graph_value_loss"] = {
                "train": Average(),
                "eval": Average()
            }

    def initialize_slot_value_lookup(self, label_ids, slot_ids, slot_token_type_ids=None):

        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask.to(dtype=torch.long))

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
                hid_label = hid_label[:, 0, :]
                hid_label = hid_label.detach()
                self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                self.value_lookup[s].padding_idx = -1
                max_value_num = max(max_value_num, hid_label.size(0))
                value_list.append(hid_label)

        # [CLS] no value [SEP]
        no_value_ids = [101, 2053, 3643, 102]
        no_value_ids = no_value_ids + [0] * (label_ids[0].size(0) - 4)
        no_value_ids = torch.tensor(no_value_ids, device=self.device, dtype=torch.long)
        no_value_type_ids = no_value_ids.new_zeros(no_value_ids.size())
        no_value_mask = no_value_ids > 0
        no_value_label, _, _ = self.sv_encoder(no_value_ids.unsqueeze(0),
                                               no_value_type_ids.unsqueeze(0),
                                               no_value_mask.unsqueeze(0),
                                               output_all_encoded_layers=False, start_offset=0, all_attn_cache=None)
        no_value_label = no_value_label[:, 0, :].squeeze(0)

        del self.sv_encoder
        torch.cuda.empty_cache()

        # add `no value` label at index 0
        value_tensor = torch.zeros((slot_ids.size(0), max_value_num + 1, self.bert_output_dim),
                                   dtype=value_list[0].dtype, device=self.device)
        value_mask = torch.zeros((slot_ids.size(0), max_value_num + 1), dtype=torch.long, device=self.device)
        for s, value in enumerate(value_list):
            value_num = value.size(0)
            value_tensor[s, 0] = no_value_label
            value_tensor[s, 1:(value_num + 1)] = value
            value_mask[s, :(value_num + 1)] = value.new_ones(value_num + 1, dtype=torch.long)
        self.register_buffer("defined_values", value_tensor)
        self.register_buffer("value_mask", value_mask)

        print("Complete initialization of slot and value lookup")

    def initialize_slot_value_lookup_parallel(self, label_ids, slot_ids, slot_token_type_ids=None):

        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask.to(dtype=torch.long))

        value_lookup = []
        value_num = []

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
                value_lookup.append(hid_label)  # (value_num, h)
                value_num.append(hid_label.size(0))

        max_value_num = max(value_num)

        value_tensor = torch.zeros((slot_ids.size(0), max_value_num, self.bert_output_dim), dtype=torch.float, device=self.device)
        value_mask = torch.zeros((slot_ids.size(0), max_value_num), dtype=torch.long, device=self.device)
        for s, value in enumerate(value_lookup):
            value_tensor[s, :value.size(0)] = value
            value_mask[s, :value.size(0)] = value.new_ones(value.size(), dtype=torch.long)

        self.register_buffer("defined_values", value_tensor)
        self.register_buffer("value_mask", value_mask)
        self.register_buffer("value_num", torch.tensor(value_num, dtype=torch.long, device=self.device))

        del self.sv_encoder
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

    def forward(self, input_ids, token_type_ids, attention_mask, answer_type_ids, labels, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)

        # Utterance encoding
        _, _, all_attn_cache = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
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
        hidden = hidden[:, 0].view(slot_dim * ds, ts, -1)

        # Iteration on dialog turns
        max_value_num = self.value_mask.size(1)
        value_mask = self.value_mask.unsqueeze(1).expand(-1, ds, -1).reshape(slot_dim * ds, 1, 1, max_value_num)
        extended_value_mask = (1 - value_mask.to(dtype=hidden.dtype)) * -10000.0
        value_tensor = self.defined_values.unsqueeze(1).expand(-1, ds, -1, -1).reshape(slot_dim * ds, max_value_num, -1)

        fused_hidden = [hidden[:, 0].unsqueeze(1)]
        masked_value_scores = []
        graph_scores = []
        value_mask = value_mask.view(slot_dim * ds, max_value_num)
        graph_mask = hidden.new_zeros(ds, 1, 1, slot_dim)
        for i in range(1, ts):
            # (slot_dim * ds, 1, h)
            pre_value_hidden, (_, _, _, pre_value_scores) = self.value_attention(fused_hidden[-1], value_tensor,
                                                                                 value_tensor, extended_value_mask)
            pre_value_scores = layers.masked_log_softmax_fp16(pre_value_scores.view(slot_dim * ds, max_value_num),
                                                              value_mask, dim=-1)
            masked_value_scores.append(pre_value_scores.squeeze(1))

            turn_hidden, (_, _, _, _turn_scores) = self.graph_attention(hidden[:, i].view(slot_dim, ds, -1).transpose(0, 1),
                                                                        fused_hidden[-1].view(slot_dim, ds, -1).transpose(0, 1),
                                                                        pre_value_hidden.view(slot_dim, ds, -1).transpose(0, 1),
                                                                        graph_mask)
            turn_hidden = turn_hidden.transpose(0, 1).reshape(slot_dim * ds, -1)
            fused_hidden.append(self.graph_project(hidden[:, i], turn_hidden).unsqueeze(1))
            graph_scores.append(_turn_scores.unsqueeze(2))

        hidden = torch.cat(fused_hidden, dim=1).view(slot_dim, ds, ts, -1)
        masked_value_scores = torch.cat(masked_value_scores, dim=1).view(slot_dim, ds, (ts - 1), -1)

        # Graph supervision
        loss = 0
        # if self.graph_add_sup > 0:
        #     num_head = graph_scores.size(1)
        #     slot_target = answer_type_ids[:, :-1].contiguous().view(ds * (ts - 1), 1, 1, slot_dim).expand(-1, num_head, slot_dim, -1)
        #     loss = self.graph_add_sup * layers.negative_entropy(score=graph_scores,
        #                                                         target=((slot_target == 0) | (slot_target == 1))) / (ds * num_head)
        #
        #     self.update_metric("slot_negative_loss", loss.item())

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

                if self.graph_value_sup > 0:
                    # add `no value` labels: set offset = 1
                    negative_mask = (masked_slot_labels_for_loss == -1)
                    off_labels = masked_slot_labels_for_loss + 1
                    no_value_mask = ((answer_type_ids[:, :, s] == 0) | (answer_type_ids[:, :, s] == 1))
                    off_labels[negative_mask] = -1  # set previous -1 labels still as -1
                    off_labels[no_value_mask] = 0  # set no value positions as 0
                    off_labels = off_labels[:, :-1]  # no supervision for the last turn
                    graph_value_loss = nn.functional.nll_loss(masked_value_scores[s].view(ds * (ts - 1), -1),
                                                              off_labels.view(-1), ignore_index=-1,
                                                              reduction='sum') / ds
                    graph_value_loss = self.graph_value_sup * graph_value_loss
                    loss += graph_value_loss
                    graph_loss += graph_value_loss.item()

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
        if self.graph_value_sup > 0:
            self.update_metric("graph_value_loss", graph_loss)

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
            return loss.unsqueeze(0), None, acc.unsqueeze(0), type_acc.unusqueeze(0), acc_slot.unsqueeze(0), type_acc_slot.unsqueeze(
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
            metrics[k] = v[state].get_metric(reset=reset)
        return metrics