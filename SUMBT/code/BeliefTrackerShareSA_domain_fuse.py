import logging
import os.path

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from allennlp.training.metrics import CategoricalAccuracy, Average

try:
    from . import layers
    from .modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from .global_logger import get_child_logger
except ImportError:
    import layers
    from modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
    from global_logger import get_child_logger


logger: logging.Logger = get_child_logger(__name__)


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
        logger.info(f"Dialog Self Attention add layer norm: {args.sa_add_layer_norm}")
        last_attention = self.utterance_encoder.bert.encoder.layer[-1].attention.self
        if args.override_attn:
            logger.info("Override self attention from last layer of BERT")
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=args.sa_add_layer_norm,
                                                         self_attention=last_attention)
        else:
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=args.sa_add_layer_norm)

        if args.share_position_weight:
            logger.info("Dialog self attention will share position embeddings with BERT")
            self.transformer.position_embeddings.weight = self.utterance_encoder.bert.embeddings.position_embeddings.weight

        self.domain_fusion1 = layers.Attention(nbt_config, add_output=False, use_residual=False, add_layer_norm=False)
        self.fusion_projection = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)
        self.domain_fusion2 = layers.Attention(nbt_config, add_output=True, use_residual=False, add_layer_norm=False)

        self.domain_classifier = nn.Linear(self.bert_output_dim, 2)
        self.classifier = layers.ProjectionTransform(self.bert_output_dim, num_classes=3)

        self.domain_spans = []
        self.domain_lens = []
        self.regularize_prob = args.regularize_prob
        logger.info(f"If regularize probability: {self.regularize_prob}")

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

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Extra metric
        self.domain_cls_acc = {
            "train": CategoricalAccuracy(),
            "eval": CategoricalAccuracy()
        }
        self.domain_cls_loss_avg = {
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

    def set_domain_spans(self, domain_spans):
        self.domain_spans = domain_spans
        self.domain_lens = [e - s for (s, e) in domain_spans]

        slot_dim = self.slot_ids.size(0)
        domain_mask = self.slot_ids.new_zeros((slot_dim, slot_dim), dtype=torch.long)
        for domain_idx, (domain_s, domain_e) in enumerate(self.domain_spans):
            domain_len = self.domain_lens[domain_idx]
            domain_mask[domain_s:domain_e, domain_s:domain_e] = domain_mask.new_ones((domain_len, domain_len))
        self.register_buffer("domain_mask", domain_mask)

        logger.info(f'Domain spans: {self.domain_spans}')

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
        hidden = hidden[:, 0].view(slot_dim, bs, -1)
        domain_hidden = hidden.transpose(0, 1).contiguous()
        domain_mask = (1. - self.domain_mask[None, None, :, :].to(next(self.parameters()).dtype)) * -10000.0
        domain_hidden = self.domain_fusion1(domain_hidden, domain_hidden, domain_hidden, domain_mask)
        hidden = self.fusion_projection(torch.cat([
            hidden.view(slot_dim * ds, ts, -1),
            domain_hidden.transpose(0, 1).reshape(slot_dim * ds, ts, -1)
        ], dim=-1))

        # NBT
        hidden = self.transformer(hidden, None).view(slot_dim, bs, -1).transpose(0, 1).contiguous()
        hidden = self.domain_fusion2(hidden, hidden, hidden, domain_mask).transpose(0, 1).view(slot_dim, ds, ts, -1)

        domain_h = []
        ex_hidden = []
        domain_labels = []
        for domain_idx, (domain_s, domain_e) in enumerate(self.domain_spans):
            _domain_h = hidden[domain_s:domain_e].mean(dim=0).unsqueeze(0)
            domain_h.append(_domain_h)
            ex_hidden.append(_domain_h.expand(self.domain_lens[domain_idx], -1, -1, -1))
            domain_label = labels[:, :, domain_s:domain_e].sum(dim=-1)
            domain_label.clamp_(-1, 1)
            domain_labels.append(domain_label.unsqueeze(0))
        domain_h = torch.cat(domain_h, dim=0)  # (num_domain, ds, ts, h)
        ex_hidden = torch.cat(ex_hidden, dim=0)  # (slot_dim, ds, ts, h)

        domain_labels = torch.cat(domain_labels, dim=0).view(-1)  # (num_domain, ds, ts)
        domain_logits = self.domain_classifier(domain_h)
        loss = self.nll(domain_logits.view(-1, 2), domain_labels) / ds
        self.domain_cls_acc["train" if self.training else "eval"](domain_logits.view(-1, 2).float(), domain_labels, mask=(domain_labels != -1))
        self.domain_cls_loss_avg["train" if self.training else "eval"](loss.item())

        answer_type_logits = self.classifier(hidden, ex_hidden)
        _, answer_type_pred = answer_type_logits.max(dim=-1)

        # if self.regularize_prob == 1:
        #     detached_type_logits = answer_type_logits.detach()
        #     expanded_domain_logits = []
        #     for d in range(domain_logits.size(0)):
        #         expanded_domain_logits.append(domain_logits[d].detach().unsqueeze(0).expand(self.domain_lens[d], -1, -1, -1))
        #     detached_type_logits[:, :, :, 0] += torch.cat(expanded_domain_logits, dim=0)[:, :, :, 0]
        #     _, answer_type_pred = detached_type_logits.max(dim=-1)
        # elif self.regularize_prob == 2:
        #     detached_type_logits = answer_type_logits.detach()
        #     expanded_domain_logits = []
        #     for d in range(domain_logits.size(0)):
        #         expanded_domain_logits.append(domain_logits[d].detach().unsqueeze(0).expand(self.domain_lens[d], -1, -1, -1))
        #     prob_scale = torch.softmax(torch.cat(expanded_domain_logits, dim=0), dim=-1)
        #     detached_type_logits[:, :, :, 0] *= prob_scale[:, :, :, 0]
        #     detached_type_logits[:, :, :, 1:] *= prob_scale[:, :, :, 1].unsqueeze(-1)
        #     _, answer_type_pred = detached_type_logits.max(dim=-1)

        # Label (slot-value) encoding
        # loss = 0
        loss_slot = []
        pred_slot = []
        output = []
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
                loss += _loss

            if answer_type_ids is not None:
                cls_loss = self.nll(answer_type_logits[s].view(ds * ts, -1), answer_type_ids[:, :, s].view(-1)) / ds
                loss_slot[-1] += cls_loss.item()
                loss += cls_loss

        if labels is None:
            return output

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
                0), torch.cat([
                answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1).unsqueeze(0)

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


    def get_metric(self, reset=False):
        if self.training:
            metric = {
                "Acc_domain_cls": self.domain_cls_acc["train"].get_metric(reset=reset),
                "Loss_domain_cls": self.domain_cls_loss_avg["train"].get_metric(reset=reset)
            }
        else:
            metric = {
                "domain_cls_acc": self.domain_cls_acc["eval"].get_metric(reset=reset),
                "Loss_domain_cls": self.domain_cls_loss_avg["eval"].get_metric(reset=reset)
            }
        return metric
