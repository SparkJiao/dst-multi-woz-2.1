import math
import os.path

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss

try:
    from . import layers
    from .modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention
except ImportError:
    import layers
    from modeling_bert_extended import BertModel, SimpleDialogSelfAttention, SimpleSelfAttention


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config, reduce_layers: int = 0, self_attention_type: int = 0):
        super(BertForUtteranceEncoding, self).__init__(config)
        print(f'Reduce {reduce_layers} of BERT.')
        print(f'Self Attention Type: {self_attention_type}')
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
            print('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.bert.parameters():
                p.requires_grad = False

        # NBT
        nbt_config = self.utterance_encoder.config
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

        self.classifier = layers.ProjectionTransform(self.bert_output_dim, 3)

        self.value_num = []
        self.value_classifier = layers.ProjectionTransform(self.bert_output_dim, 1)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1, reduction='sum')

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids, slot_token_type_ids=None):

        # register slot input buffer
        slot_mask = (slot_ids > 0).to(dtype=torch.long)

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask)

        all_label_num = [label_id.size(0) for label_id in label_ids]
        max_label_num = max(all_label_num)
        all_label_ids = torch.zeros(slot_ids.size(0), max_label_num, label_ids[0].size(1), dtype=torch.long).to(self.device)
        all_label_mask = torch.zeros(slot_ids.size(0), max_label_num, label_ids[0].size(1), dtype=torch.long).to(self.device)
        for slot_dim, label_id in enumerate(label_ids):
            all_label_ids[slot_dim, :label_id.size(0)] = label_id
            all_label_mask[slot_dim, :label_id.size(0)] = label_id > 0

        all_label_ids = torch.cat([slot_ids.unsqueeze(1).expand(-1, max_label_num, -1), all_label_ids], dim=-1)
        all_label_type_ids = torch.cat([slot_ids.new_zeros((slot_ids.size(0), max_label_num, slot_ids.size(-1))),
                                        all_label_mask], dim=-1)
        all_label_mask = torch.cat([slot_mask.unsqueeze(1).expand(-1, max_label_num, -1), all_label_mask], dim=-1)

        self.register_buffer("value_ids", all_label_ids)
        self.register_buffer("value_type_ids", all_label_type_ids)
        self.register_buffer("value_mask", all_label_mask)
        self.value_num = all_label_num

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
        utt_hidden, _, _ = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                                  token_type_ids.view(-1, self.max_seq_length),
                                                  attention_mask.view(-1, self.max_seq_length),
                                                  output_all_encoded_layers=False)
        utt_hidden = utt_hidden[:, 0].view(ds, ts, -1)
        utt_hidden = self.transformer(utt_hidden, None).view(1, bs, -1).expand(slot_dim, -1, -1)

        slot_hidden, _, _ = self.utterance_encoder(self.slot_ids, token_type_ids=None, attention_mask=self.slot_mask,
                                                   output_all_encoded_layers=False)
        slot_hidden = slot_hidden[:, 0].unsqueeze(1).expand(-1, bs, -1)

        answer_type_logits = self.classifier(utt_hidden, slot_hidden)
        _, answer_type_pred = answer_type_logits.max(dim=-1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  # note: target_slots are successive
            value_num = self.value_num[s]
            value_hidden, _, _ = self.utterance_encoder(self.value_ids[s][:value_num],
                                                        token_type_ids=self.value_type_ids[s][:value_num],
                                                        attention_mask=self.value_mask[s][:value_num],
                                                        output_all_encoded_layers=False)
            value_hidden = value_hidden[:, 0].unsqueeze(0).expand(bs, -1, -1)
            _hidden = utt_hidden[s].view(bs, 1, -1).expand(-1, value_num, -1)
            value_logits = self.value_classifier(_hidden, value_hidden).squeeze(-1)

            if labels is not None:
                """ FIX_UNDEFINED: Mask all undefined values """
                undefined_value_mask = (labels[:, :, s] == value_num)
                masked_slot_labels_for_loss = labels[:, :, s].masked_fill(undefined_value_mask, -1)

                _loss = self.nll(value_logits, masked_slot_labels_for_loss.view(-1)) / (ds * 1.0)
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
