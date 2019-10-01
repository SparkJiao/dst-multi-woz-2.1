import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.to(vector.dtype)
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask = mask.unsqueeze(1)
            # scores = scores.masked_fill(mask == 0, -1e9)
            scores = masked_softmax(scores, mask.unsqueeze(1), dim=-1)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class BeliefTracker(nn.Module):
    def __init__(self, args, device):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_query_length = args.max_query_length

        self.attn_head = args.attn_head
        self.device = device

        # Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz')
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        # Query and Response Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        # Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        # RNN Belief Tracker
        self.nbt = None
        if args.task_name.find("gru") != -1:
            self.nbt = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.rnn_num_layers,
                              dropout=self.hidden_dropout_prob,
                              batch_first=True)
            self.init_parameter(self.nbt)
        elif args.task_name.find("lstm") != -1:
            self.nbt = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.rnn_num_layers,
                               dropout=self.hidden_dropout_prob,
                               batch_first=True)
            self.init_parameter(self.nbt)
        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob)
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        # Consider query vector into similarity computing.
        if args.use_query:
            print("Consider query vector into similarity computing.")
            self.query_map = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)
        else:
            self.query_map = None

        # Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    @staticmethod
    def _flat_tensor(tensor):
        return tensor.reshape(-1, tensor.size(-1))

    def _encoding_query_response(self, query_input_ids, query_token_type_ids, query_input_mask,
                                 sample_input_ids, sample_token_type_ids, sample_input_mask, flat: bool = True):
        batch, max_query_length = query_input_ids.size()
        sample_num = sample_input_ids.size(1)

        self.sv_encoder.eval()
        query, _ = self.sv_encoder(input_ids=self._flat_tensor(query_input_ids),
                                   token_type_ids=self._flat_tensor(query_token_type_ids),
                                   attention_mask=self._flat_tensor(query_input_mask))
        query = query[:, 0]
        response, _ = self.sv_encoder(input_ids=self._flat_tensor(sample_input_ids),
                                      token_type_ids=self._flat_tensor(sample_token_type_ids),
                                      attention_mask=self._flat_tensor(sample_input_mask))
        response = response[:, 0]
        if flat:
            return query, response
        else:
            return query, response.reshape(batch, sample_num, -1)

    def forward(self, dialog_input_ids, dialog_token_type_ids, dialog_input_mask, dialog_mask,
                query_input_ids, query_token_type_ids, query_input_mask,
                sample_input_ids, sample_token_type_ids, sample_input_mask,
                end_index, label):
        batch, max_turns, max_seq_length = dialog_input_ids.size()
        sample_num = sample_input_ids.size(1)
        dialog_hidden, _ = self.utterance_encoder(input_ids=self._flat_tensor(dialog_input_ids),
                                                  token_type_ids=self._flat_tensor(dialog_token_type_ids),
                                                  attention_mask=self._flat_tensor(dialog_input_mask))

        # dialog_hidden = dialog_hidden.reshape(batch, max_turns, max_seq_length, -1)

        query, response = self._encoding_query_response(query_input_ids, query_token_type_ids, query_input_mask,
                                                        sample_input_ids, sample_token_type_ids, sample_input_mask,
                                                        flat=False)
        extended_query = query.unsqueeze(1).expand(-1, max_turns, -1).reshape(batch * max_turns, -1)
        # [batch * max_turns, 1, h] -> [batch, max_turns, h]
        hidden = self.attn(extended_query, dialog_hidden, dialog_hidden,
                           mask=dialog_input_mask.view(-1, 1, self.max_seq_length)).reshape(batch, max_turns, -1)

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, batch, self.hidden_dim).to(self.device)  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [slot_dim*ds, turn, hidden]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers,  batch, self.hidden_dim).to(self.device)  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        else:
            raise RuntimeError(f'Wrong neural belief tracker type for {self.nbt.__class__}')

        # rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))
        index = end_index.unsqueeze(1).expand(-1, self.hidden_dim).unsqueeze(1)
        last_utt_h = rnn_out.gather(index=index, dim=1)
        last_utt_h = self.layer_norm(self.linear(self.dropout(last_utt_h)))

        if self.query_map:
            last_utt_h = self.query_map(torch.cat([query.unsqueeze(1), last_utt_h], dim=-1))

        _dist = self.metric(response, last_utt_h)
        # assert _dist.size() == (batch, sample_num), _dist.size()
        if self.distance_metric == "euclidean":
            _dist = -_dist
        output = {
            'logits': _dist
        }
        if label is not None:
            loss = self.nll(_dist, label)
            output['loss'] = loss
        # if n_gpu == 1:
        #     output = {k: v.unsqueeze(0) for k, v in output.items()}
        return output


        # hidden = rnn_out.view(slot_dim, ds, ts, -1)
        #
        # # Label (slot-value) encoding
        # loss = 0
        # loss_slot = []
        # pred_slot = []
        # output = []
        # for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
        #     # loss calculation
        #     hid_label = self.value_lookup[slot_id].weight
        #     num_slot_labels = hid_label.size(0)
        #
        #     _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
        #     _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds * ts * num_slot_labels, -1)
        #     _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
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
        #         _loss = self.nll(_dist.view(ds * ts, -1), masked_slot_labels_for_loss.view(-1))
        #         loss_slot.append(_loss.item())
        #         loss += _loss
        #
        # if labels is None:
        #     return output
        #
        # # calculate joint accuracy
        # pred_slot = torch.cat(pred_slot, 2)
        # accuracy = (pred_slot == labels).view(-1, slot_dim)
        # acc_slot = torch.sum(accuracy, 0).float() \
        #            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        # acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
        #       / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy
        #
        # if n_gpu == 1:
        #     return loss, loss_slot, acc, acc_slot, pred_slot
        # else:
        #     return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot.unsqueeze(0)

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
