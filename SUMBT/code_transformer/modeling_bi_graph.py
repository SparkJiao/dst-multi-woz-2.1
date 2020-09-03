import math

import torch
from torch import nn
from transformers.modeling_bert import BertSelfAttention, ACT2FN


class GraphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cls_n_head = config.cls_n_head
        self.cls_d_head = config.cls_d_head
        self.cls_d_model = self.cls_n_head * self.cls_d_head

        self.graph_q = nn.Linear(config.hidden_size, self.cls_d_model)
        self.graph_kv = nn.Linear(config.hidden_size, self.cls_d_model)
        self.graph_update = nn.Linear(self.cls_d_model, config.hidden_size)
        self.graph_residual = config.graph_residual

        if isinstance(config.hidden_act, str):
            self.graph_act_fn = ACT2FN[config.hidden_act]
        else:
            self.graph_act_fn = config.hidden_act

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden, mask=None):
        batch, seq_len, dim = hidden.size()

        q = self.graph_q(hidden).view(batch, seq_len, self.cls_n_head, self.cls_d_head)
        kv = self.graph_kv(hidden).view(batch, seq_len, self.cls_n_head, self.cls_d_head)

        scores = torch.einsum("bmhd,bnhd->bhmn", q, kv)
        scores = scores / math.sqrt(self.cls_d_head)

        if mask is not None:
            scores = scores + mask.unsqueeze(1) * -10000.0

        alpha = torch.softmax(scores, dim=-1)

        res = torch.einsum("bhmn,bnhd->bmhd", alpha, kv).reshape(batch, seq_len, dim)
        res = self.graph_update(res)
        if self.graph_residual:
            res = hidden + res
        res = self.graph_act_fn(res)
        return res


class BiGraphSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.slot_gat = GraphAttention(config)
        self.turn_gat = GraphAttention(config)

        self.fuse_f = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.mask_self = config.mask_self

        self.cls_index = None
        self.dialog_turns = None
        self.dialog_mask = None

    def set_slot_dim(self, cls_index, dialog_turns, dialog_mask):
        self.cls_index = cls_index
        self.dialog_turns = dialog_turns
        self.dialog_mask = dialog_mask

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                ):
        outputs = super().forward(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  output_attentions=output_attentions)

        hidden = outputs[0]

        # cls_h = hidden.index_select(index=self.cls_index, dim=1)
        cls_h = hidden[:, self.cls_index]
        bs, seq_len, dim = cls_h.size()

        if self.mask_self:
            mask = torch.eye(seq_len).unsqueeze(0).expand(bs, -1, -1)
        else:
            mask = None

        slot_hidden = self.slot_gat(cls_h, mask=mask)

        cls_turn_h = cls_h.reshape(-1, self.dialog_turns, seq_len, dim)
        ds = cls_turn_h.size(0)
        cls_turn_h = cls_turn_h.transpose(1, 2).reshape(ds * seq_len, self.dialog_turns, dim)

        dial_hidden = self.turn_gat(cls_turn_h,
                                    mask=self.dialog_mask.unsqueeze(0).expand(ds * seq_len, -1, -1))
        dial_hidden = dial_hidden.reshape(
            ds, seq_len, self.dialog_turns, dim).transpose(1, 2).reshape(
            bs, seq_len, dim)

        cls_h = self.fuse_f(torch.cat([cls_h, slot_hidden, dial_hidden], dim=-1))

        res = hidden.clone()
        # res = res.scatter_(index=self.cls_index, dim=1, src=cls_h)
        res[:, self.cls_index] = cls_h
        return (res,) + outputs[1:]
