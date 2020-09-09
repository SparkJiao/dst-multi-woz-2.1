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


class RelPosGraphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cls_n_head = config.cls_n_head
        self.cls_d_head = config.cls_d_head
        self.cls_d_model = self.cls_n_head * self.cls_d_head

        self.q = nn.Parameter(torch.FloatTensor(config.hidden_size, self.cls_n_head, self.cls_d_head))
        self.kv = nn.Parameter(torch.FloatTensor(config.hidden_size, self.cls_n_head, self.cls_d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.hidden_size, self.cls_n_head, self.cls_d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.cls_n_head, self.cls_d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.cls_n_head, self.cls_d_head))

        self.graph_update = nn.Linear(self.cls_d_model, config.hidden_size)
        self.graph_residual = config.graph_residual

        if isinstance(config.hidden_act, str):
            self.graph_act_fn = ACT2FN[config.hidden_act]
        else:
            self.graph_act_fn = config.hidden_act

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        self._init_weights()

    def _init_weights(self):
        for param in [
            self.q, self.kv, self.r, self.r_r_bias, self.r_w_bias
        ]:
            param.data.normal_(mean=0.0, std=self.config.initializer_range)

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        # x = x[:, :, :, :klen]

        return x

    def forward(self, hidden, pos_emb, mask=None):
        batch, seq_len, dim = hidden.size()

        q_head_h = torch.einsum("bih,hnd->bind", hidden, self.q)
        kv_head_h = torch.einsum("bih,hnd->bind", hidden, self.kv)

        k_head_r = torch.einsum("bih,hnd->bind", pos_emb, self.r)

        ac = torch.einsum("bind,bjnd->bnij", q_head_h + self.r_w_bias, kv_head_h)

        bd = torch.einsum("bind,bjnd->bnij", q_head_h + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.size(3))

        # scores = torch.einsum("bmhd,bnhd->bhmn", q, kv)
        scores = ac + bd
        scores = scores / math.sqrt(self.cls_d_head)

        if mask is not None:
            scores = scores + mask.unsqueeze(1) * -65500.0

        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        res = torch.einsum("bhmn,bnhd->bmhd", alpha, kv_head_h).reshape(batch, seq_len, dim)
        res = self.graph_update(res)
        res = self.dropout(res)

        if self.graph_residual:
            res = hidden + res
        res = self.graph_act_fn(res)

        return res


class BiGraphSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.slot_gat = GraphAttention(config)
        # self.turn_gat = GraphAttention(config)
        self.turn_gat = RelPosGraphAttention(config)

        self.fuse_f = nn.Linear(config.hidden_size * 3, config.hidden_size)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mask_self = config.mask_self
        self.d_model = config.hidden_size

        self.cls_index = None
        self.dialog_turns = None
        self.dialog_mask = None

    def set_slot_dim(self, cls_index, dialog_turns, dialog_mask):
        self.cls_index = cls_index
        self.dialog_turns = dialog_turns
        self.dialog_mask = dialog_mask

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[None, :, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(bsz, -1, -1)

        return pos_emb

    def relative_positional_encoding(self, q_len, k_len, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        # unidirectional attention
        beg, end = k_len, -1

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        # pos_emb = pos_emb.to(self.device)
        return pos_emb

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

        turn_pos_emb = self.relative_positional_encoding(self.dialog_turns, self.dialog_turns, ds * seq_len)
        turn_pos_emb = turn_pos_emb.to(dtype=cls_turn_h.dtype, device=cls_turn_h.device)

        dial_hidden = self.turn_gat(cls_turn_h,
                                    pos_emb=turn_pos_emb,
                                    mask=self.dialog_mask.unsqueeze(0).expand(ds * seq_len, -1, -1))

        dial_hidden = dial_hidden.reshape(
            ds, seq_len, self.dialog_turns, dim).transpose(1, 2).reshape(
            bs, seq_len, dim)

        cls_h = self.fuse_f(torch.cat([cls_h, slot_hidden, dial_hidden], dim=-1))
        # cls_h = self.LayerNorm(self.dropout(cls_h))

        res = hidden.clone()
        # res = res.scatter_(index=self.cls_index, dim=1, src=cls_h)
        # res[:, self.cls_index] = cls_h.to(dtype=res.dtype)
        res[:, self.cls_index] = cls_h
        return (res,) + outputs[1:]
