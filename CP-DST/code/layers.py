import logging
import math
import copy

import random

import torch
from pytorch_pretrained_bert.modeling import BertConfig, BertPreTrainedModel, gelu, BertSelfAttention, BertLayerNorm, ACT2FN, \
    BertIntermediate, BertOutput
from torch import distributions
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

try:
    from .global_logger import get_child_logger
except:
    from global_logger import get_child_logger

logger: logging.Logger = get_child_logger(__name__)


class ProductSimilarity(nn.Module):
    def __init__(self, input_size):
        super(ProductSimilarity, self).__init__()

    def forward(self, x1, x2):
        return x1.bmm(x2.transpose(1, 2))


class MultiClassHingeLoss(nn.Module):
    def __init__(self, margin, ignore_index=-1):
        super(MultiClassHingeLoss, self).__init__()
        self.margin = margin
        self.ignore_index = ignore_index

    def forward(self, x, target, do_mean: bool = False):
        """
        x: (batch, *, C)
        target: (batch, *)
        """
        x = x.to(dtype=torch.float)  # avoid half precision
        assert target.size() == x.size()[:-1]
        ignore_mask = (target == -1)
        target.masked_fill_(ignore_mask, 0)
        # target_score = torch.gather(x, dim=-1, index=target.unsqueeze(-1))
        target_mask = F.one_hot(target, num_classes=x.size(-1)).to(dtype=torch.uint8)
        target_score = x.masked_fill(1 - target_mask, 0).sum(dim=-1)
        masked_target = x.masked_fill(target_mask, -40000.0)
        second_max, _ = masked_target.max(dim=-1)
        loss = self.margin - target_score + second_max
        loss[loss < 0] = 0
        loss[ignore_mask] = 0
        loss = loss.sum()
        if do_mean:
            loss = loss / x.size(0)
        return loss


class ProjectionTransform(nn.Module):
    """
    Defined in https://arxiv.org/pdf/1909.05855.pdf
    """

    def __init__(self, input_dim, num_classes, act_fn=gelu):
        super(ProjectionTransform, self).__init__()
        self.transform1 = nn.Linear(input_dim, input_dim)
        self.transform2 = nn.Linear(input_dim * 2, input_dim)
        self.transform3 = nn.Linear(input_dim, num_classes)
        self.act_fn = act_fn

    def forward(self, x, y, return_h2=False):
        h1 = self.act_fn(self.transform1(x))
        h2 = self.act_fn(self.transform2(torch.cat([y, h1], dim=-1)))
        # if return_h2:
        #     return h2
        res = self.transform3(h2)
        if return_h2:
            return res, h2
        return res


class SimpleTransform(nn.Module):
    def __init__(self, input_dim, act_fn=gelu):
        super(SimpleTransform, self).__init__()
        self.transform1 = nn.Linear(input_dim, input_dim)
        self.transform2 = nn.Linear(input_dim * 2, input_dim)
        self.act_fn = act_fn

    def forward(self, x, y):
        h1 = self.act_fn(self.transform1(x))
        h2 = self.act_fn(self.transform2(torch.cat([y, h1], dim=-1)))
        return h2


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask, return_score: bool = False, hard: bool = False):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        unmasked_scores = attention_scores

        if return_score:
            return unmasked_scores, (mixed_query_layer, mixed_key_layer, mixed_value_layer)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if hard:
            if self.training:
                attention_probs = F.gumbel_softmax(attention_scores, hard=True, dim=-1)
            else:
                index = attention_scores.max(dim=-1, keepdim=True)[1]
                attention_probs = torch.zeros_like(attention_scores).scatter_(-1, index, 1.0)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, (mixed_query_layer, mixed_key_layer, mixed_value_layer, unmasked_scores)


class Attention(nn.Module):
    def __init__(self, config: BertConfig, add_output: bool = True, use_residual: bool = False, add_layer_norm: bool = False):
        super(Attention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)

        logger.info(f'Attention hyper-parameters:\n'
                    f'add_output: {add_output}, use_residual: {use_residual}, add_layer_norm: {add_layer_norm}')

        self.use_residual = use_residual
        self.add_layer_norm = add_layer_norm
        self.add_output = add_output

        if self.add_output:
            self.linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.add_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, query, key, value, attention_mask, hard=False):
        hidden, qkv = self.multi_head_attention(query, key, value, attention_mask, hard=hard)

        if self.add_output:
            hidden = self.dropout(self.linear(hidden))
        if self.add_layer_norm:
            if self.use_residual:
                hidden = query + hidden
            hidden = self.LayerNorm(hidden)
        return hidden, qkv


class DynamicFusion(nn.Module):
    def __init__(self, input_size, act_fn=gelu, gate_type=0, no_transform=False):
        super(DynamicFusion, self).__init__()
        self.no_transform = no_transform
        if not self.no_transform:
            self.transform1 = nn.Linear(input_size, input_size)
        self.fuse_f = nn.Linear(input_size * 4, input_size)
        logger.info(f'{self.__class__.__name__} parameters:\n'
                    f'Gate type: {gate_type}\n'
                    f'No transform before fusion: {no_transform}\n'
                    f'Activation function: {act_fn}')
        if gate_type == 0:
            self.gate_f = nn.Linear(input_size * 4, input_size)
        elif gate_type == 1:
            self.gate_f = nn.Linear(input_size * 4, 1)
        else:
            raise RuntimeError()
        self.act_fn = act_fn

    def forward(self, x, y, **inputs):
        """
        :param x: initial sequence
        :param y: attended sequence
        :return: fused sequence
        """

        if not self.no_transform:
            y = self.act_fn(self.transform1(y))
        z = torch.cat([x, y, x - y, x * y], dim=-1)
        gate = torch.sigmoid(self.gate_f(z))
        fusion = self.act_fn(self.fuse_f(z))
        res = gate * fusion + (1 - gate) * x
        # return res, gate.detach().mean(dim=0).mean(dim=-1).cpu()
        return res, gate


class FusionGate(nn.Module):
    def __init__(self, input_size, act_fn=gelu, gate_type=0, no_transform=False):
        super(FusionGate, self).__init__()
        self.no_transform = no_transform
        if not self.no_transform:
            self.transform1 = nn.Linear(input_size, input_size)
        logger.info(f'{self.__class__.__name__} parameters:\n'
                    f'Gate type: {gate_type}\n'
                    f'No transform before fusion: {no_transform}\n'
                    f'Activation function: {act_fn}')
        self.fuse_f = nn.Linear(input_size * 4, input_size)
        if gate_type == 0:
            self.gate_f = nn.Linear(input_size * 2, input_size)
        elif gate_type == 1:
            self.gate_f = nn.Linear(input_size * 2, 1)
        else:
            raise RuntimeError()
        self.act_fn = act_fn

    def forward(self, x, y):
        """
        :param x: initial sequence
        :param y: attended sequence
        :return: fused sequence
        """
        if not self.no_transform:
            y = self.act_fn(self.transform1(y))
        z = torch.cat([x, y, x - y, x * y], dim=-1)

        f = self.act_fn(self.fuse_f(z))

        gate = torch.sigmoid(self.gate_f(torch.cat([x, f], dim=-1)))

        res = gate * f + (1 - gate) * x
        return res


class LinearSeqAttention(nn.Module):
    def __init__(self, config: BertConfig, head_num=-1, add_output=False, add_layer_norm=False):
        super(LinearSeqAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads if head_num == -1 else head_num
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.num_attention_heads)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_output = add_output
        self.add_layer_norm = add_layer_norm
        if add_output:
            self.output = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.Dropout(config.hidden_dropout_prob))
            if add_layer_norm:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, x.size(-1) // self.num_attention_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer).squeeze(-1).unsqueeze(2)
        assert query_layer.size() == (mixed_value_layer.size(0), self.num_attention_heads, 1,
                                      mixed_query_layer.size(1)), (mixed_query_layer.size(), query_layer.size())
        # logger.info(query_layer.size())
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # logger.info(value_layer.size())

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = query_layer + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # logger.info(context_layer.size())
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # logger.info(context_layer.size())

        if self.add_output:
            context_layer = self.output(context_layer)
            if self.add_layer_norm:
                context_layer = self.LayerNorm(context_layer)

        return context_layer, (mixed_query_layer, mixed_value_layer)


class DiagonalAttentionScore(nn.Module):
    """
    s_ij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, hidden_size, dropout=0.1, do_similarity=False, act_fn=torch.relu):
        super(DiagonalAttentionScore, self).__init__()
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, hidden_size, bias=False)
        if do_similarity:
            self.diagonal = Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
        else:
            self.diagonal = Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        logger.info(f'{self.__class__.__name__}.act_fn = {act_fn}')
        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1_rep = self.act_fn(self.linear(x1))
        x2_rep = self.act_fn(self.linear(x2))
        x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)

        return x1_rep.bmm(x2_rep.transpose(1, 2))


class DiagonalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, do_similarity=False, act_fn=torch.relu):
        super(DiagonalAttention, self).__init__()
        self.scoring = DiagonalAttentionScore(input_size, hidden_size, dropout=dropout, do_similarity=do_similarity, act_fn=act_fn)

    def forward(self, x1, x2, x2_mask=None, x3=None, drop_diagonal=False, return_scores=False):
        if x3 is None:
            x3 = x2

        scores = self.scoring(x1, x2)

        if x2_mask is None:
            x2_mask = x2.new_ones((x2.size(0), 1, x2.size(1)), dtype=torch.long)
        else:
            if x2_mask.dim() == 2:
                x2_mask = x2_mask.unsqueeze(1)

        if drop_diagonal:
            assert scores.size(1) == scores.size(2)
            diagonal_mask = 1 - torch.diag(x2.new_ones(x2.size(1), dtype=torch.long), diagonal=0).unsqueeze(0)
            x2_mask = diagonal_mask * x2_mask

        x2_mask = x2_mask.to(dtype=x2.dtype)
        x2_mask = (1.0 - x2_mask) * -10000.0
        masked_scores = scores + x2_mask

        alpha = torch.softmax(masked_scores, dim=-1)
        res = alpha.bmm(x3)

        if return_scores:
            return res, scores
        return res


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, act_fn=gelu, dropout: float = 0.1,
                 add_layer_norm: bool = False, add_residual: bool = False):
        super(FeedForwardNetwork, self).__init__()

        self.ff_1 = nn.Linear(input_size, hidden_size)
        self.act_fn = act_fn
        self.ff_2 = nn.Linear(hidden_size, input_size)

        self.add_layer_norm = add_layer_norm
        self.add_residual = add_residual
        if self.add_layer_norm:
            self.LayerNorm = BertLayerNorm(input_size, eps=1e-12)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.ff_2(self.act_fn(self.ff_1(x))))

        if self.add_residual:
            h = x + h
        if self.add_layer_norm:
            h = self.LayerNorm(h)

        return h


class BertFFN(BertPreTrainedModel):
    def __init__(self, config):
        super(BertFFN, self).__init__(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.config = config
        self.apply(self.init_bert_weights)

    def forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn=None):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        if act_fn is not None:
            self.act_fn = ACT2FN[act_fn]
        else:
            self.act_fn = None
        logger.info(f'MLP activation function: {act_fn}')

    def forward(self, x):
        flat_x = x.reshape(-1, x.size(-1))
        flat_x = self.linear(flat_x)
        if self.act_fn is not None:
            flat_x = self.act_fn(flat_x)
        return flat_x.view(x.size()[:-1] + (self.output_dim,))


class ActLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn=gelu, dropout=0.1, use_residual=False, use_layer_norm=False):
        super(ActLayer, self).__init__()
        self.act_fn = act_fn
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.dropout(self.linear(self.act_fn(x)))
        return h


class SimpleFusion(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn=gelu, add_layer_norm=False, dropout=0.2):
        super(SimpleFusion, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.act_fn = act_fn
        self.add_layer_norm = add_layer_norm
        if self.add_layer_norm:
            self.LayerNorm = nn.LayerNorm(output_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.linear2(self.act_fn(self.linear1(x)))
        h = self.dropout(h)
        if self.add_layer_norm:
            h = self.LayerNorm(h)
        return h


# ===============================
# Function
# ===============================


def expand_mask(mask):
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    return mask


def negative_entropy(score, target, reduction='sum'):
    """
    :param score: [B, *]
    :param target: [B, *]
    :param reduction
    :return:
    """
    dtype = score.dtype
    # log_score = torch.log_softmax(score.to(dtype=torch.float), dim=-1)
    # loss = log_score[target].sum().to(dtype=dtype)
    neg_score = 1 - score.to(dtype=torch.float).softmax(dim=-1)
    # logger.info(neg_score)
    # log_score = -torch.log(neg_score + 1e-6)
    log_score = -torch.log(neg_score + 1e-6)
    loss = log_score[target].sum().to(dtype=dtype)
    # log_score = torch.log_softmax(score, dim=-1)
    # loss = log_score[target].sum()
    if reduction == 'mean':
        loss = loss / score.size(0)
    logger.debug(loss.item())
    return loss


def masked_log_softmax_fp16(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    initial_dtype = vector.dtype
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector.float() + (mask + 1e-45).log()
        vector = vector.to(dtype=initial_dtype)
    return F.log_softmax(vector, dim=dim)


def extract_context(*hidden_states, pre_turn: int = 0, max_turn: int = 0):
    bs = hidden_states[0].size(0)
    ds = bs // max_turn
    h = hidden_states[0].size(-1)
    # key = hidden_states_key.view(ds, max_turn, -1)
    # value = hidden_states_value.view(ds, max_turn, -1)
    seq_len = hidden_states[0].size(-2)
    turn_shape = (ds, max_turn) + hidden_states[0].size()[1:]
    hidden_states = [h.view(*turn_shape) for h in hidden_states]

    context_index = torch.arange(max_turn, dtype=torch.long, device=hidden_states[0].device).unsqueeze(-1)  # (max_turn, pre_turn)
    # (max_turn, pre_turn)
    _offset = torch.arange(start=pre_turn, end=0, step=-1, dtype=torch.long, device=hidden_states[0].device).unsqueeze(0)
    context_index = context_index - _offset

    context_index[context_index < 0] = 0
    context_index = context_index.view(-1)

    if hidden_states[0].dim() == 4:
        # (ds, max_turn, seq_len, h)
        output_states = [
            h.index_select(index=context_index, dim=1).reshape(bs, pre_turn * seq_len, -1) for h in hidden_states
        ]
        # assert output_states[0].size() == (bs, pre_turn * seq_len, h)
    elif hidden_states[0].dim() == 5:
        # (ds, max_turn, num_head, seq_len, h)
        new_shape = (bs, pre_turn) + hidden_states[0].size()[2:]
        num_head = new_shape[2]
        output_states = [
            h.index_select(index=context_index, dim=1).reshape(*new_shape).transpose(1, 2).reshape(
                bs, num_head, pre_turn * seq_len, -1) for h in hidden_states
        ]
        # assert output_states[0].size() == (bs, num_head, pre_turn * seq_len, h)
    else:
        raise RuntimeError()

    # context_key = key.index_select(index=context_index, dim=1).reshape(bs, pre_turn * seq_len, -1)
    # context_value = value.index_select(index=context_index, dim=1).reshape(bs, pre_turn * seq_len, -1)
    # return context_key, context_value
    return tuple(output_states)


def get_context_mask(attention_mask, pre_turn, max_turn):
    bs = attention_mask.size(0)
    ds = bs // max_turn
    seq_len = attention_mask.size(-1)

    ini_dim = attention_mask.dim()
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask[:, None, :, :]

    context_index = torch.arange(max_turn, dtype=torch.long, device=attention_mask.device).unsqueeze(-1)  # (max_turn, pre_turn)
    # (max_turn, pre_turn)
    _offset = torch.arange(start=pre_turn, end=0, step=-1, dtype=torch.long, device=attention_mask.device).unsqueeze(0)
    context_index = context_index - _offset

    # out of context dimension will be clamped as -1
    # context_mask = (context_index.clamp(-1, 0).to(dtype=attention_mask.dtype) * 10000.0)
    context_mask = context_index.clamp(-1, 0) + 1
    # logger.info(context_mask)
    context_mask = context_mask[None, :, :, None].expand(ds, -1, -1, seq_len)
    context_mask = context_mask.reshape(bs, 1, 1, pre_turn * seq_len)

    context_index[context_index < 0] = 0
    context_index = context_index.view(-1)

    self_len = attention_mask.size(2)
    context_attention_mask = attention_mask.view(ds, max_turn, self_len, seq_len).index_select(index=context_index, dim=1)
    context_attention_mask = context_attention_mask.reshape(bs, pre_turn, self_len, seq_len).transpose(1, 2)
    context_attention_mask = context_attention_mask.reshape(bs, 1, self_len, pre_turn * seq_len)
    attention_mask = torch.cat([context_mask * context_attention_mask, attention_mask], dim=-1)

    if ini_dim == 2:
        attention_mask = attention_mask.view(bs, (pre_turn + 1) * seq_len)
    elif ini_dim == 3:
        attention_mask = attention_mask.view(bs, self_len, (pre_turn + 1) * seq_len)

    return attention_mask


def dropout_mask(x, value, mask=None, p=0.):
    prop = (1 - p) * x.new_ones(x.size(), dtype=torch.float)
    if mask is not None:
        prop[mask] = 0.
    mask = torch.bernoulli(prop)
    new_x = x.masked_fill(mask.byte(), value)
    return new_x


def get_casual_mask(seq_length, device):
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    casual_mask = position_ids[None, None, :].repeat(1, seq_length, 1) <= position_ids[None, :, None]
    return casual_mask


def get_domain_mask_5domain(mask_self: bool = True):
    slot_idx = {'attraction': [0, 1, 2], 'hotel': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'restaurant': [13, 14, 15, 16, 17, 18, 19], 'taxi': [20, 21, 22, 23], 'train': [24, 25, 26, 27, 28, 29]}
    slot_dim = 30

    domain_mask = torch.zeros((slot_dim, slot_dim))
    for d in slot_idx.values():
        d_s, d_e = d[0], d[0] + len(d)
        domain_mask[d_s:d_e, d_s:d_e] = torch.ones((len(d), len(d)))
        if mask_self:
            for x in d:
                domain_mask[x, x] = 0

    return domain_mask


def get_domain_mask_7domain(mask_self: bool = True):
    slot_idx = {
        'attraction': [0, 1, 2], 'bus': [3, 4, 5, 6], 'hospital': [7], 'hotel': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'restaurant': [18, 19, 20, 21, 22, 23, 24], 'taxi': [25, 26, 27, 28], 'train': [29, 30, 31, 32, 33, 34]
    }
    slot_dim = 35

    domain_mask = torch.zeros((slot_dim, slot_dim))
    for d in slot_idx.values():
        d_s, d_e = d[0], d[0] + len(d)
        domain_mask[d_s:d_e, d_s:d_e] = torch.ones((len(d), len(d)))
        if mask_self:
            for x in d:
                domain_mask[x, x] = 0

    return domain_mask


def get_restaurant_attraction_mask(mask_self: bool = False):
    slot_idx = {'attraction': [0, 1, 2], 'hotel': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'restaurant': [13, 14, 15, 16, 17, 18, 19], 'taxi': [20, 21, 22, 23], 'train': [24, 25, 26, 27, 28, 29]}
    slot_dim = 30

    domain_mask = torch.zeros((slot_dim, slot_dim))
    for d_name, d in slot_idx.items():
        if d_name not in ['restaurant', 'attraction']:
            continue
        d_s, d_e = d[0], d[0] + len(d)
        domain_mask[d_s:d_e, d_s:d_e] = torch.ones((len(d), len(d)))
        if mask_self:
            for x in d:
                domain_mask[x, x] = 0

    return domain_mask


def sample_one_hot(similarity, sample_num, training: bool):
    _probability = torch.softmax(similarity)
    dtype = _probability.dtype
    _probability = _probability.to(dtype=torch.float)
    if training:
        _distribution = distributions.Categorical(probs=_probability)
        _sample_index = _distribution.sample((sample_num,))
        new_shape = (sample_num,) + similarity.size()
        _sample_one_hot = F.one_hot(_sample_index, num_classes=similarity.size(-1))
        _log_prob = _distribution.log_prob(_sample_index)
        assert _log_prob.size() == new_shape[:-1], (_log_prob.size(), new_shape)
        return _sample_one_hot.to(dtype=dtype), _log_prob.to(dtype=dtype)
    else:
        _max_index = _probability.max(dim=-1, keepdim=True)[1]
        _one_hot = F.one_hot(_max_index, num_classes=_probability.size(-1))
        return _one_hot, None
