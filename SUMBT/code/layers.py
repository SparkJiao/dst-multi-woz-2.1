import math
import logging

import torch
from pytorch_pretrained_bert.modeling import gelu
from pytorch_pretrained_bert.modeling import BertConfig
from torch import nn
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
        if return_h2:
            return h2
        vec_l = self.transform3(h2)
        return vec_l


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

    def forward(self, query, key, value, attention_mask, return_score: bool = False):
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
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.use_residual = use_residual
        self.add_layer_norm = add_layer_norm
        self.add_output = add_output

    def forward(self, query, key, value, attention_mask):
        hidden, _ = self.multi_head_attention(query, key, value, attention_mask)

        if self.add_output:
            hidden = self.dropout(self.linear(hidden))
        if self.add_layer_norm:
            if self.use_residual:
                hidden = query + hidden
            hidden = self.LayerNorm(hidden)
        return hidden


class DynamicFusion(nn.Module):
    def __init__(self, input_size, act_fn=gelu, gate_type=0):
        super(DynamicFusion, self).__init__()
        self.transform1 = nn.Linear(input_size, input_size)
        self.fuse_f = nn.Linear(input_size * 4, input_size)
        logger.info(f'{self.__class__.__name__} parameters:')
        logger.info(f'Gate type: {gate_type}')
        if gate_type == 0:
            self.gate_f = nn.Linear(input_size * 4, input_size)
        elif gate_type == 1:
            self.gate_f = nn.Linear(input_size * 4, 1)
        else:
            raise RuntimeError()
        self.act_fn = act_fn

    def forward(self, x, y):
        """
        :param x: initial sequence
        :param y: attended sequence
        :return: fused sequence
        """
        y_t = self.act_fn(self.transform1(y))
        z = torch.cat([x, y_t, x - y_t, x * y_t], dim=-1)
        gate = torch.sigmoid(self.gate_f(z))
        fusion = self.act_fn(self.fuse_f(z))
        res = gate * fusion + (1 - gate) * x
        # return res, gate.detach().mean(dim=0).mean(dim=-1).cpu()
        return res


class HierarchicalAttention(nn.Module):
    def __init__(self, config: BertConfig, add_output=False, residual=False, do_layer_norm=False,
                 add_output2=False, residual2=False, do_layer_norm2=False):
        super(HierarchicalAttention, self).__init__()
        logger.info(f'{self.__class__.__name__} parameters:')
        self.word_attention = MultiHeadAttention(config)
        self.sentence_attention = MultiHeadAttention(config)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.add_output = add_output
        self.residual = residual
        self.do_layer_norm = do_layer_norm
        logger.info(f'Num attention head: {self.num_attention_heads}')
        logger.info(f'add_output: {self.add_output}')
        logger.info(f'residual: {self.residual}')
        logger.info(f'do_layer_norm: {self.do_layer_norm}')
        if self.add_output:
            self.output = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.Dropout(config.hidden_dropout_prob))
            if self.do_layer_norm:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.add_output2 = add_output2
        self.residual2 = residual2
        self.do_layer_norm2 = do_layer_norm2
        logger.info(f'add output 2: {self.add_output2}')
        logger.info(f'residual 2: {self.residual2}')
        logger.info(f'do layer norm 2: {self.do_layer_norm2}')
        if self.add_output2:
            self.output2 = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.Dropout(config.hidden_dropout_prob))
            if self.do_layer_norm2:
                self.w_LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _fold(x, size1, size2):
        return x.reshape(x.size(0) * size1, size2, x.size(-1))

    @staticmethod
    def _unfold(x, size1, size2):
        return x.reshape(x.size(0) // size1, size1 * size2, x.size(-1))

    def forward(self, query, key, value, key_vec, word_mask, sentence_mask=None, return_sentence_score=False):
        """
        :param query: [batch, seq_len1, h]
        :param key: [batch, seq_len1, seq_len2, h]
        :param value: [batch, seq_len1, seq_len2, h]
        :param key_vec: [batch, seq_len1, h]
        :param word_mask: [batch, seq_len1, seq_len2]
        :param sentence_mask: [batch * seq_len1, seq_len1]
        :param return_sentence_score
        :return:
        """
        word_mask = (1 - word_mask.to(dtype=next(self.parameters()).dtype)) * -10000.0
        sentence_mask = (1 - sentence_mask.to(dtype=next(self.parameters()).dtype)) * -10000.0 if sentence_mask is not None else \
            key.new_zeros(key_vec.size(0) * key_vec.size(1), key_vec.size(1))

        bs, seq1, seq2, _ = key.size()
        query_seq = query.size(1)
        flat_key = key.view(bs, seq1 * seq2, -1)
        value = value.view(bs * seq1, seq2, -1)
        # (batch, num_head, q_seq, seq1 * seq2)
        scores1, (_, _, value) = self.word_attention(query, flat_key, value, attention_mask=None, return_score=True)
        scores1 = scores1.reshape(bs, self.num_attention_heads, query_seq, seq1, seq2).permute(0, 3, 1, 2, 4).reshape(
            bs * seq1, self.num_attention_heads, query_seq, seq2)
        word_mask = word_mask.reshape(bs * seq1, seq2)[:, None, None, :]
        value1 = self.transpose_for_scores(value)  # (batch * seq1, num_head, seq2, h)
        prob = self.dropout(torch.softmax(scores1 + word_mask, dim=-1))
        word_hidden = prob.matmul(value1)  # (batch * seq1, num_head, q_seq, h)
        # (batch * q_seq, seq1, h)
        word_hidden = word_hidden.permute(0, 2, 1, 3).reshape(bs, seq1, query_seq, -1).transpose(1, 2).reshape(
            bs * query_seq, seq1, -1)

        if self.add_output2:
            word_hidden = self.output2(word_hidden)
        if self.residual2:
            word_hidden = word_hidden + query.view(bs * query_seq, 1, -1)
        if self.do_layer_norm2:
            word_hidden = self.w_LayerNorm(word_hidden)

        if sentence_mask.dim() == 3:
            sentence_mask = sentence_mask[:, None, :, :]
        elif sentence_mask.dim() == 2:
            sentence_mask = sentence_mask[:, None, None, :]

        query = query.view(bs * query_seq, 1, -1)
        key_vec = key_vec.view(bs, 1, seq1, -1).expand(-1, query_seq, -1, -1).reshape(bs * query_seq, seq1, -1)
        # (bs * query_seq, 1, h) * (bs * query_seq, seq1, h)^T * (bs * query_seq, seq1, h)
        # logger.info(query.size())
        # logger.info(key_vec.size())
        # logger.info(sentence_mask.size())
        # (batch * q_seq, 1, h)
        sentence_hidden, (_, _, _, scores) = self.sentence_attention(query, key_vec, word_hidden, sentence_mask)

        if self.add_output:
            sentence_hidden = self.output(sentence_hidden)
        if self.residual:
            sentence_hidden = sentence_hidden + query
        if self.do_layer_norm:
            sentence_hidden = self.LayerNorm(sentence_hidden)
        if return_sentence_score:
            return sentence_hidden.squeeze(1), scores
        return sentence_hidden.squeeze(1)  # (batch * query_seq, h)


# ===============================
# Function
# ===============================


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
