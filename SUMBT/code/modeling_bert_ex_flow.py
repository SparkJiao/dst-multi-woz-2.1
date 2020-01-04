# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math

import torch
from pytorch_pretrained_bert.modeling import BertLayerNorm, BertPooler, \
    BertPreTrainedModel, BertIntermediate, BertOutput, BertSelfOutput
from pytorch_pretrained_bert.modeling import BertConfig
from torch import nn

# from .file_utils import cached_path
try:
    from .global_logger import get_child_logger
except ImportError:
    from global_logger import get_child_logger

logger = get_child_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, start_offset=0):
        """
        start_offset: offset for position ids
        """
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) + start_offset
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
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

    def forward(self, hidden_states, attention_mask, attn_cache=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        new_attn_cache = {
            "key": mixed_key_layer,
            "value": mixed_value_layer
        }

        if attn_cache is not None:
            mixed_key_layer = torch.cat([attn_cache["key"], mixed_key_layer], dim=1)
            mixed_value_layer = torch.cat([attn_cache["value"], mixed_value_layer], dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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
        return context_layer, new_attn_cache


class BertReshapeAttention(nn.Module):
    def __init__(self, config):
        super(BertReshapeAttention, self).__init__()
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

        self.key_type = config.key_type
        logger.info(f"Key type is {self.key_type}")
        if self.key_type in [2, 3]:
            self.extra_key = nn.Linear(self.attention_head_size, self.attention_head_size)
        elif self.key_type in [4, 5]:
            self.extra_key = copy.deepcopy(self.key)
            self.extra_value = copy.deepcopy(self.value)

    def copy_weight(self):
        self.extra_key = copy.deepcopy(self.key)
        self.extra_value = copy.deepcopy(self.value)

    def share_weight(self):
        self.extra_key.weight = self.key.weight
        self.extra_value.weight = self.value.weight

    def full_share(self):
        self.extra_key = self.key
        self.extra_value = self.value

    def from_scratch(self):
        pass

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def back_transpose_for_scores(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_x_shape)

    def transpose_for_slot(self, x, slot_dim):
        full_size, num_head, seq_len, h = x.size()  # (num_head, seq_len, h) / (num_head, len1, len2)
        assert num_head in [self.num_attention_heads, 1]
        bs = full_size // slot_dim
        new_shape = (slot_dim, bs, num_head, seq_len, h)
        x = x.view(*new_shape).permute(1, 2, 0, 3, 4)
        new_shape = (bs, num_head, slot_dim * seq_len, h)
        return x.reshape(*new_shape)

    @staticmethod
    def back_transpose_for_slot(x, slot_dim):
        bs, num_head, total_len, h = x.size()  # (bs, num_head, slot_dim * len1, len2 / h)
        seq_len = total_len // slot_dim
        new_shape = (bs, num_head, slot_dim, seq_len, h)
        x = x.view(*new_shape).permute(2, 0, 1, 3, 4)
        new_shape = (slot_dim * bs, num_head, seq_len, h)
        return x.reshape(*new_shape)

    def forward(self, hidden_states, attention_mask, attn_cache=None, slot_dim=0, slot_unified_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # if attn_cache is not None:
        #     mixed_value_layer = torch.cat([attn_cache["value"], mixed_value_layer], dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        new_attn_cache = {
            "key": key_layer,
            "value": value_layer
        }

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        single_dim_value = []
        single_dim_len = 0
        if attn_cache is not None:
            """
            :cache_key: (bs, num_head, seq_len, h)
            :cache_value: (bs, num_head, seq_len, h)
            :query_layer: (slot_dim * bs, num_head, slot_len, h)
            :value_layer: (bs, num_head, seq_len + slot_len, h)
            :attention_mask: (slot_dim * bs, 1, 1, seq_len + slot_len)
            :slot_unified_mask: (slot_dim * bs, 1, 1, slot_dim * slot_len)
            :attention_scores: (slot_dim * bs, num_head, slot_len, ...)
            
            While calculating attention scores, since for each token in all slots, the process of calculation is independent
            so I transpose them into single dimension and transpose them back while normalization.
            I split the `value` as several segments since the weighted sum is done in different dimension for them.
            After the calculation, sum the sums for all segments.
            This method will have little drop on the time of forwarding, but will also save much memory.
            """
            cache_key = attn_cache["key"]
            bs, num_head, seq_len, h = cache_key.size()
            slot_len = query_layer.size(2)
            total_len = slot_dim * slot_len
            # query_layer = query_layer.view(slot_dim, bs, num_head, slot_len, h).permute(1, 2, 0, 3, 4)
            # query_layer = query_layer.reshape(bs, num_head, total_len, h)
            query_layer = self.transpose_for_slot(query_layer, slot_dim)
            cache_scores = torch.matmul(query_layer, cache_key.transpose(-1, -2))  # (bs, num_head, slot_dim * slot_len, seq_len)

            extra_scores = []
            if self.key_type != 0:
                assert slot_unified_mask.size() == (slot_dim * bs, 1, 1, total_len)
                # (bs, num_head, slot_dim * slot_len, h)
                ini_mask = attention_mask.view(slot_dim, bs, 1, 1, seq_len + slot_len)[0, :, :, :, :seq_len]
                seq_att_value = nn.Softmax(dim=-1)(
                    cache_scores / math.sqrt(self.attention_head_size) + ini_mask
                ).matmul(attn_cache["value"])
                # (bs, num_head, slot_dim * slot_len, slot_dim * slot_len)
                # if self.key_type == 3:
                #     seq_att_value = seq_att_value.detach()
                if self.key_type == 1:
                    seq_value_scores = query_layer.matmul(seq_att_value.transpose(-1, -2))
                elif self.key_type in [2, 3]:
                    if self.key_type == 3:
                        seq_att_value = seq_att_value.detach()
                    seq_value_scores = query_layer.matmul(self.extra_key(seq_att_value).transpose(-1, -2))
                elif self.key_type in [4, 5]:
                    if self.key_type == 5:
                        seq_att_value = seq_att_value.detach()
                    seq_att_value_new_key = self.transpose_for_scores(self.extra_key(self.back_transpose_for_scores(seq_att_value)))
                    seq_value_scores = query_layer.matmul(seq_att_value_new_key.transpose(-1, -2))
                    seq_att_value = self.transpose_for_scores(self.extra_value(self.back_transpose_for_scores(seq_att_value)))
                else:
                    raise RuntimeError("Wrong key type !", self.key_type)
                # seq_value_scores = seq_value_scores.view(bs, num_head, slot_dim, slot_len, total_len).permute(2, 0, 1, 3, 4)
                # extra_scores.append(seq_value_scores.reshape(-1, num_head, slot_len, total_len))
                seq_value_scores = self.back_transpose_for_slot(seq_value_scores, slot_dim)
                extra_scores.append(seq_value_scores)
                # value_layer = torch.cat([
                #     seq_att_value.unsqueeze(0).expand(slot_dim, -1, -1, -1, -1).reshape(-1, num_head, total_len, h),
                #     value_layer], dim=2)
                attention_mask = torch.cat([slot_unified_mask, attention_mask], dim=-1)
                single_dim_value.append(seq_att_value)
                single_dim_len += total_len

            # cache_scores = cache_scores.view(bs, num_head, slot_dim, slot_len, seq_len).permute(2, 0, 1, 3, 4)
            # cache_scores = cache_scores.reshape(-1, num_head, slot_len, seq_len)
            cache_scores = self.back_transpose_for_slot(cache_scores, slot_dim)
            attention_scores = torch.cat(extra_scores + [cache_scores, attention_scores], dim=-1)
            single_dim_value.append(attn_cache["value"])
            single_dim_len += seq_len

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if attn_cache is not None:
            value_len = attention_probs.size(-1) - single_dim_len
            single_dim_value = torch.cat(single_dim_value, dim=-2)
            single_dim_probs, value_probs = torch.split(attention_probs, [single_dim_len, value_len], dim=-1)
            # single_dim_probs = single_dim_probs.reshape(slot_dim, bs, num_head, slot_len, single_dim_len).permute(1, 2, 0, 3, 4)
            # single_dim_probs = single_dim_probs.reshape(bs, num_head, slot_dim * slot_len, single_dim_len)
            single_dim_probs = self.transpose_for_slot(single_dim_probs, slot_dim)
            single_dim_context = single_dim_probs.matmul(single_dim_value)
            single_dim_context = self.back_transpose_for_slot(single_dim_context, slot_dim)
            value_context = value_probs.matmul(value_layer)
            context_layer = single_dim_context + value_context
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, new_attn_cache


class BertCacheAttention(nn.Module):
    def __init__(self, config):
        super(BertCacheAttention, self).__init__()
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

    def back_transpose_for_scores(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_x_shape)

    def transpose_for_slot(self, x, slot_dim):
        full_size, num_head, seq_len, h = x.size()  # (num_head, seq_len, h) / (num_head, len1, len2)
        assert num_head in [self.num_attention_heads, 1]
        bs = full_size // slot_dim
        new_shape = (slot_dim, bs, num_head, seq_len, h)
        x = x.view(*new_shape).permute(1, 2, 0, 3, 4)
        new_shape = (bs, num_head, slot_dim * seq_len, h)
        return x.reshape(*new_shape)

    @staticmethod
    def back_transpose_for_slot(x, slot_dim):
        bs, num_head, total_len, h = x.size()  # (bs, num_head, slot_dim * len1, len2 / h)
        seq_len = total_len // slot_dim
        new_shape = (bs, num_head, slot_dim, seq_len, h)
        x = x.view(*new_shape).permute(2, 0, 1, 3, 4)
        new_shape = (slot_dim * bs, num_head, seq_len, h)
        return x.reshape(*new_shape)

    def forward(self, hidden_states, attention_mask, attn_cache=None, slot_dim=0, extra_dropout=-1):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        new_attn_cache = [{
            "key": key_layer,
            "value": value_layer
        }]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        cache_scores = []
        cache_scores_if_tf = []
        if attn_cache is not None and attn_cache != []:

            # slot_tf_query = self.transpose_for_slot(query_layer, slot_dim)
            for cache in attn_cache:
                cache_key = cache["key"]
                if cache_key.size(0) != query_layer.size(0):
                    # slot attention
                    # query: (slot_dim * bs, num_head, len1, h)
                    # key & value: (bs, num_head, len2, h)
                    slot_tf_query = self.transpose_for_slot(query_layer, slot_dim)
                    # (bs, num_head, slot_dim * len1, len2)
                    tf_scores = torch.matmul(slot_tf_query, cache_key.transpose(-1, -2))
                    # (slot_dim * bs, num_head, len1, len2)
                    cache_scores.append(self.back_transpose_for_slot(tf_scores, slot_dim))
                    cache_scores_if_tf.append(1)
                else:
                    # equal size attention
                    # query & key & value: (slot_dim * bs, num_head, len1, len2, h)
                    # logger.warning("reach here")
                    # logger.warn(cache_key.size())
                    # logger.warn(query_layer.size())
                    cache_scores.append(query_layer.matmul(cache_key.transpose(-1, -2)))
                    cache_scores_if_tf.append(0)

        if cache_scores:
            attention_scores = torch.cat(cache_scores + [attention_scores], dim=-1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # the size of mask should be expand before forward.

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (slot_dim * bs, num_head, len1, \sum len2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        start_len = 0
        init_prob = attention_probs[:, :, :, -key_layer.size(-2):]
        context_layer = torch.matmul(init_prob, value_layer)
        for cache_idx, cache_score in enumerate(cache_scores):
            end_len = start_len + cache_score.size(-1)
            cache_prob = attention_probs[:, :, :, start_len:end_len]  # ((slot_dim *) bs, num_head, len1, len2)
            if cache_idx == 0 and extra_dropout > 0 and self.training:
                cache_prob = nn.functional.dropout(cache_prob, p=extra_dropout, training=self.training)
            start_len += cache_score.size(-1)
            if cache_scores_if_tf[cache_idx] == 0:
                cache_context = cache_prob.matmul(attn_cache[cache_idx]["value"])
                context_layer = context_layer + cache_context
            else:
                tf_cache_prob = self.transpose_for_slot(cache_prob, slot_dim)
                cache_context = tf_cache_prob.matmul(attn_cache[cache_idx]["value"])
                context_layer = context_layer + self.back_transpose_for_slot(cache_context, slot_dim)
        assert start_len == attention_scores.size(-1) - key_layer.size(-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, new_attn_cache


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        if config.self_attention_type == 0:
            self.self = BertSelfAttention(config)
        elif config.self_attention_type == 1:
            self.self = BertReshapeAttention(config)
        elif config.self_attention_type == 2:
            self.self = BertCacheAttention(config)
        else:
            raise RuntimeError()
        logger.info(f"Use {self.self.__class__.__name__} as self attention module of BERT")
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attn_cache=None, **kwargs):
        self_output, attn_cache = self.self(input_tensor, attention_mask, attn_cache=attn_cache, **kwargs)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attn_cache


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)

        if not hasattr(config, "slot_attention_type"):
            self.slot_attention_type = -1
        else:
            self.slot_attention_type = config.slot_attention_type
        if self.slot_attention_type != -1:
            self.slot_attention = BertAttention(config)

        if not hasattr(config, "use_flow"):
            self.use_flow = False
        else:
            self.use_flow = config.use_flow
        logger.info(f"Use flow: {self.use_flow}")
        if self.use_flow:
            if hasattr(config, "flow_head"):
                new_config = copy.deepcopy(config)
                new_config.num_attention_heads = config.flow_head
                self.flow = BertAttention(new_config)
            else:
                self.flow = BertAttention(config)
            logger.info(f"Flow head num: {self.flow.self.num_attention_heads}")

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.config = config

    def _flow_op(self, x, dialog_mask):
        """
        :param x:
        :param dialog_mask: [1, 1, ts, ts]
        :return:
        """
        ts = dialog_mask.size(-1)
        bs, seq_len, h = x.size()
        ds = bs // ts
        x = x.view(ds, ts, seq_len, h).transpose(1, 2).reshape(ds * seq_len, ts, h)
        x, _ = self.flow(x, dialog_mask)
        return x.view(ds, seq_len, ts, h).transpose(1, 2).reshape(bs, seq_len, h)

    def _extract_slot_cls(self, x):
        slot_dim = self.config.slot_dim
        cls = x[:, 0].view(slot_dim, -1, self.config.hidden_size).transpose(0, 1)
        cls_mask = cls.new_zeros(cls.size()[:-1])
        cls_mask = cls_mask[:, None, None, :]
        return cls, cls_mask

    def _inject_slot_cls(self, x, cls):
        b, slot_dim = cls.size()[:-1]
        cls = cls.transpose(0, 1).reshape(slot_dim * b, self.config.hidden_size)
        x[:, 0] = cls
        return x.contiguous()

    def slot_seq_attention(self, hidden_states):
        cls, cls_mask = self._extract_slot_cls(hidden_states)
        cls, _ = self.slot_attention(cls, cls_mask)
        hidden_states = self._inject_slot_cls(hidden_states, cls)
        return hidden_states

    def copy_weight(self):
        self.slot_attention = copy.deepcopy(self.attention)

    def share_weight(self):
        self.slot_attention.self.query.weight = self.attention.self.query.weight
        self.slot_attention.self.key.weight = self.attention.self.key.weight
        self.slot_attention.self.value.weight = self.attention.self.value.weight
        self.slot_attention.output.dense.weight = self.attention.output.dense.weight

    def full_share(self):
        self.slot_attention = self.attention

    def forward(self, hidden_states, attention_mask, attn_cache=None, do_slot_attention: bool = False,
                use_flow: bool = False, dialog_mask=None, **kwargs):
        if use_flow:
            flow_out = self._flow_op(hidden_states, dialog_mask)

        attention_output, attn_cache = self.attention(hidden_states, attention_mask, attn_cache=attn_cache, **kwargs)

        if self.slot_attention_type != -1 and do_slot_attention:
            attention_output = self.slot_seq_attention(attention_output)

        intermediate_output = self.intermediate(attention_output)

        if use_flow:
            # logger.warn("Reach flow")
            layer_output = self.output(intermediate_output, attention_output + flow_out)
        else:
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn_cache


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, all_attn_cache=None, flow_layer=100,
                dialog_position_encoding=None, dialog_position_layer_norm=None, **kwargs):
        all_encoder_layers = []
        attn_caches = []
        for layer_index, layer_module in enumerate(self.layer):
            if all_attn_cache is not None:
                attn_cache = all_attn_cache[layer_index]
            else:
                attn_cache = None
            if layer_index >= flow_layer:
                if layer_index == flow_layer and dialog_position_encoding is not None and dialog_position_layer_norm is not None:
                    hidden_states = dialog_position_layer_norm(hidden_states + dialog_position_encoding)
                use_flow = True
            else:
                use_flow = False
            hidden_states, attn_cache = layer_module(hidden_states, attention_mask, attn_cache=attn_cache, use_flow=use_flow, **kwargs)
            attn_caches.append(attn_cache)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attn_caches


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                output_all_encoded_layers=True,
                start_offset=0, all_attn_cache=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise RuntimeError(f"The attention mask should has 2 or 3 dimension, found {attention_mask.dim()}")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids=position_ids,
                                           start_offset=start_offset)
        encoded_layers, attn_caches = self.encoder(embedding_output,
                                                   extended_attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers,
                                                   all_attn_cache=all_attn_cache,
                                                   **kwargs)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, attn_caches


class DialogTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super(DialogTransformer, self).__init__(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, hidden, attention_mask=None, output_all_encoded_layers=False):
        seq_length = hidden.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden.device)
        hidden = self.LayerNorm(hidden + self.position_embeddings(position_ids).unsqueeze(0).expand_as(hidden))
        hidden = self.dropout(hidden)

        if attention_mask is None:
            attention_mask = hidden.new_ones(hidden.size()[:-1], dtype=torch.long)
        causal_mask = position_ids[None, None, :].repeat(hidden.size(0), seq_length, 1) <= position_ids[None, :, None]
        extended_mask = causal_mask[:, None, :, :].long() * attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(hidden.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        hidden, _ = self.encoder(hidden, extended_mask, output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            hidden = hidden[-1]
        return hidden


class SimpleDialogSelfAttention(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_output: bool = True, add_layer_norm: bool = False,
                 self_attention: BertSelfAttention = None):
        super(SimpleDialogSelfAttention, self).__init__(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if self_attention is not None:
            self.sa = copy.deepcopy(self_attention)
        else:
            self.sa = BertSelfAttention(config)
        # self.sa = BertSelfAttention(config)
        self.add_output = add_output
        if add_output:
            output_module_list = [nn.Linear(config.hidden_size, config.hidden_size),
                                  nn.Dropout(config.hidden_dropout_prob)]
            if add_layer_norm:
                output_module_list.append(nn.LayerNorm(config.hidden_size, eps=1e-12))
            self.sa_output = nn.Sequential(*output_module_list)
        else:
            self.sa_output = None
        self.apply(self.init_bert_weights)

    def forward(self, hidden, attention_mask=None):
        seq_length = hidden.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden.device)
        hidden = self.LayerNorm(hidden + self.position_embeddings(position_ids).unsqueeze(0).expand_as(hidden))
        hidden = self.dropout(hidden)

        if attention_mask is None:
            attention_mask = hidden.new_ones(hidden.size()[:-1], dtype=torch.long)
        causal_mask = position_ids[None, None, :].repeat(hidden.size(0), seq_length, 1) <= position_ids[None, :, None]
        extended_mask = causal_mask[:, None, :, :].long() * attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(hidden.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        hidden, _ = self.sa(hidden, extended_mask)
        if self.add_output:
            hidden = self.sa_output(hidden)

        return hidden


class SimpleSelfAttention(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_output: bool = True, add_layer_norm: bool = False):
        super(SimpleSelfAttention, self).__init__(config)

        self.sa = BertSelfAttention(config)
        self.add_output = add_output
        if add_output:
            output_module_list = [nn.Linear(config.hidden_size, config.hidden_size),
                                  nn.Dropout(config.hidden_dropout_prob)]
            if add_layer_norm:
                output_module_list.append(nn.LayerNorm(config.hidden_size, eps=1e-12))
            self.sa_output = nn.Sequential(*output_module_list)
        else:
            self.sa_output = None
        self.apply(self.init_bert_weights)

    def forward(self, hidden, attention_mask=None):

        if attention_mask is None:
            attention_mask = hidden.new_ones(hidden.size()[:-1], dtype=torch.long)
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.to(hidden.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        hidden, _ = self.sa(hidden, extended_mask)
        if self.add_output:
            hidden = self.sa_output(hidden)

        return hidden


class TransposeBertEncoder(nn.Module):
    def __init__(self, config):
        super(TransposeBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.transpose_layer = config.transpose_layer
        self.dialog_reshape: DialogReshape = config.dialog_reshape

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, all_attn_cache=None):
        all_encoder_layers = []
        attn_caches = []
        if self.transpose_layer >= 0:
            hidden_states, attention_mask = self.dialog_reshape.flatten_slot(hidden_states, attention_mask)
        for layer_index, layer_module in enumerate(self.layer):
            if all_attn_cache is not None and layer_index >= self.transpose_layer:
                attn_cache = all_attn_cache[layer_index]
            else:
                attn_cache = None
            if layer_index == self.transpose_layer:
                hidden_states, attention_mask = self.dialog_reshape.back_and_expand(hidden_states)
            hidden_states, attn_cache = layer_module(hidden_states, attention_mask, attn_cache=attn_cache)
            attn_caches.append(attn_cache)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attn_caches


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

    def forward(self, query, key, value, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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
        return context_layer, (mixed_query_layer, mixed_key_layer, mixed_value_layer)


class DialogReshape:
    def __init__(self):
        self.slot_dim = 0
        self.dialog_size = 0
        self.turn_size = 0
        self.slot_len = 0
        self.old_slot_mask = None
        self.concat_mask = None

    def set_size(self, slot_dim, dialog_size, turn_size, slot_len, concat_mask):
        self.slot_dim = slot_dim
        self.dialog_size = dialog_size
        self.turn_size = turn_size
        self.slot_len = slot_len
        self.concat_mask = concat_mask

    def flatten_slot(self, slot, slot_mask):
        """
         slot: (slot_dim, max_slot_len, h) -> (1, slot_dim * max_slot_len, h)
         slot_mask: (slot_dim, 1, x, max_slot_len) -> (1, 1, x, slot_dim * max_slot_len)
         `x` may be the size of pre-defined seq length
        """
        slot = slot.reshape(-1, slot.size(-1)).unsqueeze(0)
        a = slot_mask.size(1)
        b = slot_mask.size(2)
        self.old_slot_mask = slot_mask
        new_slot_mask = slot_mask.permute(1, 2, 0, 3).reshape(a, b, self.slot_dim * self.slot_len).unsqueeze(0)
        return slot, new_slot_mask

    def back_and_expand(self, slot):
        h = slot.size(-1)
        bs = self.dialog_size * self.turn_size
        slot = slot.reshape(self.slot_dim, self.slot_len, h).unsqueeze(1).expand(-1, bs, -1, -1).reshape(-1,
                                                                                                         self.slot_len,
                                                                                                         h)
        return slot, self.concat_mask[:, None, None, :].to(dtype=slot.dtype)
