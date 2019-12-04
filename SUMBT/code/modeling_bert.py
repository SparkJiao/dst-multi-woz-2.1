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

logger = logging.getLogger(__name__)


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

    def forward(self, input_ids, token_type_ids=None, start_offset=0):
        """
        start_offset: offset for position ids
        """
        seq_length = input_ids.size(1)
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


# class BertSlotSelfAttention(nn.Module):
#     def __init__(self, config):
#         super(BertSlotSelfAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#
#         self.config = config
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, hidden_states, attention_mask, attn_cache=None):
#         bs, slot_len, h = hidden_states.size()
#         slot_dim = self.config.slot_dim
#         hidden_states = hidden_states.reshape(slot_dim, -1, slot_len, h).transpose(0, 1).reshape(-1, slot_len * slot_dim, h)
#         # (slot_dim * bs * slot_len
#
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)
#
#         new_attn_cache = {
#             "key": mixed_key_layer,
#             "value": mixed_value_layer
#         }
#
#         if attn_cache is not None:
#             mixed_key_layer = torch.cat([attn_cache["key"], mixed_key_layer], dim=1)
#             mixed_value_layer = torch.cat([attn_cache["value"], mixed_value_layer], dim=1)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         attention_scores = attention_scores + attention_mask
#
#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         return context_layer, new_attn_cache


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attn_cache=None):
        self_output, attn_cache = self.self(input_tensor, attention_mask, attn_cache=attn_cache)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attn_cache


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attn_cache=None):
        attention_output, attn_cache = self.attention(hidden_states, attention_mask, attn_cache=attn_cache)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attn_cache


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, all_attn_cache=None):
        all_encoder_layers = []
        attn_caches = []
        for layer_index, layer_module in enumerate(self.layer):
            if all_attn_cache is not None:
                attn_cache = all_attn_cache[layer_index]
            else:
                attn_cache = None
            hidden_states, attn_cache = layer_module(hidden_states, attention_mask, attn_cache=attn_cache)
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                start_offset=0, all_attn_cache=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, start_offset=start_offset)
        encoded_layers, attn_caches = self.encoder(embedding_output,
                                                   extended_attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers,
                                                   all_attn_cache=all_attn_cache)
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


class BertHalfEncoder(nn.Module):
    def __init__(self, config):
        super(BertHalfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, all_attn_cache=None,
                start_layer=None, end_layer=None):
        all_encoder_layers = []
        attn_caches = []

        layer_start_index = start_layer if start_layer is not None else 0
        layer_end_index = end_layer if end_layer is not None else len(self.layer)
        layers = self.layer[layer_start_index: layer_end_index]

        for layer_index, layer_module in enumerate(layers):
            if all_attn_cache is not None:
                attn_cache = all_attn_cache[layer_index]
            else:
                attn_cache = None
            hidden_states, attn_cache = layer_module(hidden_states, attention_mask, attn_cache=attn_cache)
            attn_caches.append(attn_cache)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attn_caches


class HalfBert(BertPreTrainedModel):
    def __init__(self, config):
        super(HalfBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertHalfEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                start_offset=0, all_attn_cache=None, start_layer=None, end_layer=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, start_offset=start_offset)

        encoded_layers, attn_caches = self.encoder(embedding_output,
                                                   extended_attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers,
                                                   all_attn_cache=all_attn_cache, start_layer=start_layer,
                                                   end_layer=end_layer)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, attn_caches


class HalfDialogTransformer(BertPreTrainedModel):
    def __init__(self, config, bert_encoder: BertHalfEncoder):
        super(HalfDialogTransformer, self).__init__(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        self.encoder = bert_encoder

    def forward(self, hidden, attention_mask=None, output_all_encoded_layers=False, start_layer=None, end_layer=None):
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

        hidden, _ = self.encoder(hidden, extended_mask, output_all_encoded_layers=output_all_encoded_layers,
                                 start_layer=start_layer, end_layer=end_layer)
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


# class SlotAttention(BertPreTrainedModel):
#     def __init__(self, config: BertConfig, add_output: bool = True, add_layer_norm: bool = False):
#         super(SlotAttention, self).__init__(config)
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         # Attention between slot tokens and utterance tokens
#         self.context_query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.context_key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.context_value = nn.Linear(config.hidden_size, self.all_head_size)
#         # Attention between slot tokens(Only query and key)
#         self.slot_query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.slot_key = nn.Linear(config.hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#         self.apply(self.init_bert_weights)
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, slot_hidden, context_hidden, slot_attention_mask=None, context_attention_mask=None):
#         slot_dim, bs, slot_len, _ = slot_hidden.size()
#
#         context_query = self.transpose_for_scores(self.context_query(slot_hidden))
#         context_key = self.transpose_for_scores(self.context_key(context_hidden))
#         context_value = self.transpose_for_scores(self.context_value(context_hidden))
#
#         context_attention_scores = torch.matmul(context_query, context_key.transpose(-1, -2))
#         context_attention_scores = context_attention_scores / math.sqrt(self.attention_head_size)
#         context_attention_scores = context_attention_scores + context_attention_mask
#
#         # Normalize the attention scores to probabilities.
#         context_attention_probs = nn.Softmax(dim=-1)(context_attention_scores)
#
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         context_attention_probs = self.dropout(context_attention_probs)
#
#         slot_context = torch.matmul(context_attention_probs, context_value)
#         slot_context = slot_context.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = slot_context.size()[:-2] + (self.all_head_size,)
#         slot_context = slot_context.view(*new_context_layer_shape)
#
#         return hidden


# ====================
# Function
# ====================

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
        slot = slot.reshape(self.slot_dim, self.slot_len, h).unsqueeze(1).expand(-1, bs, -1, -1).reshape(-1, self.slot_len, h)
        return slot, self.concat_mask[:, None, None, :].to(dtype=slot.dtype)
