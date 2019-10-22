import math

import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertSelfAttention, BertSelfOutput, \
    BertIntermediate, BertOutput
from transformers.modeling_utils import prune_linear_layer


class MultiHeadAttention(nn.Module):
    """
    Some code comes from
        transformers.model_bert.BertSelfAttention
        https://github.com/nlpyang/PreSumm/blob/master/src/models/neural.py
    """

    def __init__(self, config: BertConfig):
        super(MultiHeadAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

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

    def forward(self, query, key, value, mask=None, layer_cache=None, attn_type=None):
        """
        Multi-Head Scaled Attention
        :param query: (bs, q_len, hidden_size)
        :param key:  (bs, k_len, hidden_size)
        :param value:  (bs, k_len, hidden_size)
        :param mask:  (bs, 1, q_len, k_len), 1 for head dim
        :param layer_cache: Dict[str, Tensor]
        :param attn_type: "self" for self-attention and "context" for encoder-decoder attention
        :return: attended vectors: (bs, q_len, hidden_size) (without final linear projection)
        """
        # Project query, key, value and load cache if given
        q = self.transpose_for_scores(self.query(query))
        if layer_cache is not None:
            if attn_type == 'self':
                # (bs, num_attention_heads, k_len, attention_head_size)
                k = self.transpose_for_scores(self.key(query))
                v = self.transpose_for_scores(self.value(query))

                if layer_cache is not None:
                    device = k.device
                    if layer_cache['self_keys'] is not None:
                        k = torch.cat([layer_cache['self_keys'].to(device), k], dim=2)
                    if layer_cache['self_values'] is not None:
                        v = torch.cat([layer_cache['self_values'].to(device), v], dim=2)
                    layer_cache['self_keys'] = k
                    layer_cache['self_values'] = v
            elif attn_type == 'context':
                if layer_cache is not None:
                    if layer_cache['memory_keys'] is None:
                        k = self.transpose_for_scores(self.key(key))
                        v = self.transpose_for_scores(self.value(value))
                        layer_cache['memory_keys'] = k
                        layer_cache['memory_values'] = v
                    else:
                        k = layer_cache['memory_keys']
                        v = layer_cache['memory_values']
                else:
                    k = self.transpose_for_scores(self.key(key))
                    v = self.transpose_for_scores(self.value(value))
            else:
                raise RuntimeError(f"attn_type must in (context, self) but find {attn_type}.")
        else:
            k = self.transpose_for_scores(self.key(key))
            v = self.transpose_for_scores(self.value(value))

        # key_len = k.size(2)
        # query_len = q.size(2)

        # Calculate and scale attention scores
        q = q / math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            scores = scores + mask

        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)

        context_layer = torch.matmul(attn, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TFEncoderDecoderAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super(TFEncoderDecoderAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

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

    def forward(self, source_q, source_k, source_v, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(source_q)
        mixed_key_layer = self.key(source_k)
        mixed_value_layer = self.value(source_v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# class TFAttention(nn.Module):
#     def __init__(self, config):
#         super(TFAttention, self).__init__()
#         self.self = BertSelfAttention(config)
#         self.output1 = BertSelfOutput(config)
#         self.encoder_decoder = TFEncoderDecoderAttention(config)
#         self.output2 = BertSelfOutput(config)
#         self.pruned_heads = set()
#
#     def prune_heads(self, heads):
#         if len(heads) == 0:
#             return
#         mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
#         heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
#         for head in heads:
#             # Compute how many pruned heads are before the head and move the index accordingly
#             head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
#             mask[head] = 0
#         mask = mask.view(-1).contiguous().eq(1)
#         index = torch.arange(len(mask))[mask].long()
#
#         # Prune linear layers
#         self.self.query = prune_linear_layer(self.self.query, index)
#         self.self.key = prune_linear_layer(self.self.key, index)
#         self.self.value = prune_linear_layer(self.self.value, index)
#         self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
#
#         # Update hyper params and store pruned heads
#         self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
#         self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
#         self.pruned_heads = self.pruned_heads.union(heads)
#
#     def forward(self, input_tensor, encoder_output, self_attention_mask=None, encoder_attention_mask=None,
#                 head_mask=None):
#         self_outputs = self.self(input_tensor, self_attention_mask, head_mask)
#         attention_output = self.output1(self_outputs[0], input_tensor)
#         seq2seq_outputs = self.encoder_decoder(attention_output, encoder_output, encoder_output,
#                                                encoder_attention_mask, head_mask)
#         attention_output = self.output2(seq2seq_outputs[0], attention_output)
#         # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
#         return attention_output

class TFAttention(nn.Module):
    def __init__(self, config):
        super(TFAttention, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.output1 = BertSelfOutput(config)
        self.context_attn = MultiHeadAttention(config)
        self.output2 = BertSelfOutput(config)

    def forward(self, inputs, memory, self_attention_mask=None, encoder_attention_mask=None, layer_cache=None):
        """
        Attention in transformer decoder. The length of input sequence can be the total length,
        or 1 for predicting.
        :param inputs: input sequence (bs, 1/seq_len, hidden_size)
        :param memory: output from encoder (bs, src_len, hidden_size)
        :param self_attention_mask: subsequent and padding mask for decoder (bs, 1, 1/seq_len, seq_len)
        :param encoder_attention_mask: mask for encoder padding (bs, 1, 1, src_len)
        :param layer_cache: cache saving memory of previous decoder inputs and encoder outputs, Dict[str, Tensor]
        :return: Attention output: (bs, 1/seq_len, hidden_size)
        """
        self_output = self.self_attn(inputs, inputs, inputs,
                                      mask=self_attention_mask, layer_cache=layer_cache, attn_type='self')
        # self_outputs = self.self(input_tensor, self_attention_mask, head_mask)
        attention_output = self.output1(self_output, inputs)
        context_output = self.context_attn(attention_output, memory, memory,
                                            mask=encoder_attention_mask, layer_cache=layer_cache, attn_type='context')
        attention_output = self.output2(context_output, attention_output)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return attention_output


class TFDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TFDecoderLayer, self).__init__()
        self.attention = TFAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_states,
                self_attention_mask=None, encoder_attention_mask=None, layer_cache=None):
        attention_output = self.attention(hidden_states, encoder_states, self_attention_mask, encoder_attention_mask,
                                          layer_cache)
        # attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return layer_output


class TFDecoder(nn.Module):
    def __init__(self, config):
        super(TFDecoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TFDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, encoder_states, self_attention_mask=None, encoder_attention_mask=None, cache=None):
        """

        :param hidden_states:
        :param encoder_states:
        :param self_attention_mask:
        :param encoder_attention_mask:
        :param cache: List[Dict[str, Tensor]], len(cache) == config.num_hidden_layers
        :return:
        """
        all_hidden_states = ()
        # all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            # hidden_states = layer_outputs[0]
            hidden_states = layer_module(hidden_states, encoder_states, self_attention_mask, encoder_attention_mask,
                                         layer_cache=(cache[i] if cache is not None else None))

            # if self.output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        # if self.output_attentions:
        #     outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
