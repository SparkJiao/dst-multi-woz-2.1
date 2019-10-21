import math

import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertSelfAttention, BertSelfOutput, \
    BertIntermediate, BertOutput
from transformers.modeling_utils import prune_linear_layer


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


class TFAttention(nn.Module):
    def __init__(self, config):
        super(TFAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output1 = BertSelfOutput(config)
        self.encoder_decoder = TFEncoderDecoderAttention(config)
        self.output2 = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, encoder_output, self_attention_mask=None, encoder_attention_mask=None,
                head_mask=None):
        self_outputs = self.self(input_tensor, self_attention_mask, head_mask)
        attention_output = self.output1(self_outputs[0], input_tensor)
        seq2seq_outputs = self.encoder_decoder(attention_output, encoder_output, encoder_output,
                                               encoder_attention_mask, head_mask)
        attention_output = self.output2(seq2seq_outputs[0], attention_output)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return attention_output


class TFLayer(nn.Module):
    def __init__(self, config):
        super(TFLayer, self).__init__()
        self.attention = TFAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_states,
                self_attention_mask=None, encoder_attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, encoder_states, self_attention_mask, encoder_attention_mask,
                                          head_mask)
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
        self.layer = nn.ModuleList([TFLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, encoder_states, self_attention_mask=None, encoder_attention_mask=None,
                head_mask=None):
        all_hidden_states = ()
        # all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            # hidden_states = layer_outputs[0]
            hidden_states = layer_module(hidden_states, encoder_states, self_attention_mask, encoder_attention_mask,
                                         head_mask[i])

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
