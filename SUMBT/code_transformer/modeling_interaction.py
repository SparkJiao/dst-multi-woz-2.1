import math

import torch
from torch import nn
from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, BertSelfOutput, \
    find_pruneable_heads_and_indices, prune_linear_layer


class InteractionSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

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

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            # mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            # attention_mask = encoder_attention_mask
        else:
            # mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        mixed_key_layer = self.key(hidden_states)

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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class InteractionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = InteractionSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class InteractionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.interaction_attention = InteractionAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    @staticmethod
    def reshape(x, slot_dim):
        ds_sd = x.size(0)
        ds = ds_sd // slot_dim
        ts = x.size(1)
        return x.view(ds, slot_dim, ts, -1).transpose(1, 2).reshape(ds * ts, slot_dim, -1), ts

    @staticmethod
    def un_reshape(x, ts):
        ds = x.size(0) // ts
        slot_dim = x.size(1)
        return x.view(ds, ts, slot_dim, -1).transpose(1, 2).reshape(ds * slot_dim, ts, -1)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            interaction_mask=None,
            # head_mask=None,
            encoder_hidden_states=None,
            slot_dim=None,
            # encoder_attention_mask=None,
            # output_attentions=False,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        inter_attention_input, ts = self.reshape(attention_output, slot_dim)
        inter_attention_outputs = self.interaction_attention(inter_attention_input, interaction_mask,
                                                             encoder_hidden_states=encoder_hidden_states)
        attention_output = self.un_reshape(inter_attention_outputs[0], ts)
        outputs = outputs + inter_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class InteractionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([InteractionLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            interaction_mask=None,
            # head_mask=None,
            encoder_hidden_states=None,
            slot_dim=None,
            # encoder_attention_mask=None,
            # output_attentions=False,
            # output_hidden_states=False,
    ):
        # all_hidden_states = ()
        # all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, slot_dim)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    interaction_mask,
                    encoder_hidden_states,
                    # head_mask[i],
                    # encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    interaction_mask,
                    encoder_hidden_states,
                    slot_dim,
                    # head_mask[i],
                    # encoder_attention_mask,
                    # output_attentions,
                )
            hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if output_hidden_states:
        #     outputs = outputs + (all_hidden_states,)
        # if output_attentions:
        #     outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
