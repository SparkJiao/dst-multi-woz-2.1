try:
    from pytorch_pretrained_bert.modeling import gelu
except ImportError:
    from transformers.activations import gelu
import copy
import math

import torch
from torch import nn


def bert_config2distill_bert_config(config):
    new_config = copy.deepcopy(config)
    new_config.n_heads = config.num_attention_heads
    new_config.dim = config.hidden_size
    new_config.attention_dropout = config.attention_probs_dropout_prob
    new_config.hidden_dim = config.intermediate_size
    new_config.activation = config.hidden_act
    new_config.output_attentions = False
    new_config.n_layers = config.num_hidden_layers
    new_config.output_hidden_states = False
    return new_config


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        if len(mask.size()) == 2:
            mask_reshp = [bs, 1, 1, k_length]
        elif len(mask.size()) == 3:
            mask_reshp = [bs, 1, mask.size(1), k_length]
        else:
            mask_reshp = list(mask.size())

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(*mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        # scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -60000.0)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ["relu", "gelu"], "activation ({}) must be in ['relu', 'gelu']".format(
            config.activation
        )
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class PreLNTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions

        assert config.dim % config.n_heads == 0

        self.context_attn = config.context_attn
        if config.context_attn:
            self.context_attention = MultiHeadSelfAttention(config)
            self.context_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.interact_attn = config.interact_attn
        if config.interact_attn:
            self.interact_attention = MultiHeadSelfAttention(config)
            self.interact_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.matching_attn = config.matching_attn
        if config.matching_attn:
            self.matching_attention = MultiHeadSelfAttention(config)
            self.matching_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        # self.attention = MultiHeadSelfAttention(config)
        # self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, utt_h, utt_mask=None, attn_mask=None, casual_mask=None):
        """
        Parameters
        ----------
        # x: torch.tensor(bs, seq_length, dim)
        x: torch.tensor(ds, ts, slot_dim, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        ds, ts, slot_dim, _ = x.size()

        def shape(x):
            return x.transpose(1, 2).reshape(ds * slot_dim, ts, -1)

        def unshape(x):
            return x.reshape(ds, slot_dim, ts, -1).transpose(1, 2).reshape(ds * ts, slot_dim, -1)

        sa_weights = []

        # Context-Attention
        if self.context_attn:
            x = shape(x)
            normed_x = self.context_layer_norm(x)
            context_sa_output = self.context_attention(query=normed_x, key=normed_x, value=normed_x, mask=casual_mask)
            if self.output_attentions:
                context_sa_output, context_sa_weights = context_sa_output
                sa_weights.append(context_sa_weights)
            else:
                context_sa_output = context_sa_output[0]
            context_sa_output = context_sa_output + x
            x = unshape(context_sa_output)

        x = x.reshape(ds * ts, slot_dim, -1)

        # Interact-Attention
        if self.interact_attn:
            normed_x = self.interact_layer_norm(x)
            interact_sa_output = self.interact_attention(query=normed_x, key=normed_x, value=normed_x, mask=attn_mask)
            if self.output_attentions:
                interact_sa_output, interact_sa_weights = interact_sa_output
                sa_weights.append(interact_sa_weights)
            else:
                interact_sa_output = interact_sa_output[0]
            x = interact_sa_output + x

        # Matching-Attention
        if self.matching_attn:
            normed_x = self.matching_layer_norm(x)
            matching_sa_output = self.matching_attention(query=normed_x, key=utt_h, value=utt_h, mask=utt_mask)
            if self.output_attentions:
                matching_sa_output, matching_sa_weights = matching_sa_output
                sa_weights.append(matching_sa_weights)
            else:
                matching_sa_output = matching_sa_output[0]
            x = matching_sa_output + x

        # Feed Forward Network
        normed_x = self.output_layer_norm(x)
        ffn_output = self.ffn(normed_x)  # (bs, seq_length, dim)
        # ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        ffn_output = ffn_output + x
        ffn_output = ffn_output.reshape(ds, ts, slot_dim, -1)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class PreLNTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = PreLNTransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, utt_h, utt_mask=None, attn_mask=None, casual_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state, utt_h=utt_h, utt_mask=utt_mask, attn_mask=attn_mask, casual_mask=casual_mask)
            hidden_state = layer_outputs[-1]

            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
