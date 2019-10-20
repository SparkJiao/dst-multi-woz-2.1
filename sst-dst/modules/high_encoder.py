import torch
from torch import nn
from transformers.modeling_bert import BertEncoder, BertConfig


class HighTransformer(nn.Module):
    def __init__(self, config: BertConfig, bidirectional: bool = False):
        super(HighTransformer, self).__init__()

        self.bidirectional = bidirectional

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.transformer = BertEncoder(config)
        # Is a LayerNorm needed here?
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.high_transformer = BertEncoder(config)

    def forward(self, x, attn_mask=None):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.position_embeddings(position_ids)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_length, -1)
            if not self.bidirectional:
                attn_mask = attn_mask.tril(diagonal=0)

        h = x + pos_embed
        h = self.high_transformer(h, attn_mask)
        return h[0]
