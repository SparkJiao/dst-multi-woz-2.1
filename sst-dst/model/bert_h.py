import torch
from torch import nn
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertPreTrainingHeads

from modules import layers
from modules.high_encoder import HighTransformer
from modules.transformer_decoder import TFDecoder


# TODO:
#  1. Decide the pre-training task and correct the shape of input tensor.
#  2. Correct the location of decoder output to do pooling and use the result vector as classification pointer.

class HierarchicalGenQA(BertPreTrainedModel):
    def __init__(self, config, num_layers_high: int = 3, intermediate_size: int = 2048):
        super(HierarchicalGenQA, self).__init__(config)
        self.bert = BertModel(config)

        # High level transformer
        self.config = config
        self.config.num_hidden_layers = num_layers_high
        self.config.intermediate_size = intermediate_size
        self.high_transformer = HighTransformer(self.config, bidirectional=False)
        self._tie_or_clone_weights(self.high_transformer.position_embeddings, self.bert.embeddings.position_embeddings)

        # Decoder
        self.de_word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.de_position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.de_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.de_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder = TFDecoder(self.config)
        self.decoder_pool = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Tanh())
        self._tie_or_clone_weights(self.de_word_embedding, self.bert.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.de_position_embedding, self.bert.embeddings.position_embeddings)

        # Copy mechanism
        self.context_pointer = layers.AttentionScore(self.config.hidden_size, self.config.hidden_size // 2,
                                                     dropout=config.hidden_dropout_prob)
        self.gate = nn.Sequential(nn.Linear(self.config.hidden_size * 3, 1), nn.Sigmoid())

        # Prediction head
        self.cls = BertPreTrainingHeads(config)
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

        self.init_weights()

    def forward(self, src, tgt, src_mask, tgt_mask, src_type_ids, sent_mask, step=None, cache=None):
        """
        :param src: (batch, sent_num, src_len)
        :param tgt: (batch, (sent_num,) tgt_len)
        :param src_mask: (batch, sent_num, src_len)
        :param tgt_mask: (batch, (sent_num,) tgt_len)
        :param src_type_ids: (batch, sent_num, src_len)
        :param sent_mask: (batch, sent_num)
        :param step: decoding step, (batch)
        :param cache: save previous inputs and encoder outputs for all layers while predicting,
                List[Dict[str, Tensor]]
        :return:
        """
        # Bert Encoder
        bs, sent_num, src_len = src.size()
        flat_src = src.reshape(bs * sent_num, src_len)
        flat_src_mask = src_mask.reshape(bs * sent_num, src_len)
        flat_src_type_ids = src_type_ids.reshape(bs * sent_num, src_len)
        seq_outputs = self.bert(flat_src, flat_src_mask, flat_src_type_ids)
        seq_output = seq_outputs[0].reshape(bs, sent_num, src_len, -1)
        # pool_output = seq_outputs[1].reshape(bs, sent_num, -1)

        # Document Level Encoder
        sent_input = seq_output[:, 0]
        sent_output = self.high_transformer(sent_input, sent_mask)

        # Decoder
        dec_word_emb = self.de_word_embedding(tgt)
        assert dec_word_emb.dim() == 3
        dec_pos_emb = self.get_position_emb(tgt, step)
        dec_input_emb = self.de_embedding_dropout(self.de_layer_norm(dec_word_emb + dec_pos_emb))

        # The tgt_mask should be truncated before input
        dec_output = self.decoder(dec_input_emb, sent_output, tgt_mask, sent_mask, cache)
        dec_pool = self.decoder_pool(dec_output)

        # LM Prediction
        scores = self.cls(dec_output)

        # Copy Distribution
        seq_output = seq_output.reshape(bs, sent_num * src_len, -1)
        seq_scores = self.context_pointer(dec_pool, seq_output)
        copy_distribution = torch.gather(seq_scores, dim=-1, index=src.reshape(bs, sent_num * src_len))

        # Copy prob
        h = layers.weight_sum(seq_scores, seq_output, src_mask.reshape(bs, sent_num * src_len))
        g = self.gate(torch.cat([dec_pool, dec_input_emb, h], dim=-1))
        final_dist = (1. - g) * copy_distribution + g * scores

        return final_dist

    def get_position_emb(self, inputs, step=None):
        if step is None:
            seq_len = inputs.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputs)
            return self.de_position_embedding(position_ids)
        else:
            return self.de_position_embedding(step).unsqueeze(1)  # (bs, 1, hidden_size)
