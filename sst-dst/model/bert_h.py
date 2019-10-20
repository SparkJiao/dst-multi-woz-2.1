from torch import nn
from torch import nn
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertPreTrainingHeads

from modules.high_encoder import HighTransformer
from modules.transformer_decoder import TFDecoder


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
        self.decoder = TFDecoder(self.config)
        self._tie_or_clone_weights(self.de_word_embedding, self.bert.embeddings.word_embeddings)
        self._tie_or_clone_weights(self.de_position_embedding, self.bert.embeddings.position_embeddings)
        self.cls = BertPreTrainingHeads(config)
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

        self.init_weights()

    def forward(self, src, tgt, src_mask=None, sent_mask=None):
        pass
