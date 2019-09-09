import torch
from torch import nn
from torch.nn import functional as F
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

from util.logger import get_child_logger
from . import layers

logger = get_child_logger(__name__)


class BertDialogMatching(BertPreTrainedModel):
    def __init__(self, config, rnn_hidden_size, dropout):
        super(BertDialogMatching, self).__init__(config)
        self.bert = BertModel(config)

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(p=dropout)
        self.rnn = layers.StackedBRNN(config.hidden_size, rnn_hidden_size, num_layers=1, rnn_type=nn.GRU, bidir=False)

        self.dis_metric = None

    def forward(self, input_ids, token_type_ids, input_mask, dialog_mask, value_embedding, value_ids=None):
        batch, max_turns, _ = input_ids.size()
        input_ids = input_ids.reshape(batch * max_turns, -1)
        token_type_ids = token_type_ids.reshape(batch * max_turns, -1)
        input_mask = input_mask.reshape(batch * max_turns, -1)
        seq_output = self.bert(input_ids, token_type_ids, input_mask)[0]
        seq_vec = seq_output[:, 0].reshape(batch, max_turns, -1)
        dialog_hidden = self.rnn(seq_vec, dialog_mask)

        value_dim = value_embedding.size(0)
        value_embedding = value_embedding.unsqueeze(0).reshape(batch * max_turns, value_dim, -1)
        logits = self.dis_metric(dialog_hidden.reshape(-1, 1, dialog_hidden.size(-1)), value_embedding)

        undefined_mask = (value_ids >= value_dim)
        value_ids.masked_fill_(undefined_mask, -1)
        loss = F.cross_entropy(logits, value_ids, ignore_index=-1)

        return {
            "logits": logits,
            "loss": loss
        }

    @classmethod
    def from_params(cls, _config):
        bert_model_path = _config.pop("bert_model")
        return cls.from_pretrained(pretrained_model_name_or_path=bert_model_path, **_config)
