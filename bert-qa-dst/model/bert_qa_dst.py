import torch
import json
from torch import nn
from torch.nn import functional as F
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

from util.logger import get_child_logger
from . import layers

logger = get_child_logger(__name__)


class BertDialogMatching(nn.Module):
    def __init__(self, bert_model, rnn_hidden_size, dropout):
        super(BertDialogMatching, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(p=dropout)
        config = self.bert.config
        self.rnn = layers.StackedBRNN(config.hidden_size, rnn_hidden_size, num_layers=1, rnn_type=nn.GRU, bidir=False)

        self.dis_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, input_ids, token_type_ids, input_mask, dialog_mask, value_embedding, value_ids=None, value_mask=None):
        batch, max_turns, _ = input_ids.size()
        input_ids = input_ids.reshape(batch * max_turns, -1)
        token_type_ids = token_type_ids.reshape(batch * max_turns, -1)
        input_mask = input_mask.reshape(batch * max_turns, -1)
        seq_output = self.bert(input_ids, token_type_ids, input_mask)[0]
        seq_vec = seq_output[:, 0].reshape(batch, max_turns, -1)
        dialog_hidden = self.rnn(seq_vec, dialog_mask)

        # value_dim = value_embedding.size(0)
        # value_embedding = value_embedding.unsqueeze(0).expand(batch * max_turns, -1, -1)
        _, value_dim, _ = value_embedding.size()
        value_embedding = value_embedding.unsqueeze(1).expand(-1, max_turns, -1, -1).reshape(batch * max_turns, value_dim, -1)
        logits = self.dis_metric(dialog_hidden.reshape(-1, 1, dialog_hidden.size(-1)), value_embedding).squeeze(1)

        undefined_mask = (value_ids >= value_dim)
        value_ids.masked_fill_(undefined_mask, -1)
        if value_mask is not None:
            value_mask = value_mask.unsqueeze(1).expand(-1, max_turns, -1).reshape(batch * max_turns, value_dim)
            logits = layers.masked_log_softmax(logits, value_mask, dim=-1)
            loss = F.nll_loss(logits, value_ids.reshape(-1), ignore_index=-1)
        else:
            loss = F.cross_entropy(logits, value_ids.reshape(-1), ignore_index=-1)

        logits = logits.reshape(batch, max_turns, value_dim)
        return {
            "logits": logits,
            "loss": loss
        }

    @classmethod
    def from_params(cls, _config):
        logger.info("Initialize model from params:")
        logger.info(json.dumps(_config, indent=2))
        return cls(**_config)
