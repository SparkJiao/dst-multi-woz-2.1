import torch
from torch import nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

from . import layers
from utils.logger import get_child_logger

logger = get_child_logger(__name__)


class DialogDecoder(BertPreTrainedModel):
    def __init__(self, config, decoder_name, decoder_input_size, decoder_hidden_size, decoder_dropout, vocab_size):
        super(DialogDecoder, self).__init__()
        self.bert = BertModel(config)
        for p in self.bert.parameters():
            p.requires_grad = False

        if decoder_name == 'lstm' or decoder_name == 'gru':
            self.decoder = layers.StackedRNN(config.hidden_size, decoder_hidden_size, bidirectional=False,
                                             dropout=decoder_dropout, rnn_type=decoder_name)
        else:
            raise RuntimeError('Decoder type only supports `lstm` and `gru`.')
        self.vocab_size = vocab_size
        self.classifier = nn.Linear(decoder_hidden_size, vocab_size)

        self.domain_slot_lookup = None

    def initialize_domain_slot_embeddings(self, domain_slot_input_ids):
        self.bert.eval()
        with torch.no_grad:
            domain_slot_hidden = self.bert(domain_slot_input_ids)[0]
        domain_slot_hidden = domain_slot_hidden[:, 0]
        self.domain_slot_lookup = nn.Embedding.from_pretrained(domain_slot_hidden, freeze=True)

    def forward(self, dialog_hidden, dialog_mask, target_token_ids=None):
        """
        :param dialog_hidden: [batch, max_turn, h]
        :param dialog_mask:  [batch, max_turn], 1 for true value and 0 for masked value
        :param target_token_ids:
        :return:
        """
        batch, max_turn = dialog_mask.size()

