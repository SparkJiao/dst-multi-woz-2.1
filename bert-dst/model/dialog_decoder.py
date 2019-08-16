import torch
from torch import nn

from . import layers
from utils.logger import get_child_logger

logger = get_child_logger(__name__)


class DialogDecoder(nn.Module):
    def __init__(self, embedding, decoder_name, decoder_input_size, decoder_hidden_size, decoder_dropout, vocab_size):
        super(DialogDecoder, self).__init__()
        self.embedding = embedding
        if decoder_name == 'lstm' or decoder_name == 'gru':
            self.decoder = layers.StackedRNN(decoder_input_size, decoder_hidden_size, bidirectional=False,
                                             dropout=decoder_dropout, rnn_type=decoder_name)
        else:
            raise RuntimeError('Decoder type only supports `lstm` and `gru`.')


    def forward(self, x):
        pass
