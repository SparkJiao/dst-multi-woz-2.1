import torch
from torch import nn
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.models import LanguageModel


def seq_dropout(x, p=0., training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1, bidirectional=True, dropout=0.2, rnn_type='lstm'):
        super(StackedRNN, self).__init__()
        self.dropout = dropout
        self.rnn_list = nn.ModuleList()
        if rnn_type == 'lstm':
            rnn = nn.LSTM
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise RuntimeError(f'Don\'t support rnn type with {rnn_type}.')
        self.rnn_list.append(
            PytorchSeq2SeqWrapper(rnn(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)))

        self.num_layers = num_layers
        hidden_input_size = hidden_size * 2 if bidirectional else hidden_size
        for _ in range(1, num_layers):
            self.rnn_list.append(PytorchSeq2SeqWrapper(
                rnn(hidden_input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
            ))
        self.dropout = dropout

    def forward(self, x, x_mask=None, concat=True):
        output = [x]
        for i in range(self.num_layers):
            input_x = seq_dropout(output[-1], p=self.dropout, training=self.training)
            h = self.rnn_list[i](input_x, mask=x_mask)
            output.append(h)

        if concat:
            return torch.cat(output[1:], dim=-1)
        else:
            return output[-1]
