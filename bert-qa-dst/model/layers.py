import torch
from torch import nn
from torch.nn import functional as F


def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def set_my_dropout_prob(p):  # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, do_residual=False, bidir=True):
        super(StackedBRNN, self).__init__()
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.do_residual = do_residual
        self.hidden_size = hidden_size

        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            if i != 0:
                if not bidir:
                    input_size = hidden_size
                else:
                    input_size = 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=bidir))

    def forward(self, x, x_mask=None, return_list=False):
        # return_list: return a list for layers of hidden vectors
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            # Apply dropout to input
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            if self.do_residual and i > 0:
                rnn_output = rnn_output + hiddens[-1]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens]

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output
