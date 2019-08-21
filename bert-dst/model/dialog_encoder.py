from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from utils.logger import get_child_logger

from . import layers

logger = get_child_logger(__name__)


class DialogEncoder(BertPreTrainedModel):
    def __init__(self, config, dialog_encoder: str = 'LSTM', freeze_bert: bool = True,
                 encoder_hidden_size: int = 125, encoder_num_layers: int = 1, encoder_dropout: float = 0.2):
        super(DialogEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.config = config

        self.freeze_bert = freeze_bert
        if freeze_bert:
            logger.info(f'Freeze parameters of bert in {self.__class__.__name__}')
            for p in self.parameters():
                p.requires_grad = False

        if dialog_encoder == 'lstm' or dialog_encoder == 'gru':
            self.dialog_encoder = layers.StackedRNN(config.hidden_size, encoder_hidden_size, bidirectional=False,
                                                    num_layers=encoder_num_layers, dropout=encoder_dropout, rnn_type=dialog_encoder)
        elif dialog_encoder == 'transformer':
            pass
        else:
            raise RuntimeError(f'Dialog encoder only support `lstm`, `gru` and `transformer`.')

        self.output_dim = encoder_hidden_size

        self.apply(self.init_weights)

    @staticmethod
    def flat(x):
        return x.contiguous().view(-1, x.size(-1))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, dialog_mask=None):
        batch, max_turns, max_seq_length = input_ids.size()
        # [batch * max_turns, max_seq_length]
        input_ids = self.flat(input_ids)
        token_type_ids = self.flat(token_type_ids)
        attention_mask = self.flat(token_type_ids)
        seq_output, pool_output = self.bert(input_ids, token_type_ids, attention_mask)
        seq_output = seq_output.reshape(batch, max_turns, max_seq_length, -1)
        # [batch, max_turns, h]
        dialog_hidden = seq_output[:, :, 0]
        dialog_hidden = self.dialog_encoder(dialog_hidden, dialog_mask)
        return dialog_hidden, seq_output
