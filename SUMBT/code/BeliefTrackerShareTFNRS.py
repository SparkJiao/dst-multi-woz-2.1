import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss

try:
    from . import layers
    from .modeling_bert import BertModel, DialogTransformer
except ImportError:
    import layers
    from modeling_bert import BertModel, DialogTransformer


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config, reduce_layers: int = 0):
        super(BertForUtteranceEncoding, self).__init__(config)
        print(f'Reduce {reduce_layers} of BERT.')
        config.num_hidden_layers = config.num_hidden_layers - reduce_layers
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class BeliefTracker(nn.Module):
    def __init__(self, args, device):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.max_seq_length = args.max_seq_length
        self.max_sample_length = args.max_sample_length

        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'), reduce_layers=args.reduce_layers
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False
        if args.fix_bert:
            print('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.bert.parameters():
                p.requires_grad = False

        ### values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        # NBT
        nbt_config = self.sv_encoder.config
        nbt_config.num_attention_heads = self.attn_head
        nbt_config.num_hidden_layers = self.rnn_num_layers
        nbt_config.intermediate_size = 2048
        self.transformer = DialogTransformer(nbt_config)
        self.transformer.position_embeddings.weight = self.utterance_encoder.bert.embeddings.position_embeddings.weight

        ### Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        elif self.distance_metric == 'product':
            self.metric = layers.ProductSimilarity(self.bert_output_dim)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _encode_options(self, option_input_ids, option_token_type_ids, option_input_mask):
        op_num = option_input_ids.size(1)
        bs = option_input_ids.size(0)

        with torch.no_grad():
            op_hidden, _, _ = self.sv_encoder(option_input_ids.reshape(-1, self.max_sample_length),
                                              option_token_type_ids.reshape(-1, self.max_sample_length),
                                              option_input_mask.reshape(-1, self.max_sample_length))

        op_hidden = op_hidden[:, 0].reshape(bs, op_num, self.bert_output_dim)
        return op_hidden

    def forward(self, dialog_input_ids, dialog_token_type_ids, dialog_input_mask, dialog_mask,
                option_input_ids, option_token_type_ids, option_input_mask, label):

        ds = dialog_input_ids.size(0)
        ts = dialog_input_ids.size(1)
        bs = ds * ts
        dialog_input_ids = dialog_input_ids.reshape(-1, self.max_seq_length)
        dialog_token_type_ids = dialog_token_type_ids.reshape(-1, self.max_seq_length)
        dialog_input_mask = dialog_input_mask.reshape(-1, self.max_seq_length)
        flat_hidden, _, _ = self.utterance_encoder(dialog_input_ids, dialog_token_type_ids, dialog_input_mask,
                                                   output_all_encoded_layers=False)
        hidden = flat_hidden[:, 0].reshape(ds, ts, self.bert_output_dim)
        hidden = self.transformer(hidden, dialog_mask, False)
        last_index = dialog_mask.sum(dim=1) - 1
        last_index = last_index.unsqueeze(1).expand(-1, self.bert_output_dim).unsqueeze(1)
        hidden = hidden.gather(dim=1, index=last_index)

        op_hidden = self._encode_options(option_input_ids, option_token_type_ids, option_input_mask)

        logits = self.metric(hidden, op_hidden).squeeze(1)
        loss = self.nll(logits, label)

        return {
            'loss': loss,
            'logits': logits
        }

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)
