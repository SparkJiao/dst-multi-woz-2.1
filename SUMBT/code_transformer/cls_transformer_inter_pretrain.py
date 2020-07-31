import copy
from logging import Logger

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.activations import gelu
from transformers.modeling_bert import BertEncoder
from transformers.modeling_xlm import create_sinusoidal_embeddings

try:
    from . import layers
    from .metric import Average
    from .global_logger import get_child_logger
    from .modeling_bert import BertPreTrainedModel, BertModel
    from .modeling_interaction import InteractionEncoder
except ImportError:
    import layers
    from metric import Average
    from global_logger import get_child_logger
    from modeling_bert import BertPreTrainedModel, BertModel
    from modeling_interaction import InteractionEncoder

logger: Logger = get_child_logger(__name__)

ACT2FN = {'gelu': gelu, "relu": F.relu, "tanh": F.tanh}


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, **kwargs):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, **kwargs)


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device):
        super(BeliefTracker, self).__init__()

        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.max_slot_length = args.max_slot_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        # Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir)
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob if args.dropout is None else args.dropout
        logger.info(f'Dropout prob: {self.hidden_dropout_prob}')
        if args.fix_bert:
            logger.info('Fix all parameters of bert encoder')
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        # values Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_dir)
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        logger.info(f'Value vectors embedding type: {args.value_embedding_type}')
        self.value_embedding_type = args.value_embedding_type
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # NBT
        nbt_config = copy.deepcopy(self.sv_encoder.config)
        nbt_config.num_hidden_layers = args.num_layers
        nbt_config.intermediate_size = args.intermediate_size
        nbt_config.num_attention_heads = self.attn_head
        nbt_config.gradient_checkpointing = True
        self.nbt_config = nbt_config

        self.position_embedding = self.utterance_encoder.bert.embeddings.position_embeddings
        if args.sinusoidal_embeddings:
            self.position_embedding = nn.Embedding(args.max_turn_length + 8, self.bert_output_dim)
            create_sinusoidal_embeddings(args.max_turn_length + 8, self.bert_output_dim, self.position_embedding.weight)
            assert not self.position_embedding.weight.requires_grad
        self.positionLayerNorm = nn.LayerNorm(self.bert_output_dim, eps=nbt_config.layer_norm_eps)

        self.add_interaction = args.add_interaction
        if self.add_interaction:

            self.add_query_attn = args.add_query_attn
            if args.add_query_attn:
                query_attn_config = copy.deepcopy(self.sv_encoder.config)
                query_attn_config.num_attention_heads = self.attn_head
                self.query_attn = layers.Attention(query_attn_config, add_output=False, use_residual=False,
                                                   add_layer_norm=True)
            self.transformer = InteractionEncoder(nbt_config)
        else:
            self.transformer = BertEncoder(nbt_config)

        self.project1 = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        self.project2 = nn.Linear(self.bert_output_dim, self.bert_output_dim)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        logger.info(f'Classification loss weight: {args.cls_loss_weight}')
        self.cls_loss_weight = args.cls_loss_weight

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.efficient = args.efficient

    def initialize_slot_value_lookup(self, label_ids, slot_ids, slot_token_type_ids=None):

        self.sv_encoder.eval()

        # register slot input buffer
        slot_mask = slot_ids > 0  # (slot_dim, slot_len)
        slot_dim, slot_len = slot_mask.size()
        slot_to_slot_mask = torch.zeros(slot_dim, slot_dim, slot_len)
        for i in range(slot_dim):
            slot_to_slot_mask[i, i] = slot_mask[i]
        slot_to_slot_mask = slot_to_slot_mask.unsqueeze(1).expand(-1, slot_len, -1, -1).reshape(slot_dim * slot_len,
                                                                                                slot_dim * slot_len)
        # seq_to_slot_mask = torch.zeros((self.max_seq_length, slot_len * slot_dim),
        #                                 dtype=torch.long, device=self.device)

        # utt_pos_ids = torch.arange(self.max_seq_length, device=self.device, dtype=torch.long)
        slot_pos_ids = self.max_seq_length + torch.arange(self.max_slot_length, device=self.device, dtype=torch.long)
        slot_pos_ids = slot_pos_ids.unsqueeze(0).repeat(slot_dim, 1).reshape(slot_dim * self.max_slot_length)
        # pos_ids = torch.cat([utt_pos_ids, slot_pos_ids], dim=-1)

        self.register_buffer("slot_ids", slot_ids)
        if slot_token_type_ids is None:
            slot_token_type_ids = slot_ids.new_zeros(slot_ids.size())
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask.to(dtype=torch.long))
        # self.register_buffer("slot_to_slot_mask", slot_to_slot_mask.to(dtype=torch.long, device=self.device))
        # self.register_buffer("seq_to_slot_mask", seq_to_slot_mask)
        # self.register_buffer("pos_ids", pos_ids)
        # self.register_buffer("slot_pos_ids", slot_pos_ids)

        diagonal_mask = torch.diag(torch.ones(slot_dim, device=self.device))[None, None, :, :] * -10000.0
        self.register_buffer("diagonal_mask", diagonal_mask)

        max_value_num = 0
        value_list = []

        with torch.no_grad():
            for s, label_id in enumerate(label_ids):
                label_mask = label_id > 0
                hid_label = self.sv_encoder(input_ids=label_id.view(-1, self.max_label_length),
                                            attention_mask=label_mask.view(-1, self.max_label_length))[0]
                if self.value_embedding_type == 'cls':
                    hid_label = hid_label[:, 0, :]
                else:
                    hid_label = hid_label[:, 1:-1].mean(dim=1)
                hid_label = hid_label.detach()
                # self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                # self.value_lookup[s].padding_idx = -1
                max_value_num = max(max_value_num, hid_label.size(0))
                value_list.append(hid_label)
        del self.sv_encoder
        torch.cuda.empty_cache()

        value_tensor = torch.zeros((slot_ids.size(0), max_value_num, self.bert_output_dim),
                                   dtype=value_list[0].dtype, device=self.device)
        value_mask = torch.zeros((slot_ids.size(0), max_value_num), dtype=torch.long, device=self.device)
        for s, value in enumerate(value_list):
            value_num = value.size(0)
            value_tensor[s, :value_num] = value
            value_mask[s, :value_num] = value.new_ones(value.size()[:-1], dtype=torch.long)
        value_tensor = value_tensor.half()
        self.register_buffer("defined_values", value_tensor)
        # self.register_buffer("value_mask", value_mask)
        self.register_buffer("value_mask", ((1 - value_mask).to(dtype=value_tensor.dtype) * -10000.0)[:, None, None, :])

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0:  # padding
                    break
                elif len[i, j, 1] > 0:  # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, q_valid_turn, q_slot_idx, q_input_ids, q_token_type_ids, q_attention_mask,
                key_valid_turn, key_slot_idx, k_input_ids, k_token_type_ids, k_attention_mask,
                label, n_gpu=1):
        i_ds = q_valid_turn.size(0)
        slot_dim = q_slot_idx.size(1)
        sample_num = key_valid_turn.size(1)
        ts = q_input_ids.size(1)
        total_slot_len = slot_dim * self.max_slot_length

        q_slot_idx = q_slot_idx.view(-1)
        key_slot_idx = key_slot_idx.view(-1)
        q_slot_ids = self.slot_ids.index_select(index=q_slot_idx, dim=0).reshape(i_ds, slot_dim, -1)
        k_slot_ids = self.slot_ids.index_select(index=key_slot_idx,
                                                dim=0).reshape(i_ds * sample_num, slot_dim, -1)
        slot_ids = torch.cat([q_slot_ids, k_slot_ids], dim=0)

        q_slot_type_ids = self.slot_token_type_ids.index_select(index=q_slot_idx,
                                                                dim=0).reshape(i_ds, slot_dim, self.max_slot_length)
        k_slot_type_ids = self.slot_token_type_ids.index_select(index=key_slot_idx,
                                                                dim=0).reshape(i_ds * sample_num, slot_dim, -1)
        slot_type_ids = torch.cat([q_slot_type_ids, k_slot_type_ids], dim=0)
        q_slot_mask = self.slot_mask.index_select(index=q_slot_idx, dim=0).reshape(i_ds, slot_dim, self.max_slot_length)
        k_slot_mask = self.slot_mask.index_select(index=key_slot_idx, dim=0).reshape(i_ds * sample_num, slot_dim, -1)
        slot_mask = torch.cat([q_slot_mask, k_slot_mask], dim=0)
        slot_to_slot_mask = slot_mask.new_zeros(i_ds * (sample_num + 1), slot_dim, slot_dim, self.max_slot_length)
        for i in range(slot_dim):
            slot_to_slot_mask[:, i, i] = slot_mask[:, i]
        slot_to_slot_mask = slot_to_slot_mask.unsqueeze(2).expand(-1, -1, self.max_slot_length, -1, -1
                                                                  ).reshape(-1, total_slot_len, total_slot_len)
        ds = i_ds * (sample_num + 1)
        bs = ds * ts

        # Combine input
        input_ids = torch.cat([q_input_ids, k_input_ids.view(i_ds * sample_num, ts, -1)], dim=0)
        token_type_ids = torch.cat([q_token_type_ids, k_token_type_ids.view(i_ds * sample_num, ts, -1)], dim=0)
        attention_mask = torch.cat([q_attention_mask, k_attention_mask.view(i_ds * sample_num, ts, -1)], dim=0)

        # Utterance encoding
        input_ids = input_ids.view(-1, self.max_seq_length)
        token_type_ids = token_type_ids.view(-1, self.max_seq_length)
        attention_mask = attention_mask.view(-1, self.max_seq_length)
        output = self.utterance_encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)
        seq_hidden = output[0]
        seq_kv = output[-1]
        # assert len(seq_kv) == 12

        # Slot encoding
        slot_ids = slot_ids.unsqueeze(1).expand(-1, ts, -1, -1).reshape(bs, total_slot_len)
        slot_type_ids = slot_type_ids.unsqueeze(1).expand(-1, ts, -1, -1).reshape(bs, total_slot_len)
        slot_mask = slot_to_slot_mask.unsqueeze(1).expand(-1, ts, -1, -1).reshape(bs, total_slot_len, total_slot_len)
        slot_pos_ids = torch.arange(total_slot_len, device=self.device).unsqueeze(0).expand(bs, -1)
        extended_mask = self.utterance_encoder.get_extended_attention_mask(attention_mask, slot_ids.size(),
                                                                           device=slot_ids.device)
        output = self.utterance_encoder(input_ids=slot_ids, attention_mask=slot_mask, token_type_ids=slot_type_ids,
                                        position_ids=slot_pos_ids, concat_hidden=seq_kv,
                                        concat_hidden_mask=extended_mask.expand(-1, -1, total_slot_len, -1))
        slot_hidden = output[0]

        slot_hidden = slot_hidden.view(bs, slot_dim, self.max_slot_length, self.bert_output_dim)[:, :, 0]
        slot_h = slot_hidden.view(ds, ts, slot_dim, -1).transpose(1, 2).reshape(ds * slot_dim, ts, -1)

        turn_ids = torch.arange(ts, dtype=torch.long, device=self.device)
        casual_mask = turn_ids[None, :].repeat(ts, 1) <= turn_ids[:, None]
        casual_mask = casual_mask.unsqueeze(0).expand(ds * slot_dim, -1, -1)
        casual_mask = self.utterance_encoder.get_extended_attention_mask(casual_mask, (ds * slot_dim, ts), self.device)

        turn_embedding = self.position_embedding(turn_ids).unsqueeze(0).expand(ds * slot_dim, -1, -1)
        slot_h = self.dropout(self.positionLayerNorm(turn_embedding + slot_h))

        if self.add_interaction:
            full_mask = slot_h.new_zeros(bs, 1, 1, slot_dim)
            if self.add_query_attn:
                queried_seq_h = self.query_attn(slot_hidden, seq_hidden, seq_hidden,
                                                attention_mask=extended_mask)[0]
                hidden = self.transformer(slot_h, casual_mask, full_mask, queried_seq_h, slot_dim=slot_dim)[0]
            else:
                hidden = self.transformer(slot_h, casual_mask, full_mask, slot_dim=slot_dim)[0]
        else:
            # (ds * slot_dim, ts, h)
            hidden = self.transformer(
                slot_h, casual_mask,
                head_mask=self.utterance_encoder.get_head_mask(None, self.nbt_config.num_hidden_layers))[0]

        hidden = hidden.view(ds, slot_dim, ts, -1).mean(dim=1)
        q_hidden = hidden[:i_ds].reshape(i_ds, ts, -1)
        k_hidden = hidden[i_ds:].reshape(i_ds, sample_num, ts, -1)
        q_hidden = q_hidden.gather(dim=1, index=q_valid_turn.view(i_ds, 1, 1).expand(-1, -1, self.bert_output_dim))
        k_hidden = k_hidden.gather(
            index=key_valid_turn.view(i_ds, sample_num, 1, 1).expand(-1, -1, -1, self.bert_output_dim),
            dim=2).squeeze(2)

        query = self.project1(q_hidden)
        key = self.project2(k_hidden)
        logits = query.bmm(key.transpose(1, 2)).squeeze(1)

        # print(logits.size(), label.size())

        loss = self.nll(logits, label.view(-1))
        _, pred = logits.max(dim=-1)
        acc = torch.sum(pred == label).float() / i_ds

        if n_gpu == 1:
            return loss, acc
        else:
            return loss.unsqueeze(0), acc.unsqueeze(0)

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



