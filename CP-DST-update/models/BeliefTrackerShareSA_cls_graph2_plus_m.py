import copy
from logging import Logger

import torch
import torch.nn as nn
from allennlp.training.metrics import Average
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig, ACT2FN, BertModel as BertModelOrig
from general_util.training_utils import batch_to_device

from general_util.logger import get_child_logger
from models import layers
from models.modeling_bert_extended_f import BertModel, SimpleDialogSelfAttention

logger: Logger = get_child_logger(__name__)


class BeliefTracker(BertPreTrainedModel):
    def __init__(self, config: BertConfig, num_labels_ls,
                 hidden_dim: int,
                 attn_head,
                 self_attention_type: int,
                 extra_nbt: bool = False,
                 value_embedding_type: str = 'cls',
                 dropout: float = None,
                 fix_utterance_encoder: bool = False,
                 fix_bert: bool = False,
                 bert_path: str = 'bert-base-uncased',
                 sa_add_layer_norm: bool = False,
                 sa_add_residual: bool = False,
                 sa_no_position_embedding: bool = False,
                 override_attn: bool = False,
                 share_position_weight: bool = False,
                 extra_nbt_attn_head: int = 6,
                 override_attn_extra: bool = False,
                 sa_act_1: str = 'gelu',
                 diag_attn_hidden_scale: float = 1.0,
                 diag_attn_act: str = 'gelu',
                 diag_attn_act_fn: str = 'gelu',
                 mask_self: bool = False,
                 inter_domain: bool = False,
                 key_add_value: bool = False,
                 key_add_value_pro: bool = False,
                 mask_top_k: int = 0,
                 test_mode: int = -1,
                 graph_attn_type: int = 0,
                 graph_attn_head: int = 6,
                 graph_dropout: float = 0.1,
                 graph_add_output: bool = False,
                 graph_add_residual: bool = False,
                 graph_add_layer_norm: bool = False,
                 fuse_type: int = 0,
                 fusion_no_transform: bool = False,
                 fusion_act_fn: str = 'gelu',
                 slot_res: str = None,
                 cls_type: int = 0,
                 distance_metric: str = 'product',
                 cls_loss_weight: float = 1.0,
                 graph_add_sup: bool = False,
                 graph_value_sup: bool = False,
                 transfer_sup: bool = False,
                 save_gate: bool = False,
                 ):
        super(BeliefTracker, self).__init__(config)

        self.config = config

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels_ls
        self.num_slots = len(num_labels_ls)
        self.attn_head = attn_head

        # Utterance Encoder
        # self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
        #     os.path.join(args.bert_dir, 'bert-base-uncased.tar.gz'), reduce_layers=args.reduce_layers,
        #     self_attention_type=args.self_attention_type
        # )

        utt_enc_config = copy.deepcopy(config)
        logger.info(f'Self Attention Type of Utterance Encoder: {self_attention_type}')
        utt_enc_config.slot_attention_type = -1
        utt_enc_config.key_type = 0
        utt_enc_config.self_attention_type = self_attention_type

        self.bert = BertModel(utt_enc_config)
        self.bert_output_dim = hidden_dim
        self.hidden_dropout_prob = config.hidden_dropout_prob if dropout is None else dropout
        logger.info(f'Dropout prob: {self.hidden_dropout_prob}')
        if fix_utterance_encoder:
            for p in self.bert.pooler.parameters():
                p.requires_grad = False
        if fix_bert:
            logger.info('Fix all parameters of bert encoder')
            for p in self.bert.parameters():
                p.requires_grad = False

        # values Encoder (not trainable)
        self.sv_encoder = BertModelOrig.from_pretrained(bert_path)
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        logger.info(f'Value vectors embedding type: {value_embedding_type}')
        self.value_embedding_type = value_embedding_type
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels_ls])

        # NBT
        nbt_config = self.sv_encoder.config
        nbt_config.num_attention_heads = self.attn_head
        nbt_config.hidden_dropout_prob = self.hidden_dropout_prob
        nbt_config.slot_attention_type = -1
        nbt_config.key_type = 0
        nbt_config.self_attention_type = 0
        logger.info(f"Dialog Self Attention add layer norm: {sa_add_layer_norm}")
        logger.info(f"Dialog Self Attention add residual: {sa_add_residual}")
        last_attention = self.bert.encoder.layer[-1].attention.self
        if override_attn:
            logger.info("Override self attention from last layer of BERT")
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=sa_add_layer_norm,
                                                         add_residual=sa_add_residual,
                                                         self_attention=last_attention)
        else:
            self.transformer = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                         add_layer_norm=sa_add_layer_norm,
                                                         add_residual=sa_add_residual)
        if share_position_weight:
            logger.info("Dialog self attention will share position embeddings with BERT")
            self.transformer.position_embeddings.weight = self.bert.embeddings.position_embeddings.weight

        logger.info(f"If extra neural belief tracker: {extra_nbt}")
        self.extra_nbt = extra_nbt
        if self.extra_nbt:
            nbt_config.num_attention_heads = extra_nbt_attn_head
            self.override_attn_extra = override_attn_extra
            logger.info(f'If override self attention from last layer of BERT for extra belief tracker: {self.override_attn_extra}')
            if self.override_attn_extra:
                self.belief_tracker = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                                add_layer_norm=sa_add_layer_norm,
                                                                add_residual=sa_add_residual,
                                                                self_attention=last_attention,
                                                                no_position_embedding=sa_no_position_embedding)
            else:
                self.belief_tracker = SimpleDialogSelfAttention(nbt_config, add_output=True,
                                                                add_layer_norm=sa_add_layer_norm,
                                                                add_residual=sa_add_residual,
                                                                no_position_embedding=sa_no_position_embedding)
            if not sa_no_position_embedding and share_position_weight:
                self.belief_tracker.position_embeddings.weight = self.bert.embeddings.position_embeddings.weight

        self.sa_act_1 = sa_act_1
        logger.info(f'Dialog self attention use activation: {self.sa_act_1}')
        if sa_act_1 is not None:
            self.tf_act = layers.ActLayer(self.bert_output_dim, self.bert_output_dim, act_fn=ACT2FN[self.sa_act_1],
                                          dropout=self.hidden_dropout_prob)

        diag_attn_hidden_dim = int(diag_attn_hidden_scale * self.bert_output_dim)
        logger.info(f'Diagonal attention hidden size: {diag_attn_hidden_dim}')
        self.diag_attn_act = diag_attn_act
        self.value_attention = layers.DiagonalAttention(self.bert_output_dim, diag_attn_hidden_dim, dropout=self.hidden_dropout_prob,
                                                        act_fn=ACT2FN[diag_attn_act_fn])
        if self.diag_attn_act is not None:
            self.value_act = layers.MLP(self.bert_output_dim, self.bert_output_dim, self.diag_attn_act)

        self.mask_self = mask_self
        logger.info(f'If mask self during graph attention: {self.mask_self}')
        self.inter_domain = inter_domain
        logger.info(f'If do inter-domain graph attention: {self.inter_domain}')
        self.key_add_value = key_add_value
        logger.info(f'Slot attention key add value: {self.key_add_value}')
        self.key_add_value_pro = key_add_value_pro
        logger.info(f'Slot attention key add value with projection: {self.key_add_value_pro}')
        if self.key_add_value_pro:
            self.key_value_project = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)

        # Test
        self.mask_top_k = mask_top_k
        logger.info(f'Mask top k slot attention scores: {self.mask_top_k}')
        self.test_mode = test_mode
        logger.info(f'Test mode: {self.test_mode}')

        logger.info(f'Graph attention type: {graph_attn_type}')
        self.graph_attn_type = graph_attn_type
        if self.graph_attn_type == 0:
            self.graph_attention = layers.DiagonalAttention(self.bert_output_dim, diag_attn_hidden_dim, dropout=self.hidden_dropout_prob,
                                                            act_fn=ACT2FN[diag_attn_act_fn])
        elif self.graph_attn_type == 1:
            nbt_config.num_attention_heads = graph_attn_head
            nbt_config.attention_probs_dropout_prob = graph_dropout
            self.graph_attention = layers.Attention(nbt_config, add_output=graph_add_output,
                                                    use_residual=graph_add_residual,
                                                    add_layer_norm=graph_add_layer_norm)

        if self.diag_attn_act is not None:
            self.graph_act = layers.MLP(self.bert_output_dim, self.bert_output_dim, self.diag_attn_act)

        self.fuse_type = fuse_type
        logger.info(f'Fuse type: {self.fuse_type}')
        if self.fuse_type == 0:
            self.graph_project = layers.DynamicFusion(self.bert_output_dim, gate_type=1, no_transform=fusion_no_transform,
                                                      act_fn=ACT2FN[fusion_act_fn], test_mode=self.test_mode)
        elif self.fuse_type == 1:
            self.graph_project = layers.SimpleTransform(self.bert_output_dim)
        elif self.fuse_type == 2:
            self.graph_project = layers.FusionGate(self.bert_output_dim, gate_type=1, no_transform=fusion_no_transform,
                                                   act_fn=ACT2FN[fusion_act_fn])
        elif self.fuse_type == 3:
            self.graph_project = layers.DynamicFusionDropout(self.bert_output_dim, gate_type=1, no_transform=fusion_no_transform,
                                                             dropout=self.hidden_dropout_prob, act_fn=ACT2FN[fusion_act_fn])
        elif self.fuse_type == 4:
            self.graph_project = layers.DynamicFusion2(self.bert_output_dim, gate_type=1, no_transform=fusion_no_transform,
                                                       act_fn=ACT2FN[fusion_act_fn])
        else:
            raise RuntimeError()

        if slot_res is not None:
            self.slot_res = [int(x) for x in slot_res.split(":")]
            logger.info(f'Slot restriction: {self.slot_res}')
        else:
            self.slot_res = None

        if cls_type == 0:
            self.classifier = nn.Linear(self.bert_output_dim, 3)
        elif cls_type == 1:
            self.classifier = nn.Sequential(nn.Linear(self.bert_output_dim, self.bert_output_dim),
                                            nn.Tanh(),
                                            nn.Linear(self.bert_output_dim, 3))
        self.hidden_output = nn.Linear(self.bert_output_dim, self.bert_output_dim, bias=False)

        # Measure
        self.distance_metric = distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        elif self.distance_metric == 'product':
            self.metric = layers.ProductSimilarity(self.bert_output_dim)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        logger.info(f'Classification loss weight: {cls_loss_weight}')
        self.cls_loss_weight = cls_loss_weight
        logger.info(f"Graph scores supervision coherent: {graph_add_sup}")
        self.graph_add_sup = graph_add_sup
        logger.info(f"Graph value scores supervision coherent: {graph_value_sup}")
        self.graph_value_sup = graph_value_sup
        logger.info(f"Transfer labels supervision coherent: {transfer_sup}")
        self.transfer_sup = transfer_sup

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.init_weights()

        # Metric
        self.metrics = {
            "cls_loss": {
                "train": Average(),
                "eval": Average(),
            },
            "matching_loss": {
                "train": Average(),
                "eval": Average(),
            }
        }
        if self.graph_add_sup > 0:
            self.metrics["slot_negative_loss"] = {
                "train": Average(),
                "eval": Average()
            }
        if self.graph_value_sup > 0:
            self.metrics["graph_value_loss"] = {
                "train": Average(),
                "eval": Average()
            }
        if self.transfer_sup > 0:
            self.metrics["transfer_loss"] = {
                "train": Average()
            }

        self.save_gate = save_gate
        if self.save_gate:
            self.gate_metric = []
            self.value_scores = []
            self.graph_scores = []

    def initialize_slot_value_lookup(self, label_emb_inputs, slot_emb_inputs):

        self.sv_encoder.eval()

        # register slot input buffer
        # slot_mask = slot_ids > 0
        slot_ids = slot_emb_inputs["input_ids"]
        slot_mask = slot_emb_inputs["attention_mask"]
        slot_token_type_ids = slot_emb_inputs["token_type_ids"] if "token_type_ids" in slot_emb_inputs else None

        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("slot_token_type_ids", slot_token_type_ids)
        self.register_buffer("slot_mask", slot_mask)

        if self.inter_domain:
            if slot_ids.size(0) == 30:
                inter_domain_mask = layers.get_domain_mask_5domain(mask_self=False)
            elif slot_ids.size(0) == 35:
                inter_domain_mask = layers.get_domain_mask_7domain(mask_self=False)
            else:
                raise RuntimeError(f"Incompatible slot dim: {slot_ids.size(0)}")
            inter_domain_mask = inter_domain_mask.to(dtype=torch.float, device=self.device)
            inter_domain_mask = (1 - inter_domain_mask) * -10000.0
            self.register_buffer("inter_domain_mask", inter_domain_mask)

        if self.slot_res is not None:
            slot_dim = slot_ids.size(0)
            slot_res = torch.ones(slot_dim, slot_dim, device=self.device, dtype=torch.long)
            slot_res[:, self.slot_res] = torch.zeros(slot_dim, len(self.slot_res), device=self.device, dtype=torch.long)
            self.slot_res = slot_res

        max_value_num = 0
        value_list = []

        with torch.no_grad():
            for s, label_inputs in enumerate(label_emb_inputs):
                # label_ids = label_inputs["input_ids"].to(self.device)
                # label_type_ids = label_inputs["token_type_ids"].to(self.device) if "token_type_ids" in label_inputs else None
                # label_mask = label_inputs["attention_mask"].to(self.device)
                hid_label = self.sv_encoder(**batch_to_device(label_inputs, self.device))[0]
                if self.value_embedding_type == 'cls':
                    hid_label = hid_label[:, 0, :]
                else:
                    hid_label = hid_label[:, 1:-1].mean(dim=1)
                hid_label = hid_label.detach()
                self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
                self.value_lookup[s].padding_idx = -1
                max_value_num = max(max_value_num, hid_label.size(0))
                value_list.append(hid_label)
        self.sv_encoder = None
        # del self.sv_encoder
        torch.cuda.empty_cache()

        value_tensor = torch.zeros((slot_ids.size(0), max_value_num, self.bert_output_dim),
                                   dtype=value_list[0].dtype, device=self.device)
        value_mask = torch.zeros((slot_ids.size(0), max_value_num), dtype=torch.long, device=self.device)
        for s, value in enumerate(value_list):
            value_num = value.size(0)
            value_tensor[s, :value_num] = value
            value_mask[s, :value_num] = value.new_ones(value.size()[:-1], dtype=torch.long)
        self.register_buffer("defined_values", value_tensor)
        self.register_buffer("value_mask", value_mask)

        logger.info("Complete initialization of slot and value lookup")

    def get_no_grad_modules(self):
        res = {
            nn.ModuleList,
            nn.ModuleDict,
            layers.DiagonalAttentionScore,
            nn.Embedding,
        }
        return res

    def forward(self, input_ids, token_type_ids, attention_mask, answer_type_ids, labels, n_gpu=1, target_slot=None, transfer_labels=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(target_slot)
        seq_len = input_ids.size(2)

        # Utterance encoding
        _, _, all_attn_cache = self.bert(input_ids.view(-1, seq_len),
                                         token_type_ids.view(-1, seq_len),
                                         attention_mask.view(-1, seq_len),
                                         output_all_encoded_layers=False)

        slot_len = self.slot_ids.size(-1)

        # Domain-slot encoding
        slot_ids = self.slot_ids.unsqueeze(1).expand(-1, bs, -1).reshape(-1, slot_len)
        slot_mask = self.slot_mask.unsqueeze(1).expand(-1, bs, -1).reshape(-1, slot_len)
        slot_mask = torch.cat(
            [attention_mask.unsqueeze(0).expand(slot_dim, -1, -1, -1).reshape(-1, seq_len),
             slot_mask.to(dtype=attention_mask.dtype)], dim=-1)

        if self.slot_token_type_ids is not None:
            slot_token_type_ids = self.slot_token_type_ids.unsqueeze(1).expand(-1, bs, -1).reshape(-1, slot_len)
        else:
            slot_token_type_ids = None

        hidden, _, _ = self.bert(slot_ids, token_type_ids=slot_token_type_ids, attention_mask=slot_mask,
                                 output_all_encoded_layers=False, all_attn_cache=all_attn_cache,
                                 start_offset=seq_len, slot_dim=slot_dim)
        hidden = hidden[:, 0].view(slot_dim * ds, ts, -1)

        # Neural belief tracking
        hidden = self.transformer(hidden, None).view(slot_dim * bs, -1)

        if self.sa_act_1 is not None:
            hidden = self.tf_act(hidden)

        # Value attention
        value_mask = self.value_mask
        value_tensor = self.defined_values
        value_hidden, value_scores = self.value_attention(hidden.view(slot_dim, bs, -1),
                                                          value_tensor, x3=value_tensor, x2_mask=value_mask, return_scores=True)
        if self.diag_attn_act:
            value_hidden = self.value_act(value_hidden)

        # Value supervision
        masked_value_scores = layers.masked_log_softmax_fp16(value_scores, value_mask, dim=-1)
        masked_value_scores = masked_value_scores.view(slot_dim, ds, ts, -1)
        if self.save_gate and not self.training:
            self.value_scores.append(torch.softmax(masked_value_scores, dim=-1).detach().cpu().float())

        # Construct graph
        hidden = hidden.view(slot_dim, ds, ts, -1)
        graph_value = value_hidden.view(slot_dim, ds, ts, -1)[:, :, :-1].permute(1, 2, 0, 3).reshape(ds * (ts - 1), slot_dim, -1)
        graph_query = hidden[:, :, 1:].permute(1, 2, 0, 3).reshape(ds * (ts - 1), slot_dim, -1)
        graph_key = hidden[:, :, :-1].permute(1, 2, 0, 3).reshape(ds * (ts - 1), slot_dim, -1)

        if self.key_add_value:
            if self.key_add_value_pro:
                graph_key = self.key_value_project(torch.cat([graph_key, graph_value], dim=-1))
            else:
                graph_key = graph_key + graph_value

        # graph_mask = graph_key.new_zeros(graph_key.size()[:-1])[:, None, None, :]
        # if self.inter_domain:
        #     graph_mask = self.inter_domain_mask[None, None, :, :].to(dtype=graph_mask.dtype) + graph_mask
        if self.graph_attn_type == 0:
            if self.slot_res is not None:
                x2_mask = self.slot_res[None, :, :]
            else:
                x2_mask = None
            graph_hidden, graph_scores = self.graph_attention(graph_query, graph_key, x3=graph_value, x2_mask=x2_mask,
                                                              drop_diagonal=self.mask_self, return_scores=True)
        elif self.graph_attn_type == 1:
            graph_mask = graph_query.new_zeros(graph_query.size()[:-1])[:, None, None, :]
            if self.mask_self:
                diag_mask = torch.diag(graph_query.new_ones(slot_dim), diagonal=0).unsqueeze(0) * -10000.0
                graph_mask = graph_mask + diag_mask
            graph_hidden, (_, _, _, graph_scores) = self.graph_attention(graph_query, graph_key, graph_value,
                                                                         attention_mask=graph_mask)
        else:
            raise RuntimeError()

        graph_hidden = graph_hidden.view(ds, ts - 1, slot_dim, -1).permute(2, 0, 1, 3)
        if self.diag_attn_act:
            graph_hidden = self.graph_act(graph_hidden)

        if self.save_gate and not self.training:
            self.graph_scores.append(torch.softmax(graph_scores.view(ds, ts - 1, slot_dim, -1), dim=-1).detach().cpu())

        # Fusion
        if self.fuse_type in [0, 3, 4]:
            graph_hidden, gate = self.graph_project(hidden[:, :, 1:], graph_hidden)
            if self.save_gate and not self.training:
                self.gate_metric.append(gate.detach().cpu().float())
        else:
            graph_hidden = self.graph_project(hidden[:, :, 1:], graph_hidden)

        # if self.slot_res is not None:
        #     ini_hidden = graph_query.view(ds, ts - 1, slot_dim, -1).permute(2, 0, 1, 3)
        #     graph_hidden[self.slot_res, :, :, :] = ini_hidden[self.slot_res, :, :, :]

        hidden = torch.cat([hidden[:, :, 0].unsqueeze(2), graph_hidden], dim=2)

        # Graph supervision
        loss = 0
        if self.graph_add_sup > 0:
            slot_target = answer_type_ids[:, :-1].contiguous().view(ds * (ts - 1), 1, slot_dim).expand(-1, slot_dim, -1)
            # loss = self.graph_add_sup * layers.negative_entropy(score=graph_scores,
            #                                                     target=((slot_target == 0) | (slot_target == 1))) / ds
            loss = self.graph_add_sup * layers.negative_entropy(score=graph_scores,
                                                                target=((slot_target == 0) | (slot_target == 1))) / bs

            self.update_metric("slot_negative_loss", loss.item())
        if self.transfer_sup > 0 and transfer_labels is not None:
            transfer_labels = transfer_labels[:, 1:].view(ds * (ts - 1), slot_dim)
            # transfer_loss = self.transfer_sup * self.nll(graph_scores, transfer_labels) / ds
            transfer_loss = self.transfer_sup * self.nll(graph_scores, transfer_labels) / bs
            self.update_metric("transfer_loss", transfer_loss.item())
            loss += transfer_loss

        # Extra neural belief tracking
        if self.extra_nbt:
            hidden = hidden.reshape(slot_dim * ds, ts, -1)
            hidden = self.belief_tracker(hidden, None).view(slot_dim, ds, ts, -1)

        answer_type_logits = self.classifier(hidden)
        _, answer_type_pred = answer_type_logits.max(dim=-1)

        hidden = self.hidden_output(hidden)

        # Label (slot-value) encoding
        loss_slot = []
        pred_slot = []
        output = []
        type_loss = 0.
        matching_loss = 0.
        graph_loss = 0.
        for s, slot_id in enumerate(target_slot):  # note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            if self.distance_metric == 'product':
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts, num_slot_labels, -1)
                _hidden = hidden[s].unsqueeze(2).view(ds * ts, 1, -1)
                _dist = self.metric(_hidden, _hid_label).squeeze(1).reshape(ds, ts, num_slot_labels)
            else:
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
                _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(
                    ds * ts * num_slot_labels, -1)
                _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                """ FIX_UNDEFINED: Mask all undefined values """
                undefined_value_mask = (labels[:, :, s] == num_slot_labels)
                masked_slot_labels_for_loss = labels[:, :, s].masked_fill(undefined_value_mask, -1)

                # _loss = self.nll(_dist.view(ds * ts, -1), masked_slot_labels_for_loss.view(-1)) / (ds * 1.0)
                _loss = self.nll(_dist.view(ds * ts, -1), masked_slot_labels_for_loss.view(-1)) / (bs * 1.0)
                loss_slot.append(_loss.item())
                matching_loss += _loss.item()
                loss += _loss

                if self.graph_value_sup > 0:
                    # graph_value_loss = nn.functional.nll_loss(masked_value_scores[s].view(bs, -1),
                    #                                           masked_slot_labels_for_loss.view(-1), ignore_index=-1,
                    #                                           reduction='sum') / ds
                    graph_value_loss = nn.functional.nll_loss(masked_value_scores[s].view(bs, -1),
                                                              masked_slot_labels_for_loss.view(-1), ignore_index=-1,
                                                              reduction='sum') / bs
                    graph_value_loss = self.graph_value_sup * graph_value_loss
                    loss += graph_value_loss
                    graph_loss += graph_value_loss.item()

            if answer_type_ids is not None:
                # cls_loss = self.nll(answer_type_logits[s].view(ds * ts, -1), answer_type_ids[:, :, s].view(-1)) / ds
                cls_loss = self.nll(answer_type_logits[s].view(ds * ts, -1), answer_type_ids[:, :, s].view(-1)) / bs
                loss_slot[-1] += cls_loss.item()
                cls_loss = cls_loss * self.cls_loss_weight
                type_loss += cls_loss.item()
                loss += cls_loss

        if labels is None:
            return output

        self.update_metric("cls_loss", type_loss)
        self.update_metric("matching_loss", matching_loss)
        if self.graph_value_sup > 0:
            self.update_metric("graph_value_loss", graph_loss)

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)  # (ds, ts, slot_dim)
        answer_type_pred = answer_type_pred.permute(1, 2, 0).detach()  # (ds, ts, slot_dim)
        classified_mask = ((answer_type_ids != 2) * (answer_type_ids != -1)).view(-1, slot_dim)  # 1 for `none` and `do not care`
        # mask classifying values as 1
        value_accuracy = (pred_slot == labels).view(-1, slot_dim).masked_fill(classified_mask, 1)
        answer_type_accuracy = (answer_type_pred == answer_type_ids).view(-1, slot_dim)
        # For `none` and `do not care`, value_accuracy is always 1, the final accuracy depends on slot_gate_accuracy
        # For `ptr`, the final accuracy is 1 if and only if value_accuracy == 1 and slot_gate_accuracy == 1
        accuracy = value_accuracy * answer_type_accuracy
        # Slot accuracy
        slot_data_num = torch.sum(answer_type_ids.view(-1, slot_dim) > -1, dim=0).float()
        acc_slot = accuracy.sum(dim=0).float() / slot_data_num
        type_acc_slot = answer_type_accuracy.sum(dim=0).float() / slot_data_num
        # Joint accuracy
        valid_turn = torch.sum(answer_type_ids[:, :, 0].view(-1) > -1, dim=0).float()
        # acc = sum(torch.sum(accuracy, dim=1) // slot_dim).float() / valid_turn
        acc = sum(torch.div(torch.sum(accuracy, dim=1), slot_dim, rounding_mode='floor').float()).float() / valid_turn
        # type_acc = sum(torch.sum(answer_type_accuracy, dim=1) // slot_dim).float() / valid_turn
        type_acc = sum(torch.div(torch.sum(answer_type_accuracy, dim=1), slot_dim, rounding_mode='floor').float()).float() / valid_turn

        # accuracy = (pred_slot == labels).view(-1, slot_dim)
        # acc_slot = torch.sum(accuracy, 0).float() / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        # acc = sum(torch.sum(accuracy, 1) / slot_dim).float() / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc.item(), type_acc.item(), acc_slot.detach().cpu(), type_acc_slot.detach().cpu(), torch.cat(
                [answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1).detach().cpu()
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), type_acc.unusqueeze(0), acc_slot.unsqueeze(0), type_acc_slot.unsqueeze(
                0), torch.cat([answer_type_pred.unsqueeze(-1), pred_slot.unsqueeze(-1)], dim=-1).unsqueeze(0)

    # @staticmethod
    # def init_parameter(module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_normal_(module.weight)
    #         torch.nn.init.constant_(module.bias, 0.0)
    #     elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
    #         torch.nn.init.xavier_normal_(module.weight_ih_l0)
    #         torch.nn.init.xavier_normal_(module.weight_hh_l0)
    #         torch.nn.init.constant_(module.bias_ih_l0, 0.0)
    #         torch.nn.init.constant_(module.bias_hh_l0, 0.0)

    def update_metric(self, metric_name, *args, **kwargs):
        state = 'train' if self.training else 'eval'
        self.metrics[metric_name][state](*args, **kwargs)

    def get_metric(self, reset=False):
        state = 'train' if self.training else 'eval'
        metrics = {}
        for k, v in self.metrics.items():
            if state in v:
                metrics[k] = v[state].get_metric(reset=reset)
        return metrics

    def get_gate_metric(self, reset=False):
        metric = torch.cat(self.gate_metric, dim=1).squeeze(-1)
        if reset:
            self.gate_metric.clear()
        return metric

    def get_value_scores(self, reset=False):
        value_scores = torch.cat(self.value_scores, dim=1)
        value_scores = value_scores.max(dim=-1)
        if reset:
            self.value_scores.clear()
        return value_scores

    def get_graph_scores(self, reset=False):
        graph_scores = torch.cat(self.graph_scores, dim=0)
        if reset:
            self.graph_scores.clear()
        return graph_scores
