import torch
from torch import nn
from . import layers
from .dialog_encoder import DialogEncoder
from utils.general_util import AverageMeter
from data.instance import State
from pytorch_transformers import BertModel


class DSTSpanModel(nn.Module):
    def __init__(self, encoder, query_encoder, freeze_query_encoder=True):
        super(DSTSpanModel, self).__init__()
        self.dialog_encoder = DialogEncoder.from_pretrained(**encoder)
        config = self.dialog_encoder.config
        dropout_p = config.hidden_dropout_prob
        self.fuse_layer = layers.FuseLayer(self.dialog_encoder.output_dim, dropout=dropout_p)
        self.value_type_predictor = nn.Linear(self.dialog_encoder.output_dim, 5)
        self.value_start_predictor = layers.BiLinear(config.hidden_size, self.dialog_encoder.output_dim, dropout=dropout_p)
        self.value_end_predictor = layers.BiLinear(config.hidden_size, self.dialog_encoder.output_dim, dropout=dropout_p)

        self.query_encoder = BertModel.from_pretrained(**query_encoder)
        self.freeze_query_encoder = freeze_query_encoder
        if self.freeze_query_encoder:
            for p in self.query_encoder:
                p.requires_grad = False
        self.domain_slot_input_ids = None
        self.domain_slot_embedding = None
        self.slot_dim = None
        self.domain_slot_set = False

        # Metric
        self.train_loss = AverageMeter()
        self.train_joint_acc = AverageMeter()
        self.train_slot_acc = AverageMeter()
        self.eval_loss = AverageMeter()
        self.eval_joint_acc = AverageMeter()
        self.eval_slot_acc = AverageMeter()

    def forward(self, input_ids, token_type_ids, attention_mask, dialog_mask):
        # batch, max_turn, h
        dialog_hidden, seq_output = self.dialog_encoder(input_ids, token_type_ids, attention_mask, dialog_mask)
        batch, max_turn, _ = dialog_hidden.size()

        # slot_embedding = self.domain_slot_embedding
        # slot_dim, slot_hidden_size = slot_embedding.size()
        # slot_embedding = slot_embedding.reshape(1, slot_dim, 1, slot_hidden_size)\
        #     .expand(batch, -1, -1, -1).reshape(batch * slot_dim, 1, slot_hidden_size)
        # dialog_hidden = dialog_hidden.unsqueeze(1).expand(-1, slot_dim, -1, -1)

    def update(self, batch, other_data):
        self.train()
        self.set_domain_slot_embedding(other_data["domain_slot_input_ids"])
        return self.forward(*batch)

    def evaluate(self, batch, other_data):
        self.eval()
        self.set_domain_slot_embedding(other_data["domain_slot_input_ids"])
        pass

    def set_domain_slot_embedding(self, domain_slot_input_ids):
        if not self.domain_slot_set and self.freeze_query_encoder:
            self.query_encoder.eval()
            with torch.no_grad:
                domain_slot_hidden = self.query_encoder(domain_slot_input_ids)[0]
            domain_slot_hidden = domain_slot_hidden[:, 0]
            self.domain_slot_embedding = nn.Embedding.from_pretrained(domain_slot_hidden, freeze=True)
            self.slot_dim = domain_slot_hidden.size(0)
            self.domain_slot_set = True
            self.domain_slot_input_ids = domain_slot_input_ids
            self.query_encoder = None

    def get_domain_slot_embedding(self, domain_slot_index):
        if self.freeze_query_encoder:
            return self.domain_slot_embedding(domain_slot_index)
        else:
            self.query_encoder.train()
            return self.query_encoder(self.domain_slot_input_ids.index_select(index=domain_slot_index))[0]

    def get_metric(self, state: State, reset=False):
        if state == State.Train:
            metric = {
                "train/loss": self.train_loss.avg,
                "train/slot_acc": self.train_slot_acc.avg,
                "train/joint_acc": self.train_joint_acc.avg
            }
            if reset:
                self.train_loss.reset()
                self.train_slot_acc.reset()
                self.train_joint_acc.reset()
            return metric
        else:
            metric = {
                "eval/loss": self.eval_loss.avg,
                "eval/slot_acc": self.eval_slot_acc.avg,
                "eval/joint_acc": self.eval_joint_acc.avg
            }
            if reset:
                self.eval_loss.reset()
                self.eval_slot_acc.reset()
                self.eval_joint_acc.reset()

            return metric
