import torch
from pytorch_transformers.modeling_bert import BertModel
from torch import nn
import time

import model
from util.general_util import AverageMeter
from util.logger import get_child_logger

logger = get_child_logger(__name__)


class BertQADst(nn.Module):
    def __init__(self, config):
        super(BertQADst, self).__init__()
        logger.info(config["bert_model"])
        self.value_encoder = BertModel.from_pretrained(pretrained_model_name_or_path=config["bert_model"])
        self.model = model.from_params(config)
        self.value_embedding = nn.ModuleList()

        self.eval_joint_acc = AverageMeter()
        self.eval_slot_acc = AverageMeter()

        self.device = config["device"]

    def forward(self, input_ids, token_type_ids, input_mask, dialog_mask, value_ids, example_index, slot_index=None):
        if self.training:
            assert len(input_ids.size()) == 3
            value_embedding, value_mask = self.prepare_value_embedding(slot_index)
            # logger.info(value_embedding.size())
            output = self.model(input_ids, token_type_ids, input_mask, dialog_mask, value_embedding,
                                value_ids=value_ids, value_mask=value_mask)
            return output
        else:
            assert len(input_ids.size()) == 4
            slot_dim = input_ids.size(2)
            outputs = []
            for slot_index in range(slot_dim):
                output = self.model(input_ids[:, :, slot_index], token_type_ids[:, :, slot_index], input_mask[:, :, slot_index],
                                    dialog_mask, self.value_embedding[slot_index].weight,
                                    value_ids=value_ids[:, :, slot_index])
                outputs.append(output)

            model_output = {key: torch.stack([x[key] for x in outputs], dim=2) for key in outputs[0].keys()}
            loss = model_output["loss"].sum() / slot_dim
            model_output["loss"] = loss
            if not self.training:
                # batch, max_turns, slot_dim, value_dim
                logits = model_output["logits"]
                _, value_pred = logits.max(dim=-1)

                slot_correct = (value_pred == value_ids).masked_fill(1 - dialog_mask.unsuqueeze(-1), 0).sum()
                joint_correct = (
                            ((value_pred == value_ids).masked_fill(1 - dialog_mask.unsqueeze(-1), 0).sum(dim=-1)) / slot_dim).sum()
                total_turns = dialog_mask.sum()
                self.eval_slot_acc.update(val=slot_correct.item(), n=total_turns.item() * slot_dim)
                self.eval_joint_acc.update(val=joint_correct.item(), n=total_turns.item())
            return model_output

    def get_value_embedding(self, value_input_ids, value_input_mask, value_token_type_ids):
        logger.info("Generating value embeddings")
        self.eval()
        with torch.no_grad():
            for slot_index, (input_ids, input_mask, token_type_ids) in enumerate(
                    zip(value_input_ids, value_input_mask, value_token_type_ids)):
                h = self.value_encoder(input_ids, token_type_ids, input_mask)[0][:, 0]
                self.value_embedding.append(nn.Embedding.from_pretrained(h, freeze=True))
                assert len(self.value_embedding[slot_index].weight.size()) == 2
        self.value_encoder = None
        logger.info("Value embedding has been saved")

    def get_metric(self, reset=False):
        metric = {
            "joint_acc": self.eval_joint_acc.avg,
            "slot_acc": self.eval_slot_acc.avg
        }
        if reset:
            self.eval_joint_acc.reset()
            self.eval_slot_acc.reset()
        return metric

    def prepare_value_embedding(self, slot_index: torch.Tensor):
        index = slot_index.detach().cpu().tolist()
        value_embedding_list = []
        max_value_dim = 0
        for x in index:
            value_embedding_list.append(self.value_embedding[x].weight)
            max_value_dim = max(max_value_dim, self.value_embedding[x].weight.size(0))
        value_hidden_dim = value_embedding_list[0].size(-1)
        value_padding = value_embedding_list[0].new_zeros(1, value_hidden_dim)
        output = []
        value_mask = value_padding.new_ones(len(index), max_value_dim)
        for value_emb_idx, value_embedding in enumerate(value_embedding_list):
            pad_size = max_value_dim - value_embedding.size(0)
            if pad_size > 0:
                value_embedding = torch.cat([value_embedding, value_padding.expand(pad_size, -1)], dim=0).unsqueeze(0)
                value_mask[value_emb_idx, -pad_size:] = value_padding.new_zeros(pad_size)
            else:
                value_embedding = value_embedding.unsqueeze(0)
            output.append(value_embedding)
        return torch.cat(output, dim=0), value_mask
