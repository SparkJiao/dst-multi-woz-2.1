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

    def forward(self, input_ids, token_type_ids, input_mask, dialog_mask, value_ids=None, example_index=None):
        slot_dim = input_ids.size(2)
        logger.info(value_ids.size())
        logger.info(input_ids.size())
        logger.info(value_ids.dtype)
        # TODO:
        #  Currently CUDA out of memory always happen since the all slot will be processed in a single batch.
        #  Make a single instance for each slot will be better.
        outputs = []
        for slot_index in range(slot_dim):
            logger.info(f"Slot Index: {slot_index}")
            inputs = (input_ids[:, :, slot_index], token_type_ids[:, :, slot_index], input_mask[:, :, slot_index],
                      dialog_mask, self.value_embedding[slot_index].weight, value_ids[:, :, slot_index])
            # inputs = tuple([t.to(self.device) for t in inputs])
            output = self.model(*inputs)
            outputs.append(output)

        model_output = {key: torch.stack([x[key] for x in outputs], dim=2) for key in outputs[0].keys()}
        loss = model_output["loss"].sum()
        if not self.training:
            # batch, max_turns, slot_dim, value_dim
            logits = model_output["logits"]
            _, value_pred = logits.max(dim=-1)

            slot_correct = (value_pred == value_ids).masked_fill(1 - dialog_mask.unsuqueeze(-1), 0).sum()
            joint_correct = (((value_pred == value_ids).masked_fill(1 - dialog_mask.unsqueeze(-1), 0).sum(dim=-1)) / slot_dim).sum()
            total_turns = dialog_mask.sum()
            self.eval_slot_acc.update(val=slot_correct.item(), n=total_turns.item() * slot_dim)
            self.eval_joint_acc.update(val=joint_correct.item(), n=total_turns.item())
        return loss

    def get_value_embedding(self, value_input_ids, value_input_mask, value_token_type_ids):
        logger.info("Generating value embeddings")
        self.eval()
        with torch.no_grad():
            for slot_index, (input_ids, input_mask, token_type_ids) in enumerate(
                    zip(value_input_ids, value_input_mask, value_token_type_ids)):
                h = self.value_encoder(input_ids, token_type_ids, input_mask)[0][:, 0]
                self.value_embedding.append(nn.Embedding.from_pretrained(h, freeze=True))
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