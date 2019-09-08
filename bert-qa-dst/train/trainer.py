from torch import nn
import torch

from pytorch_transformers.modeling_bert import BertModel


class DSTController(nn.Module):
    def __init__(self, bert_model, model):
        super(DSTController, self).__init__()
        self.value_encoder = BertModel.from_pretrained(pretrained_model_name_or_path=bert_model)
        self.model = model
        self.value_embedding = nn.ModuleList()
        # for slot_values in value_input_ids:
        #     self.value_embedding.append(nn.Embedding(len(slot_values), self.value_encoder.config.hidden_size))

    def forward(self, input_ids, token_type_ids, input_mask, value_ids, dialog_mask, example_index, device):
        slot_dim = input_ids.size(2)
        outputs = []
        for slot_index in range(slot_dim):
            inputs = (input_ids[:, :, slot_index], token_type_ids[:, :, slot_index], input_mask[:, :, slot_index],
                      input_mask[:, :, slot_index], value_ids[:, :, slot_index], dialog_mask)
            inputs = [t.to(device) for t in inputs] + [self.value_embedding[slot_index]]
            output = self.model(*inputs)
            outputs.append(output)

    def get_value_embedding(self, value_input_ids, value_input_mask, value_token_type_ids):
        for slot_index, (input_ids, input_mask, token_type_ids) in enumerate(
                zip(value_input_ids, value_input_mask, value_token_type_ids)):
            h = self.value_encoder(input_ids, token_type_ids, input_mask)[0][:, 0]
            self.value_embedding.append(nn.Embedding.from_pretrained(h, freeze=True))
        self.value_encoder = None
        torch.cuda.empty_cache()
