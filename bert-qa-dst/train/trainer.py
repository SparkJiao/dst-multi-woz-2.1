from torch import nn


class DSTController(nn.Module):
    def __init__(self, model, values_vocab):
        super(DSTController, self).__init__()
        self.model = model
        self.value_embedding = self.get_value_embedding(values_vocab)

    def forward(self, input_ids, token_type_ids, input_mask, value_ids, dialog_mask, example_index):
        slot_dim = input_ids.size(2)
        outputs = []
        for slot_index in range(slot_dim):
            output = self.model(input_ids[:, :, slot_index], token_type_ids[:, :, slot_index],
                                input_mask[:, :, slot_index],
                                value_ids[:, :, slot_index], dialog_mask, self.value_embedding[slot_index])
            outputs.append(output)

    def get_value_embedding(self, values_vocab):
        raise NotImplementedError
