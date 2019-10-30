from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

class GenDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, slot_label, domain_slot_ids, value_ids):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.slot_label = slot_label
        self.domain_slot_ids = domain_slot_ids
        self.value_ids = value_ids
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(slot_label) == len(domain_slot_ids) \
        == len(value_ids)

    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index],
            'slot_label': self.slot_label[index],
            'domain_slot_ids': self.domain_slot_ids[index],
            'value_ids': self.value_ids[index]
        }


