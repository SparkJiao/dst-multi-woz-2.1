import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers.tokenization_bert import BertTokenizer

from .utils import fix_general_label_error

logger = logging.getLogger(__name__)

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

gating_dict = {
    "ptr": 0,
    "dontcare": 1,
    "none": 2
}


class MultiWOZ:
    def __init__(self, ontology, bert_dir):
        self.ontology = self.get_slot_information(json.load(open(ontology, 'r')))
        self.value_map = self.get_value_map(self.ontology)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)

    @staticmethod
    def get_slot_information(ontology):
        return {k: v for k, v in ontology.items() if k.split('-')[0] in EXPERIMENT_DOMAINS}

    @staticmethod
    def get_value_map(ontology):
        value_map = []
        for slot, values in ontology.items():
            value_map.append({v: value_id for value_id, v in enumerate(values)})
        return value_map

    def read(self, file_name):
        logger.info(f"Reading data from {file_name}")
        dialogs = json.load(open(file_name, 'r'))

        data = []
        for dial_dict in dialogs:
            dialog_data = []
            for ti, turn in tqdm(enumerate(dial_dict)):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                sys_utt = turn["system_transcript"]
                usr_utt = turn["transcript"]
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, self.ontology.keys())

                values, gating_label = [], []
                for slot in self.ontology.keys():
                    if slot in turn_belief_dict.keys():
                        values.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == 'dontcare':
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == 'none':
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])
                    else:
                        values.append("none")
                        gating_label.append(gating_dict["none"])

                dialog_data.append({
                    "id": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    "gating_label": gating_label,
                    "sys_utt": sys_utt,
                    "usr_utt": usr_utt,
                    "values": values
                })
            data.append(dialog_data)

    def read_train(self, file_name):
        return self.read(file_name)

    def read_dev(self, file_name):
        return self.read(file_name)

    def read_test(self, file_name):
        return self.read(file_name)

    def convert_examples_to_features(self, data, max_seq_length):

        max_turns = 0
        for dialog in data:
            max_turns = max(max_turns, len(dialog))

        padding_output = self.tokenizer.encode_plus('', '', add_special_tokens=True, max_length=max_seq_length)
        assert len(padding_output["input_ids"]) == 3
        padding = [
            padding_output["input_ids"] + [0] * (max_seq_length - 3),
            padding_output["token_type_ids"] + [0] * (max_seq_length - 3),
            [1] * 3 + [0] * (max_seq_length - 3)
        ]

        all_data = []
        for dialog in tqdm(data, total=len(data), desc='Converting example into features...'):
            dialog_input = []
            for turn in dialog:
                sys_utt = turn["sys_utt"]
                usr_utt = turn["usr_utt"]
                encode_output = self.tokenizer.encode_plus(sys_utt, usr_utt, add_special_tokens=True,
                                                           max_length=max_seq_length, truncation_strategy='only_first')
                input_ids, token_type_ids = encode_output["input_ids"], encode_output["token_type_ids"]
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    token_type_ids.append(0)
                    input_mask.append(0)
                gating_label = turn["gating_label"]
                value_index = []
                for slot_index, (slot, values) in enumerate(self.ontology.items()):
                    value = turn["values"][slot_index]
                    if value in values:
                        value_index.append(self.value_map[slot_index][value])
                    else:
                        value_index.append(-1)

                dialog_input.append({
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'input_mask': input_mask,
                    'gating_label': gating_label,
                    'value_index': value_index
                })
            while len(dialog_input) < max_turns:
                dialog_input.append({
                    'input_ids': padding[0],
                    'token_type_ids': padding[1],
                    'input_mask': padding[2],
                    'gating_label': [-1] * len(self.ontology),
                    'value_index': [-1] * len(self.ontology)
                })
            all_data.append(dialog_input)

        all_input_ids = [turn["input_ids"] for dial in all_data for turn in dial]
        all_token_type_ids = [turn["token_type_ids"] for dial in all_data for turn in dial]
        all_input_mask = [turn["input_mask"] for dial in all_data for turn in dial]
        all_gating_label = [turn["gating_label"] for dial in all_data for turn in dial]
        all_value_index = [turn["value_index"] for dial in all_data for turn in dial]

        return {
            'input_ids': all_input_ids,
            'token_type_ids': all_token_type_ids,
            'input_mask': all_input_mask,
            'gating_label': all_gating_label,
            'value_index': all_value_index
        }


class MultiWOZDataset(Dataset):

    def __init__(self, data_dict):
        self.data = self.data2tensor(data_dict)
        self.length = len(self.data["input_ids"])

    def __getitem__(self, index: int) -> T_co:
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def data2tensor(data_dict):
        tensors = {
            k: torch.LongTensor(v) for k, v in data_dict.items()
        }
        data_len, max_turns, max_seq_len = tensors["input_ids"].size()
        assert tensors["token_type_ids"].size() == (data_len, max_turns, max_seq_len)
        assert tensors["input_mask"].size() == (data_len, max_turns, max_seq_len)
        assert tensors["gating_label"].size()[:2] == tensors["value_index"].size()[:2] == (data_len, max_turns)
        assert tensors["gating_label"].size(-1) == tensors["value_index"].size(-1)
        return tensors
