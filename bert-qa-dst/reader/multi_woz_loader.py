import collections
import csv
import json
from typing import List
from tqdm import tqdm
import pickle

import torch
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader

from util.data_instance import MultiWOZExample, DialogTurnExample, State
from util.logger import get_child_logger

logger = get_child_logger(__name__)


class MultiWOZReader:
    def __init__(self, vocab_file, ontology_file):
        logger.info("Params: ")
        logger.info(f"vocab_file: {vocab_file}")
        logger.info(f"ontology_file: {ontology_file}")
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_file)
        self.ontology = self._read_labels(ontology_file)
        self.target_slot = list(self.ontology.keys())
        self.domain_slot_vocab = {}
        self.value_vocab = []
        for i, (domain_slot, values) in enumerate(self.ontology.items()):
            self.domain_slot_vocab[domain_slot] = i
            self.value_vocab.append({value: j for j, value in enumerate(values)})
        logger.info('Target slot information: ')
        logger.info(self.target_slot)

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines

    def read(self, input_file, max_seq_length: int, state: State, batch_size: int):

        value_input_ids, value_input_mask, value_token_type_ids = self._convert_slot_values_into_ids(self.value_vocab)
        meta_data = {
            'value_input_ids': value_input_ids,
            'value_input_mask': value_input_mask,
            'value_token_type_ids': value_token_type_ids
        }

        cached_file = f'{input_file}-{max_seq_length}'
        try:
            logger.info(f'Load data from cache: {cached_file}')
            with open(cached_file, "rb") as f:
                input_data = pickle.load(f)
        except FileNotFoundError:
            data = self._read_tsv(input_file)
            pre_dialog_id = ''
            examples = []
            dialog_turn_examples = []
            for i, line in enumerate(tqdm(data, desc='Reading examples...')):
                dialog_id = line[0]
                turn_id = line[1]
                if pre_dialog_id is not None:
                    if dialog_id != pre_dialog_id and dialog_turn_examples:
                        examples.append(MultiWOZExample(dialog_id=pre_dialog_id,
                                                        dialog_turn_examples=dialog_turn_examples))
                        dialog_turn_examples = []
                        assert examples[-1].dialog_turn_examples != []
                values = line[4:]
                assert len(values) == len(self.target_slot)
                domain_slot_values = {domain_slot: value for domain_slot, value in zip(self.target_slot, values)}
                dialog_turn_examples.append(
                    DialogTurnExample(usr_utt=line[2], sys_utt=line[3], domain_slot_value=domain_slot_values))
                pre_dialog_id = dialog_id
            if dialog_turn_examples:
                examples.append(MultiWOZExample(dialog_id=pre_dialog_id,
                                                dialog_turn_examples=dialog_turn_examples))
            tensor_data = self._convert_examples_to_features(examples, max_seq_length)
            input_data = {
                'examples': examples,
                'tensor_data': tensor_data
            }
            with open(cached_file, "wb") as f:
                pickle.dump(input_data, f)
        examples = input_data['examples']
        tensor_data = input_data['tensor_data']
        meta_data["examples"] = examples
        data_loader = self._get_data_loader(tensor_data, state=state, batch_size=batch_size)
        logger.info(f'Load {len(examples)} examples and {len(data_loader)} input features')
        return meta_data, data_loader

    def _convert_examples_to_features(self, examples: List[MultiWOZExample], max_seq_length: int):
        max_turns = 0
        for example in examples:
            max_turns = max(len(example.dialog_turn_examples), max_turns)
        all_input_ids = []
        all_token_type_ids = []
        all_input_mask = []
        all_value_ids = []
        all_dialog_mask = []
        padding = [0] * (max_seq_length - 3)
        turn_input_id_padding = [self.bert_tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[SEP]']) + padding] * len(
            self.domain_slot_vocab)
        turn_token_type_id_padding = [[0, 0, 1] + padding] * len(self.domain_slot_vocab)
        turn_mask_padding = [[1, 1, 1] + padding] * len(self.domain_slot_vocab)
        for example in tqdm(examples, desc='Converting examples to features...'):
            dialog_examples = example.dialog_turn_examples
            dialog_input_ids = []
            dialog_token_type_ids = []
            dialog_input_mask = []
            dialog_value_ids = []
            dialog_mask = []
            for turn in dialog_examples:
                usr_utt = turn.usr_utt
                sys_utt = turn.sys_utt
                utt_tokens = self.bert_tokenizer.tokenize(usr_utt) + self.bert_tokenizer.tokenize(sys_utt)
                domain_slot_values = turn.domain_slot_value
                # domain_slot_pairs = []
                value_ids = []
                turn_input_ids = []
                turn_token_type_ids = []
                turn_input_mask = []
                for domain_slot, value in domain_slot_values.items():
                    # domain_slot_pairs.append(domain_slot)
                    # if value == 'undefined':
                    #     value_ids.append(-1)
                    # else:
                    value_ids.append(self.value_vocab[self.domain_slot_vocab[domain_slot]][value])

                    slot_tokens = ['[CLS]'] + self.bert_tokenizer.tokenize(domain_slot) + ['[SEP]']
                    type_ids = [0] * len(slot_tokens)
                    tokens = slot_tokens + utt_tokens
                    if len(tokens) > max_seq_length - 1:
                        tokens = tokens[:max_seq_length - 1]
                    tokens.append('[SEP]')
                    type_ids += [1] * (len(tokens) - len(slot_tokens))
                    mask = [1] * len(type_ids)
                    input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                    while len(input_ids) < max_seq_length:
                        input_ids.append(0)
                        type_ids.append(0)
                        mask.append(0)
                    assert len(input_ids) == max_seq_length
                    assert len(type_ids) == max_seq_length
                    assert len(mask) == max_seq_length
                    turn_input_ids.append(input_ids)
                    turn_token_type_ids.append(type_ids)
                    turn_input_mask.append(mask)
                assert len(turn_input_ids) == len(self.target_slot)
                assert len(turn_token_type_ids) == len(self.domain_slot_vocab)
                assert len(turn_input_mask) == len(self.domain_slot_vocab)
                dialog_input_ids.append(turn_input_ids)
                dialog_token_type_ids.append(turn_token_type_ids)
                dialog_input_mask.append(turn_input_mask)
                dialog_value_ids.append(value_ids)
                dialog_mask.append(1)
            while len(dialog_mask) < max_turns:
                dialog_input_ids.append(turn_input_id_padding)
                dialog_token_type_ids.append(turn_token_type_id_padding)
                dialog_input_mask.append(turn_mask_padding)
                dialog_value_ids.append([-1] * len(self.domain_slot_vocab))
                dialog_mask.append(0)
            assert len(dialog_input_ids) == max_turns
            assert len(dialog_token_type_ids) == max_turns
            assert len(dialog_input_mask) == max_turns
            assert len(dialog_value_ids) == max_turns
            assert len(dialog_mask) == max_turns
            all_input_ids.append(dialog_input_ids)
            all_token_type_ids.append(dialog_token_type_ids)
            all_input_mask.append(dialog_input_mask)
            all_value_ids.append(dialog_value_ids)
            all_dialog_mask.append(dialog_mask)

        # input_ids = torch.LongTensor(all_input_ids)
        # token_type_ids = torch.LongTensor(all_token_type_ids)
        # input_mask = torch.LongTensor(all_input_mask)
        # value_ids = torch.LongTensor(all_value_ids)
        # dialog_mask = torch.LongTensor(all_dialog_mask)
        # assert input_ids.size() == token_type_ids.size() == input_mask.size() == (len(all_input_ids), max_turns,
        #                                                                           len(self.domain_slot_vocab),
        #                                                                           max_seq_length)
        # assert value_ids.size() == (len(all_input_ids), max_turns, len(self.domain_slot_vocab))
        # assert dialog_mask.size(1) == max_turns

        input_data = {
            "input_ids": all_input_ids,
            "token_type_ids": all_token_type_ids,
            "input_mask": all_input_mask,
            "dialog_mask": all_dialog_mask,
            "value_ids": all_value_ids
        }
        logger.info(f"Max turns: {max_turns}")
        return input_data

    @staticmethod
    def _data_to_tensor(tensor_data):
        input_ids = torch.LongTensor(tensor_data["input_ids"])
        token_type_ids = torch.LongTensor(tensor_data["token_type_ids"])
        input_mask = torch.LongTensor(tensor_data["input_mask"])
        dialog_mask = torch.LongTensor(tensor_data["dialog_mask"])
        value_ids = torch.LongTensor(tensor_data["value_ids"])
        example_indexes = torch.arange(input_ids.size(0), dtype=torch.long)
        return input_ids, token_type_ids, input_mask, dialog_mask, value_ids, example_indexes

    def _get_data_loader(self, tensor_data, state: State, batch_size):
        tensors = self._data_to_tensor(tensor_data)
        dataset = TensorDataset(*tensors)
        if state == State.TRAIN:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, sampler=sampler)
        return data_loader

    @staticmethod
    def _read_labels(ontology_file):
        with open(ontology_file) as f:
            ontology = json.load(f)
        for slot in ontology:
            ontology[slot].append("none")
            ontology[slot].append("undefined")
        return collections.OrderedDict(sorted(ontology.items()))

    def _convert_slot_values_into_ids(self, value_vocab):
        slot_value_input_ids = []
        max_value_length = 0
        for slot_values in value_vocab:
            value_input_ids = []
            for value in slot_values:
                input_ids = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + self.bert_tokenizer.tokenize(value) + ['[SEP]'])
                value_input_ids.append(input_ids)
                max_value_length = max(max_value_length, len(input_ids))
            slot_value_input_ids.append(value_input_ids)

        # Pad
        all_input_mask = []
        all_token_type_ids = []
        for slot_values in slot_value_input_ids:
            input_mask = []
            token_type_ids = []
            for value_idx, input_ids in enumerate(slot_values):
                padding_len = max_value_length - len(input_ids)
                input_mask.append([1] * len(input_ids) + [0] * padding_len)
                token_type_ids.append([1] * len(input_ids) + [0] * padding_len)
                slot_values[value_idx] += [0] * padding_len
            all_input_mask.append(input_mask)
            all_token_type_ids.append(token_type_ids)

        slot_value_input_ids = [torch.LongTensor(x) for x in slot_value_input_ids]
        input_mask = [torch.LongTensor(x) for x in all_input_mask]
        token_type_ids = [torch.LongTensor(x) for x in all_token_type_ids]
        return slot_value_input_ids, input_mask, token_type_ids

    @classmethod
    def from_params(cls, _config):
        _vocab_file = _config["vocab_file"]
        _ontology_file = _config["ontology_file"]
        return cls(_vocab_file, _ontology_file)
