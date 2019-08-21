import json
from collections import namedtuple
from typing import List

import torch
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset

from data.instance import MultiWOZInputExample, State
from utils import general_util
from utils.logger import get_child_logger

logger = get_child_logger(__name__)


class MultiWOZSpanReader:
    def __init__(self, vocab_file):
        self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file)

    def read(self, input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)

        all_examples: List[List[MultiWOZInputExample]] = []
        for file_id, dialog in data.items():
            dialog_examples: List[MultiWOZInputExample] = []
            for turn in dialog:
                user_utt = turn["user_utterance"]
                system_utt = turn["system_response"]

                user_utt_tokens, user_utt_char2tok_map = general_util.get_char_to_tokens_map(user_utt)
                system_utt_tokens, system_utt_char2tok_map = general_util.get_char_to_tokens_map(system_utt)

                belief = turn["belief"]
                domain_slot_value_dict = {}
                for domain_slot, value in belief.items():
                    if value["value"] in ['none', 'yes', 'no', 'do not care']:
                        domain_slot_value_dict[domain_slot] = value
                        domain_slot_value_dict[domain_slot]["span_end"] = -1
                    else:
                        if value["turn_index"] == -1:
                            domain_slot_value_dict[domain_slot] = value
                            domain_slot_value_dict[domain_slot]["span_end"] = -1
                        else:
                            from_which = value["from_which"]
                            turn_index = value["turn_index"]

                            if turn_index >= len(dialog_examples):
                                target_user_char2tok_map = user_utt_char2tok_map
                                target_sys_char2tok_map = system_utt_char2tok_map

                                target_user_tokens = user_utt_tokens
                                target_sys_tokens = system_utt_tokens
                            else:
                                target_user_char2tok_map = dialog_examples[turn_index].user_utt_char2tok_map
                                target_sys_char2tok_map = dialog_examples[turn_index].system_utt_char2tok_map

                                target_user_tokens = dialog_examples[turn_index].user_utt_tokens
                                target_sys_tokens = dialog_examples[turn_index].system_utt_tokens

                            if from_which == "system":
                                value["span_start"] = target_sys_char2tok_map[value["span_start"]]
                                value["span_end"] = target_sys_char2tok_map[value["span_start"] + len(value["value"]) - 1]
                                extract_value = " ".join(target_sys_tokens[value["span_start"]: (value["span_end"] + 1)])
                                if extract_value != value["value"]:
                                    logger.warning(f"Can't extract exact value for {value['value']} // Only get {extract_value}")
                            elif from_which == "user":
                                value["span_start"] = target_user_char2tok_map[value["span_start"]]
                                value["span_end"] = target_user_char2tok_map[value["span_start"] + len(value["value"]) - 1]
                                extract_value = " ".join(target_user_tokens[value["span_start"]: (value["span_end"] + 1)])
                                if extract_value != value["value"]:
                                    logger.warning(f"Can't extract exact value for {value['value']} // Only get {extract_value}")
                            domain_slot_value_dict[domain_slot] = value

                dialog_examples.append(MultiWOZInputExample(
                    dialog_id=file_id,
                    turn_index=turn["turn_index"],
                    user_utt_tokens=user_utt_tokens,
                    user_utt_char2tok_map=user_utt_char2tok_map,
                    system_utt_tokens=system_utt_tokens,
                    system_utt_char2tok_map=system_utt_char2tok_map,
                    domain_slot_value=domain_slot_value_dict
                ))

            all_examples.append(dialog_examples)
        return all_examples

    def get_data_loader(self, examples: List[List[MultiWOZInputExample]], max_seq_length: int, state: State, batch_size: int):
        # Convert examples to features
        max_turn = 0
        for dialog_examples in examples:
            max_turn = max(max_turn, len(dialog_examples))
        turn_token_padding = ["[CLS]", "[SEP]", "[SEP]"]
        turn_input_id_padding = self.bert_tokenizer.convert_tokens_to_ids(turn_token_padding) + [0] * (max_seq_length - 3)
        turn_token_type_padding = [0, 0, 1] + [0] * (max_seq_length - 3)
        turn_mask_padding = [1, 1, 1] + [0] * (max_seq_length - 3)

        all_domain_slots = examples[0][0].domain_slot_value.keys()
        domain_slot_vocab = {key: idx for idx, key in enumerate(all_domain_slots)}
        domain_slot_input_ids = []
        for domain_slot in all_domain_slots:
            domain_slot_input_ids.append(self.bert_tokenizer.tokenize(str(domain_slot)))

        InputFeature = namedtuple("InputFeature", ("input_ids", "token_type_ids", "input_mask",
                                                   "value_start", "value_end", "value_type", "dialog_mask"))
        input_features: List[InputFeature] = []
        for dialog_examples in examples:
            dialog_input_ids = []
            dialog_token_type_ids = []
            dialog_input_mask = []
            dialog_value_starts = []
            dialog_value_ends = []
            dialog_value_types = []

            for turn_example in dialog_examples:
                user_tok2wp_map = []
                user_tokens = turn_example.user_utt_tokens
                user_pieces = []
                for tok_id, tok in enumerate(user_tokens):
                    user_tok2wp_map[tok_id].append(len(user_pieces))
                    pieces = self.bert_tokenizer.tokenize(tok)
                    user_pieces.extend(pieces)

                sys_tok2wp_map = []
                sys_tokens = turn_example.system_utt_tokens
                sys_pieces = []
                for tok_id, tok in enumerate(sys_tokens):
                    sys_tok2wp_map.append(len(sys_pieces))
                    pieces = self.bert_tokenizer.tokenize(tok)
                    sys_pieces.extend(pieces)

                # Get copy of word pieces to truncate
                cb_usr_pieces = user_pieces[:]
                cb_sys_pieces = sys_pieces[:]
                general_util.truncate_seq_pair(cb_usr_pieces, cb_sys_pieces, max_length=max_seq_length - 3)

                # Generate bert inputs of sequence
                turn_pieces = ["[CLS]"] + cb_usr_pieces + ["[SEP]"] + cb_sys_pieces + ["[SEP]"]
                turn_input_ids = self.bert_tokenizer.convert_tokens_to_ids(turn_pieces)
                turn_token_type_ids = [0] * (len(cb_usr_pieces) + 2) + [1] * (len(cb_sys_pieces) + 1)
                turn_input_mask = [1] * len(turn_input_ids)
                while len(turn_input_ids) < max_seq_length:
                    turn_input_ids.append(0)
                    turn_token_type_ids.append(0)
                    turn_input_mask.append(0)
                assert len(turn_input_ids) == max_seq_length
                assert len(turn_token_type_ids) == max_seq_length
                assert len(turn_input_mask) == max_seq_length

                # Generate domain-slot value span labels for all domain-slot pairs.
                value_span_starts = []
                value_span_ends = []
                value_type_dict = {"yes": 0, "no": 1, "none": 2, "do not care": 3, "span": 4}
                value_types = []
                for domain_slot, value in turn_example.domain_slot_value:
                    if value in ["yes", "no", "none", "do not care"]:
                        value_types.append(value_type_dict[value])
                        value_span_starts.append(-1)
                        value_span_ends.append(-1)
                    else:
                        if value["from_which"] == "":
                            value_types.append(-1)  # Currently, we don't compute the gradient of no-span values.
                            value_span_starts.append(-1)
                            value_span_ends.append(-1)
                        else:
                            # value_types.append(4)
                            if value["from_which"] == "system":
                                tok2wp_map = sys_tok2wp_map
                                span_offset = 2 + len(cb_usr_pieces) - 1  # We generate input as [cls] user [sep] system [sep]
                                ini_pieces_len = len(sys_pieces)
                                input_pieces = cb_sys_pieces
                            else:
                                tok2wp_map = user_tok2wp_map
                                span_offset = 0
                                ini_pieces_len = len(user_pieces)
                                input_pieces = cb_usr_pieces

                            wp_start = tok2wp_map[value["span_start"]]
                            if value["span_end"] == len(tok2wp_map) - 1:
                                wp_end = ini_pieces_len - 1
                            elif value["span_end"] < len(tok2wp_map) - 1:
                                wp_end = tok2wp_map[value["span_end"] + 1] - 1
                            else:
                                raise RuntimeError()

                            wp_start += span_offset
                            wp_end += span_offset
                            assert wp_end >= wp_start
                            if wp_start >= len(input_pieces):
                                wp_start, wp_end = -1, -1
                                value_types.append(-1)
                            elif wp_end >= len(input_pieces):
                                wp_end = len(input_pieces) - 1
                                value_types.append(4)
                            else:
                                value_types.append(4)
                            value_span_starts.append(wp_start)
                            value_span_ends.append(wp_end)

                    assert len(value_types) - 1 == domain_slot_vocab[domain_slot]

                dialog_input_ids.append(turn_input_ids)
                dialog_input_mask.append(turn_input_mask)
                dialog_token_type_ids.append(turn_token_type_ids)
                dialog_value_starts.append(value_span_starts)
                dialog_value_ends.append(value_span_ends)
                dialog_value_types.append(value_types)

            dialog_mask = [1] * len(dialog_input_ids)
            while len(dialog_mask) < max_turn:
                dialog_input_ids.append(turn_input_id_padding)
                dialog_token_type_ids.append(turn_token_type_padding)
                dialog_input_mask.append(turn_mask_padding)
                dialog_value_starts.append([-1] * len(all_domain_slots))
                dialog_value_ends.append([-1] * len(all_domain_slots))
                dialog_value_types.append([-1] * len(all_domain_slots))
                dialog_mask.append(0)

            input_features.append(InputFeature(
                input_ids=dialog_input_ids,
                token_type_ids=dialog_token_type_ids,
                input_mask=dialog_input_mask,
                value_start=dialog_value_starts,
                value_end=dialog_value_ends,
                value_type=dialog_value_types,
                dialog_mask=dialog_mask
            ))

        # Generate input tensor
        input_ids = torch.LongTensor([f.input_ids for f in input_features])
        token_type_ids = torch.LongTensor([f.token_type_ids for f in input_features])
        input_mask = torch.LongTensor([f.input_mask for f in input_features])
        value_start = torch.LongTensor([f.values_start for f in input_features])
        value_end = torch.LongTensor([f.value_end for f in input_features])
        value_type = torch.LongTensor([f.value_type for f in input_features])
        dialog_mask = torch.LongTensor([f.dialog_mask for f in input_features])
        feature_indexes = torch.arange(input_ids.size(0), dtype=torch.long)

        tensor_data = TensorDataset(input_ids, token_type_ids, input_mask,
                                    value_start, value_end, value_type, dialog_mask, feature_indexes)
        if state == State.Train:
            data_sampler = RandomSampler(tensor_data)
        else:
            data_sampler = SequentialSampler(tensor_data)
        data_loader = DataLoader(tensor_data, batch_size=batch_size, sampler=data_sampler, pin_memory=True)

        other_data = {
            "domain_slot_input_ids": domain_slot_input_ids,
            "domain_slot_vocab": domain_slot_vocab,
            "examples": examples,
            "max_turn": max_turn
        }
        return data_loader, other_data
