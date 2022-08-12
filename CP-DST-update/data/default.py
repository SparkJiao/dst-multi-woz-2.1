import collections
import csv
import logging
import os

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from data.collator.dict2dict import DictTensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        import json

        # WOZ2.0 dataset
        if config.data_dir == "woz" or config.data_dir == "woz-turn":
            fp_ontology = open(os.path.join(config.data_dir, "ontology_dstc2_en.json"), "r")
            ontology = json.load(fp_ontology)
            ontology = ontology["informable"]
            del ontology["request"]
            for slot in ontology.keys():
                ontology[slot].append("do not care")
                ontology[slot].append("none")
            fp_ontology.close()

        # MultiWOZ dataset
        elif config.data_dir == "multiwoz2.1_5":

            if config.ontology is None:
                fp_ontology = open(os.path.join(config.data_dir, "ontology.json"), "r")
            else:
                fp_ontology = open(config.ontology, "r")

            ontology = json.load(fp_ontology)
            for slot in ontology.keys():
                # ontology[slot].append("none")
                """ Pop all 'none' and 'do not care' values """
                ontology[slot] = [x for x in ontology[slot] if x not in ["none", "do not care"]]

                """ FIX_UNDEFINED: Add undefined value. """
                ontology[slot].append("undefined")
            fp_ontology.close()

            self.include_dialogue_list = []
            if config.domain_list is not None:
                domain_list = json.load(open(config.domain_list, 'r'))
            else:
                domain_list = None
                self.include_dialogue_list = None

            if not config.target_slot == 'all':
                slot_idx = {'attraction': '0:1:2', 'hotel': '3:4:5:6:7:8:9:10:11:12',
                            'restaurant': '13:14:15:16:17:18:19', 'taxi': '20:21:22:23', 'train': '24:25:26:27:28:29'}
                target_slot = []
                if domain_list:
                    for dom, dom_ls in domain_list.items():
                        if dom != config.target_slot:
                            self.include_dialogue_list.extend(dom_ls)
                for key, value in slot_idx.items():
                    if key != config.target_slot:
                        target_slot.append(value)
                config.target_slot = ':'.join(target_slot)

            elif not config.train_single == 'all':
                slot_idx = {'attraction': '0:1:2', 'hotel': '3:4:5:6:7:8:9:10:11:12',
                            'restaurant': '13:14:15:16:17:18:19', 'taxi': '20:21:22:23', 'train': '24:25:26:27:28:29'}
                # target_slot = []
                if domain_list:
                    self.include_dialogue_list.extend(domain_list[config.train_single])
                for key, value in slot_idx.items():
                    if key == config.train_single:
                        # target_slot.append(value)
                        config.target_slot = value
                        break
                # config.target_slot = ':'.join(target_slot)

            self.include_dialogue_list = set(self.include_dialogue_list) if self.include_dialogue_list else None
        else:
            raise NotImplementedError()

        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        nslots = len(ontology.keys())
        target_slot = list(ontology.keys())
        if config.target_slot == 'all' and config.train_single == 'all':
            self.target_slot_idx = [*range(0, nslots)]
        else:
            self.target_slot_idx = sorted([int(x) for x in config.target_slot.split(':')])

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())
        for i, slot in enumerate(self.target_slot):
            if slot == "pricerange":
                self.target_slot[i] = "price range"

        self.reverse = config.reverse

        logger.info('Processor: target_slot')
        logger.info(self.target_slot)
        if self.include_dialogue_list:
            logger.info(f'Include dialogue amount in total: {len(self.include_dialogue_list)}')
        logger.info(f'Will reverse input: {self.reverse}')

    def get_train_examples(self, data_dir, accumulation=False, train_file=None):
        """See base class."""
        if train_file is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train-5.tsv")), "train", accumulation)
        else:
            return self._create_examples(self._read_tsv(train_file), "train", accumulation)

    def get_dev_examples(self, data_dir, accumulation=False, dev_file=None):
        """See base class."""
        if dev_file is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev-5.tsv")), "dev", accumulation)
        else:
            return self._create_examples(self._read_tsv(dev_file), "dev", accumulation)

    def get_test_examples(self, data_dir, accumulation=False, test_file=None):
        """See base class."""
        if test_file is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test-5.tsv")), "test", accumulation)
        else:
            return self._create_examples(self._read_tsv(test_file), "test", accumulation)

    def get_labels(self):
        """See base class."""
        return [self.ontology[slot] for slot in self.target_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            if self.include_dialogue_list and line[0] not in self.include_dialogue_list:
                continue
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

                if self.reverse:
                    text_a, text_b = text_b, text_a

            label = [line[4 + idx] for idx in self.target_slot_idx]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer: PreTrainedTokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    for labels in label_map:
        assert 'none' not in labels and 'do not care' not in labels
    slot_dim = len(label_list)

    features = []
    prev_dialogue_idx = None
    # all_padding = [0] * max_seq_length
    # all_padding_len = [0, 0]
    padding_a = "[PAD]"
    padding_b = "[PAD]"
    padding_model_inputs = tokenizer(padding_a, padding_b, padding=PaddingStrategy.MAX_LENGTH,
                                     truncation=TruncationStrategy.LONGEST_FIRST,
                                     max_length=max_seq_length)

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example.guid.split('-')[2]):
            max_turn = int(example.guid.split('-')[2])
    max_turn_length = min(max_turn + 1, max_turn_length)
    logger.info("max_turn_length = %d" % max_turn)

    undefined = 0
    none_values = 0
    care_values = 0
    ptr_values = 0

    for (ex_index, example) in tqdm(enumerate(examples)):
        text_a = example.text_a.replace('#', tokenizer.sep_token)
        text_b = example.text_b.replace('#', tokenizer.sep_token) if example.text_b else None

        model_inputs = tokenizer(text_a, text_b, padding=PaddingStrategy.MAX_LENGTH,
                                 truncation=TruncationStrategy.LONGEST_FIRST,
                                 max_length=max_seq_length)

        label_id = []
        answer_type = []
        label_info = 'label: '
        answer_type_info = 'answer type: '
        for i, label in enumerate(example.label):
            if label == 'dontcare':
                # label = 'do not care'
                raise RuntimeError()
            if label == 'undefined':
                undefined += 1
            if label == 'none':
                answer_type.append(0)
                label_id.append(-1)
                answer_type_info += '%s (id = %d) ' % ('none', 0)
                label_info += 'None '
                none_values += 1
            elif label == 'do not care':
                answer_type.append(1)
                label_id.append(-1)
                answer_type_info += '%s (id = %d) ' % ('do not care', 1)
                label_info += 'None '
                care_values += 1
            else:
                answer_type.append(2)
                label_id.append(label_map[i][label])
                answer_type_info += 'pick (id = 2) '
                label_info += '%s (id = %d) ' % (label, label_map[i][label])
                ptr_values += 1

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [{
                    "model_inputs": padding_model_inputs,
                    "answer_type": [-1] * slot_dim,
                    "label_id": [-1] * slot_dim,
                } for _ in range(max_turn_length - prev_turn_idx - 1)]
                # features += [InputFeatures(input_ids=all_padding,
                #                            input_len=all_padding_len,
                #                            answer_type=[-1] * slot_dim,
                #                            label_id=[-1] * slot_dim)] * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append({
                "model_inputs": model_inputs,
                "answer_type": answer_type,
                "label_id": label_id,
            })
            # features.append(
            #     InputFeatures(input_ids=input_ids,
            #                   input_len=input_len,
            #                   label_id=label_id,
            #                   answer_type=answer_type))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        # features += [InputFeatures(input_ids=all_padding,
        #                            input_len=all_padding_len,
        #                            answer_type=[-1] * slot_dim,
        #                            label_id=[-1] * slot_dim), ] * (max_turn_length - prev_turn_idx - 1)
        features += [{
            "model_inputs": padding_model_inputs,
            "answer_type": [-1] * slot_dim,
            "label_id": [-1] * slot_dim,
        } for _ in range(max_turn_length - prev_turn_idx - 1)]

    assert len(features) % max_turn_length == 0

    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
    # all_answer_type_ids = torch.tensor([f.answer_type for f in features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    all_input_ids = torch.tensor([f["model_inputs"]["input_ids"] for f in features], dtype=torch.long
                                 ).reshape(-1, max_turn_length, max_seq_length)
    all_attention_mask = torch.tensor([f["model_inputs"]["input_ids"] for f in features], dtype=torch.long
                                      ).reshape(-1, max_turn_length, max_seq_length)
    if "token_type_ids" in features[0]["model_inputs"]:
        all_token_type_ids = torch.tensor([f["model_inputs"]["token_type_ids"] for f in features], dtype=torch.long
                                          ).reshape(-1, max_turn_length, max_seq_length)
    else:
        all_token_type_ids = None
    all_answer_type_ids = torch.tensor([f["answer_type"] for f in features], dtype=torch.long).reshape(-1, max_turn_length, slot_dim)
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long).reshape(-1, max_turn_length, slot_dim)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    # all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    # all_input_len = all_input_len.view(-1, max_turn_length, 2)
    # all_answer_type_ids = all_answer_type_ids.view(-1, max_turn_length, slot_dim)
    # all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

    logger.info(f"There are {undefined} undefined values in total.")
    logger.info(f"Answer types:")
    logger.info(f"None: {none_values}")
    logger.info(f"Do not care: {care_values}")
    logger.info(f"PICK: {ptr_values}")
    logger.info(f"Dialogue amount: {all_input_ids.size(0)}")

    # if all_token_type_ids is not None:
    #     return all_input_ids, all_attention_mask, all_token_type_ids, all_answer_type_ids, all_label_ids
    # return all_input_ids, all_attention_mask, all_answer_type_ids, all_label_ids

    return DictTensorDataset({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "token_type_ids": all_token_type_ids,
        "answer_type_ids": all_answer_type_ids,
        "labels": all_label_ids
    })


def get_label_embedding(labels, tokenizer, device):
    # features = []
    # for label in labels:
    #     """ FIX_UNDEFINED: don't compute the embedding for 'undefined' """
    #     if label == "undefined":
    #         continue
    #     label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
    #     label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
    #     label_len = len(label_token_ids)
    #
    #     label_padding = [0] * (max_seq_length - len(label_token_ids))
    #     label_token_ids += label_padding
    #     assert len(label_token_ids) == max_seq_length
    #
    #     features.append((label_token_ids, label_len))
    """ FIX_UNDEFINED: don't compute the embedding for 'undefined' """
    filtered_labels = [label for label in labels if label != "undefined"]
    model_inputs = tokenizer(filtered_labels, padding=PaddingStrategy.LONGEST, return_tensors="pt")

    for k in model_inputs.keys():
        model_inputs[k] = model_inputs[k].to(device)
    # all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    # all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return model_inputs
