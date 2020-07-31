import argparse
import collections
import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.nn import Parameter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup, tokenization_roberta, tokenization_distilbert

try:
    from .global_logger import register_logger, register_summary_writer
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from global_logger import register_logger, register_summary_writer
    from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, answer_type, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.answer_type = answer_type
        self.label_id = label_id


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
        if config.data_dir == "data/woz" or config.data_dir == "data/woz-turn":
            fp_ontology = open(os.path.join(config.data_dir, "ontology_dstc2_en.json"), "r")
            ontology = json.load(fp_ontology)
            ontology = ontology["informable"]
            del ontology["request"]
            for slot in ontology.keys():
                ontology[slot].append("do not care")
                ontology[slot].append("none")
            fp_ontology.close()

        # MultiWOZ dataset
        elif config.data_dir == "data/multiwoz2.1_5":

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

            if not config.target_slot == 'all':
                slot_idx = {'attraction': '0:1:2', 'hotel': '3:4:5:6:7:8:9:10:11:12',
                            'restaurant': '13:14:15:16:17:18:19', 'taxi': '20:21:22:23', 'train': '24:25:26:27:28:29'}
                target_slot = []
                for key, value in slot_idx.items():
                    if key != config.target_slot:
                        target_slot.append(value)
                config.target_slot = ':'.join(target_slot)
            elif not config.train_single == 'all':
                slot_idx = {'attraction': '0:1:2', 'hotel': '3:4:5:6:7:8:9:10:11:12',
                            'restaurant': '13:14:15:16:17:18:19', 'taxi': '20:21:22:23', 'train': '24:25:26:27:28:29'}
                # target_slot = []
                for key, value in slot_idx.items():
                    if key == config.train_single:
                        # target_slot.append(value)
                        config.target_slot = value
                        break
                # config.target_slot = ':'.join(target_slot)

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    for labels in label_map:
        assert 'none' not in labels and 'do not care' not in labels
    slot_dim = len(label_list)

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

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
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
        tokens_b = None
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
            logger.info("label: " + label_info)
            logger.info("answer type: " + answer_type_info)

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           answer_type=[-1] * slot_dim,
                                           label_id=[-1] * slot_dim)] * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id,
                              answer_type=answer_type))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   answer_type=[-1] * slot_dim,
                                   label_id=[-1] * slot_dim), ] * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_answer_type_ids = torch.tensor([f.answer_type for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_answer_type_ids = all_answer_type_ids.view(-1, max_turn_length, slot_dim)
    all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

    logger.info(f"There are {undefined} undefined values in total.")
    logger.info(f"Answer types:")
    logger.info(f"None: {none_values}")
    logger.info(f"Do not care: {care_values}")
    logger.info(f"PICK: {ptr_values}")

    return all_input_ids, all_input_len, all_answer_type_ids, all_label_ids


def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        """ FIX_UNDEFINED: don't compute the embedding for 'undefined' """
        if label == "undefined":
            continue
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def get_pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print("Skip", name)
            continue
    return own_state


def make_aux_tensors(ids, len):
    token_type_ids = torch.zeros(ids.size(), dtype=torch.long)
    for i in range(len.size(0)):
        for j in range(len.size(1)):
            if len[i, j, 0] == 0:  # padding
                break
            elif len[i, j, 1] > 0:  # escape only text_a case
                start = len[i, j, 0]
                ending = len[i, j, 0] + len[i, j, 1]
                token_type_ids[i, j, start:ending] = 1
    attention_mask = ids > 0
    return token_type_ids, attention_mask


class ContrastiveDataset(Dataset):
    def __init__(self, slot_dim, *features, sample_slot_num=7, sample_key_num=3):
        self.features = features
        self.slot_dim = slot_dim
        self.slot_ids = list(range(slot_dim))
        self.sample_slot_num = sample_slot_num
        self.sample_key_num = sample_key_num
        self.dialog_num = self.features[0].size(0)
        self.valid_turn = (self.features[0].sum(dim=2) > 0).sum(dim=1)

    def __len__(self):
        return self.dialog_num

    def __getitem__(self, index):
        # query
        q_valid_turn = self.valid_turn[index]
        q_slot_idx = torch.LongTensor(random.sample(self.slot_ids, self.sample_slot_num))
        q_features = tuple([feature[index] for feature in self.features])

        # positive sample
        pos_valid_turn = self.valid_turn[index]
        pos_slot_idx = torch.LongTensor(random.sample(self.slot_ids, self.sample_slot_num))
        pos_features = [feature[index] for feature in self.features]

        # negative samples
        negative_sample_index = random.sample(list(range(index)) + list(range(index, self.dialog_num)),
                                              self.sample_key_num)
        assert index not in negative_sample_index
        neg_valid_turn = []
        neg_slot_idx = []
        neg_features = [[] for _ in range(len(self.features))]
        for neg_idx in negative_sample_index:
            neg_valid_turn.append(self.valid_turn[neg_idx])
            neg_slot_idx.append(torch.LongTensor(random.sample(self.slot_ids, self.sample_slot_num)).unsqueeze(0))
            for f_id, feature in enumerate(self.features):
                neg_features[f_id].append(feature[neg_idx].unsqueeze(0))
        neg_valid_turn = torch.cat(neg_valid_turn, dim=0)  # (sample_key_num)
        neg_slot_idx = torch.cat(neg_slot_idx, dim=0)  # (sample_key_num, sample_slot_num)
        neg_features = [torch.cat(f_list, dim=0) for f_list in neg_features]  # each feature: (sample_key_num, *size)

        label = torch.LongTensor([0])
        key_valid_turn = torch.cat([pos_valid_turn, neg_valid_turn], dim=0)
        key_slot_idx = torch.cat([pos_slot_idx.unsqueeze(0), neg_slot_idx], dim=0)
        key_features = [torch.cat([pos_feature.unsqueeze(0), neg_feature], dim=0)
                        for pos_feature, neg_feature in zip(pos_features, neg_features)]
        return (q_valid_turn, q_slot_idx, *q_features,
                key_valid_turn, key_slot_idx, *key_features,
                label)


def main():
    """
    This script fix the undefined values in ontology. Please search "FIX_UNDEFINED" to find the difference with `main-multislot.py`

    For undefined values, we address `undefined` as the last possible value for each slot and remove its embedding.
    Therefore we can mask their label ids as the ignore_index while computing loss,
    so the undefined value will not have an influence on gradient and while computing
    accuracy, they will be computed as wrong for the model never 'seen' undefined value.
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--ontology", default=None, type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
    parser.add_argument('--bert_name', default='bert-base-uncased')
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--predict_dir', default=None, type=str)
    parser.add_argument("--target_slot",
                        default='all',
                        type=str,
                        required=True,
                        help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'")
    parser.add_argument('--train_single', default='all', type=str)
    parser.add_argument("--tf_dir",
                        default='tensorboard',
                        type=str,
                        required=False,
                        help="Tensorboard directory")
    parser.add_argument("--nbt",
                        default='rnn',
                        type=str,
                        required=True,
                        help="nbt type: rnn or transformer")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_slot_length", type=int, default=8)
    parser.add_argument("--max_label_length",
                        default=18,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length",
                        default=22,
                        type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--attn_head',
                        type=int,
                        default=6,
                        help="the number of heads in multi-headed attention")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--distance_metric",
                        type=str,
                        default="cosine",
                        help="The metric for distance between label embeddings: cosine, euclidean.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total dialog batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=1,
                        type=int,
                        help="Total dialog batch size for validation.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total dialog batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for BertAdam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--per_eval_steps', type=int, default=2000, help='Per steps for evaluation')
    parser.add_argument("--patience",
                        default=10.0,
                        type=float,
                        help="The number of epochs to allow no further improvement.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--max_loss_scale', type=float, default=None)
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--fix_bert', default=False, action='store_true')

    parser.add_argument('--reverse', default=False, action='store_true')
    parser.add_argument('--weighted_cls', default=False, action='store_true')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--cls_loss_weight', default=1., type=float)
    parser.add_argument('--hidden_output', default=False, action='store_true')
    parser.add_argument('--dropout', default=None, type=float)

    parser.add_argument('--value_embedding_type', default='cls', type=str)
    parser.add_argument('--save_gate', default=False, action='store_true')
    parser.add_argument('--slot_res', default=None, type=str)
    parser.add_argument('--num_layers', default=0, type=int)
    parser.add_argument('--intermediate_size', default=3072, type=int)
    parser.add_argument('--add_query_attn', default=False, action='store_true')

    parser.add_argument('--efficient', default=False, action='store_true')
    parser.add_argument('--use_copy', default=False, action='store_true')
    parser.add_argument('--hard_copy', default=False, action='store_true')
    parser.add_argument('--add_interaction', default=False, action='store_true')
    parser.add_argument('--sinusoidal_embeddings', default=False, action='store_true')

    args = parser.parse_args()

    # CUDA setup
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # check output_dir
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)
        if args.local_rank == 0:
            torch.distributed.barrier()

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Tensorboard logging
    tb_file_name = '/'.join(args.output_dir.split('/')[1:])
    if args.local_rank in [-1, 0]:
        if not args.do_not_use_tensorboard:
            summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
        else:
            summary_writer = None

        register_summary_writer(summary_writer)

    # Logger
    tb_file_name = tb_file_name.replace('/', '-')
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    f_handler = logging.FileHandler(os.path.join(args.output_dir,
                                                 f'{tb_file_name}-{cur_time}-{args.local_rank}-output.log'))
    f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                             datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(f_handler)
    register_logger(logger)
    logger.setLevel(logging.DEBUG if args.local_rank in [-1, 0] else logging.WARN)
    logger.info(args)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(args)
    label_list = processor.get_labels()
    # num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type
    """ 
    FIX_UNDEFINED: Reduce 1 for 'undefined' label to avoid that the shape of initialized embedding is not compatible 
    with the shape of that defined in __init__() method of model.
    """
    num_labels = [len(labels) - 1 for labels in label_list]  # number of slot-values in each slot-type

    # tokenizer
    # vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    # if not os.path.exists(vocab_dir):
    #     raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation,
                                                      train_file=args.train_file)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation, dev_file=args.dev_file)

        ## Training utterances
        all_input_ids, all_input_len, all_answer_type_ids, all_label_ids = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_token_type_ids, all_input_mask = make_aux_tensors(all_input_ids, all_input_len)

        num_train_features = all_input_ids.size(0)
        num_train_steps = int(
            num_train_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_data = ContrastiveDataset(len(label_list), all_input_ids, all_token_type_ids, all_input_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      pin_memory=True, num_workers=4)

        ## Dev utterances
        all_input_ids_dev, all_input_len_dev, all_answer_type_ids_dev, all_label_ids_dev = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_token_type_ids_dev, all_input_mask_dev = make_aux_tensors(all_input_ids_dev, all_input_len_dev)
        num_dev_steps = int(all_input_ids_dev.size(0) / args.dev_batch_size * args.num_train_epochs)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        dev_data = ContrastiveDataset(len(label_list), all_input_ids_dev, all_token_type_ids_dev, all_input_mask_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt == 'transformer':
        from cls_aug_transformer import BeliefTracker
    elif args.nbt == 'inter':
        from cls_transformer_inter_pretrain import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    model = BeliefTracker(args, num_labels, device)

    if args.pretrain is not None:
        logger.info(f'Loading pre-trained model from {args.pretrain}')
        pre_train = torch.load(args.pretrain)
        pre_train_state_dict = get_pretrain(model, pre_train)
        model.load_state_dict(pre_train_state_dict)

    # if args.fp16:
    #     model.half()
    model.to(device)
    # model.to(torch.device('cuda:0'))

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get domain-slot-type embeddings
    slot_token_ids, slot_len = \
        get_label_embedding(processor.target_slot, args.max_slot_length, tokenizer, device)

    ## Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)
    # model.to(torch.device('cpu'))
    # model.to(device)

    # Prepare optimizer
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad and 'pooler' not in n]
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
            return optimizer_grouped_parameters

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        # no bias correct_bias for BERT
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                    num_training_steps=t_total)

        scaler = torch.cuda.amp.GradScaler()

        logger.info(optimizer)

        # Data parallelize when use multi-gpus
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True,
                                  disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()

                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
                # input_ids, token_type_ids, input_mask, answer_type_ids, label_ids = batch

                # Forward
                if n_gpu == 1:
                    with torch.cuda.amp.autocast():
                        loss, acc = model(*batch, n_gpu)
                else:
                    with torch.cuda.amp.autocast():
                        loss, acc = model(*batch, n_gpu)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                tr_loss += loss.item()
                nb_tr_examples += batch[0].size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    lr_this_step = scheduler.get_lr()[0]
                    if args.local_rank in [-1, 0] and summary_writer is not None:
                        summary_writer.add_scalar("Epoch", epoch, global_step)
                        summary_writer.add_scalar("Train/Loss", tr_loss / global_step, global_step)
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)

                    if args.local_rank in [-1, 0] and global_step % args.per_eval_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    ###############################################################################
    # Evaluation
    ###############################################################################

    if args.local_rank not in [-1, 0]:
        return
    # Load a trained model that you have fine-tuned
    predict_dir = args.predict_dir if args.predict_dir is not None else args.output_dir
    best_checkpoint = None
    best_dev_acc = 0.0
    best_loss = 10000000
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir, exist_ok=True)

    eval_examples = processor.get_dev_examples(args.data_dir, accumulation=False, dev_file=args.dev_file)
    all_input_ids, all_input_len, all_answer_type_ids, all_label_ids = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
    all_token_type_ids, all_input_mask = make_aux_tensors(all_input_ids, all_input_len)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_data = ContrastiveDataset(len(label_list), all_input_ids, all_token_type_ids, all_input_mask)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # for state_name in ['pytorch_model.bin', 'pytorch_model_loss.bin']:
    # for state_name in ['pytorch_model.bin']:
    for checkpoint in os.listdir(args.output_dir):
        if not os.path.isdir(os.path.join(args.output_dir, checkpoint)):
            continue
        state_name = checkpoint
        checkpoint = os.path.join(checkpoint, "pytorch_model.bin")

        if not os.path.exists(os.path.join(args.output_dir, checkpoint)):
            continue
        model = BeliefTracker(args, num_labels, device)
        logger.info(f'Loading saved model from {os.path.join(args.output_dir, checkpoint)}')
        output_model_file = os.path.join(args.output_dir, checkpoint)

        # in the case that slot and values are different between the training and evaluation
        ptr_model = torch.load(output_model_file)
        # Initialize slot value look up to avoid mismatch
        for k in list(ptr_model.keys()):
            if 'slot_lookup' in k or 'value_lookup' in k:
                ptr_model.pop(k)

        if n_gpu == 1:
            state = model.state_dict()
            state.update(ptr_model)
            state = get_pretrain(model, state)
            model.load_state_dict(state)
        else:
            print("Evaluate using only one device!")
            model.module.load_state_dict(ptr_model)

        model.to(device)
        model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

        total_loss = 0
        total_acc = 0
        # Evaluation on dev set
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

            model.eval()

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        if n_gpu == 1:
                            loss, acc = model(*batch, n_gpu)
                        else:
                            loss, acc = model(*batch, n_gpu)
                total_loss += loss * batch[0].size(0)
                total_acc += acc * batch[0].size(0)

            total_loss = total_loss / all_input_ids.size(0)
            total_acc = total_acc / all_input_ids.size(0)

            out_file_name = f'eval_results_{state_name}'
            if args.target_slot == 'all':
                out_file_name += '_all'
            output_eval_file = os.path.join(predict_dir, "%s.txt" % out_file_name)

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                logger.info(f"Accuracy: {total_acc}")
                logger.info(f"Loss: {total_loss}")

                writer.write("%s = %s\n" % ('Accuracy', str(total_acc)))
                writer.write("%s = %s\n" % ('Loss', str(total_loss)))

            if total_loss < best_loss:
                best_loss = total_loss
                best_checkpoint = state_name

    logger.info(f'Best checkpoint: {best_checkpoint}')


if __name__ == "__main__":
    main()
