import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
from typing import List, Dict, Any
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, dialog, queries, labels, samples, start_index):
        self.guid = guid
        self.dialog = dialog
        self.queries = queries
        self.labels = labels
        self.samples = samples
        self.start_index = start_index


class InputFeatures(object):
    def __init__(self, dialog_input_ids, dialog_token_type_ids, dialog_input_mask,
                 query_input_ids, query_token_type_ids, query_input_mask, labels, query_mask,
                 sample_input_ids, sample_token_type_ids, sample_input_mask, start_index):
        self.dialog_input_ids = dialog_input_ids
        self.dialog_token_type_ids = dialog_token_type_ids
        self.dialog_input_mask = dialog_input_mask

        self.query_input_ids = query_input_ids
        self.query_token_type_ids = query_token_type_ids
        self.query_input_mask = query_input_mask
        self.labels = labels
        self.query_mask = query_mask

        self.sample_input_ids = sample_input_ids
        self.sample_token_type_ids = sample_token_type_ids
        self.sample_input_mask = sample_input_mask

        self.start_index = start_index


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train_nrs.json"))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "dev_nrs.json"))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test_nrs.json"))

    @staticmethod
    def _create_examples(input_file):
        """Creates examples for the training and dev sets."""
        data = json.load(open(input_file, 'r'))
        examples: List[InputExample] = []
        for dialog_idx, dialog_examples in tqdm(enumerate(data), desc='Reading examples...'):
            start_index = dialog_examples['querys'][0]['start_index']
            for x in dialog_examples['querys']:
                assert start_index == x['start_index']
            examples.append(InputExample(
                guid=f'd#{dialog_idx}',
                dialog=dialog_examples['dialog'],
                queries=[x['query'] for x in dialog_examples['querys']],
                labels=[x['response_id'] for x in dialog_examples['querys']],
                samples=[x['samples'] for x in dialog_examples['querys']],
                start_index=start_index
            ))
        return examples


def convert_examples_to_features(examples, max_seq_length, max_query_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    max_turns = 0
    max_query_num = 0
    sample_num = len(examples[0].samples[0])
    start_index = examples[0].start_index
    for example in examples:
        max_turns = max(max_turns, len(example.dialog))
        max_query_num = max(max_query_num, len(example.queries))
        for sample in example.samples:
            assert sample_num == len(sample)
        assert start_index == example.start_index
    token_padding = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    token_type_padding = [0, 1]
    token_mask_padding = [1, 1]
    pair_padding = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[SEP]'])
    pair_type_padding = [0, 0, 1]
    pair_mask_padding = [1, 1, 1]
    truncate_s = 0
    truncate_q = 0
    for example_idx, example in tqdm(enumerate(examples), desc='Converting features...'):
        # Dialog
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_input_mask = []
        for turn in example.dialog:
            tokens_a = tokenizer.tokenize(turn[0])
            tokens_b = tokenizer.tokenize(turn[1])
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'])
            type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                type_ids.append(0)
                mask.append(0)
            dialog_input_ids.append(input_ids)
            dialog_token_type_ids.append(type_ids)
            dialog_input_mask.append(mask)
        while len(dialog_input_ids) < max_turns:
            dialog_input_ids.append(pair_padding + [0] * (max_seq_length - 3))
            dialog_token_type_ids.append(pair_type_padding + [0] * (max_seq_length - 3))
            dialog_input_mask.append(pair_mask_padding + [0] * (max_seq_length - 3))

        # Query
        query_input_ids = []
        query_token_type_ids = []
        query_input_mask = []
        query_mask = []
        for query in example.queries:
            # query_tokens = tokenizer.tokenize(query)
            # if len(query_tokens) > max_query_length - 2:
            #     query_tokens = query_tokens[-(max_seq_length - 2):]
            #     truncate_q += 1
            # input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + query_tokens + ['[SEP]'])
            # type_ids = [0] * len(input_ids)
            # mask = [1] * len(input_ids)
            # while len(input_ids) < max_query_length:
            #     input_ids.append(0)
            #     type_ids.append(0)
            #     mask.append(0)
            input_ids, type_ids, mask, truncates = _generate_single_input(query, max_query_length, tokenizer)
            truncate_q += truncates
            query_input_ids.append(input_ids)
            query_token_type_ids.append(type_ids)
            query_input_mask.append(mask)
            query_mask.append(1)
        while len(query_input_ids) < max_query_num:
            query_input_ids.append(token_padding + [0] * (max_query_length - 2))
            query_token_type_ids.append(token_type_padding + [0] * (max_query_length - 2))
            query_input_mask.append(token_mask_padding + [0] * (max_query_length - 2))
            query_mask.append(0)

        # Samples
        sample_input_ids = []
        sample_token_type_ids = []
        sample_input_mask = []
        for samples in example.samples:
            input_ids = []  # sample_num * max_query_length
            type_ids = []
            mask = []
            for sample in samples:
                tmp_input, tmp_type, tmp_mask, tmp_truncate = _generate_single_input(sample, max_query_length, tokenizer)
                input_ids.append(tmp_input)
                type_ids.append(tmp_type)
                mask.append(tmp_mask)
                truncate_s += tmp_truncate
            sample_input_ids.append(input_ids)
            sample_token_type_ids.append(type_ids)
            sample_input_mask.append(mask)
        sample_padding = (
            [token_padding + [0] * (max_query_length - 2)] * sample_num,
            [token_type_padding + [0] * (max_query_length - 2)] * sample_num,
            [token_mask_padding + [0] * (max_query_length - 2)] * sample_num
        )
        while len(sample_input_ids) < max_query_num:
            sample_input_ids.append(sample_padding[0])
            sample_token_type_ids.append(sample_padding[1])
            sample_input_mask.append(sample_padding[2])

        labels = example.labels + [-1] * (max_query_num - len(example.labels))

        features.append(InputFeatures(dialog_input_ids=dialog_input_ids, dialog_token_type_ids=dialog_token_type_ids,
                                      dialog_input_mask=dialog_input_mask, query_input_ids=query_input_ids,
                                      query_token_type_ids=query_token_type_ids, query_input_mask=query_input_mask,
                                      query_mask=query_mask, labels=labels,
                                      sample_input_ids=sample_input_ids, sample_token_type_ids=sample_token_type_ids,
                                      sample_input_mask=sample_input_mask, start_index=start_index))

    logger.info(f'Truncate {truncate_q} queries and {truncate_s} samples.')
    return features


class NRSDataset(Dataset):
    def __init__(self, input_features):
        super(NRSDataset, self).__init__()
        self.inputs = self.generate_inputs(input_features)
        self.length = self.inputs["dialog_input_ids"].size(0)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.inputs.items()}
        return item

    def __len__(self):
        return self.length

    @staticmethod
    def generate_inputs(features: List[InputFeatures]):
        inputs = {
            # [data_len, max_turns, max_seq_length]
            "dialog_input_ids": torch.LongTensor([f.dialog_input_ids for f in features]),
            "dialog_token_type_ids": torch.LongTensor([f.dialog_token_type_ids for f in features]),
            "dialog_input_mask": torch.LongTensor([f.dialog_input_mask for f in features]),
            # [data_len, max_query_num, max_query_length]
            "query_input_ids": torch.LongTensor([f.query_input_ids for f in features]),
            "query_token_type_ids": torch.LongTensor([f.query_token_type_ids for f in features]),
            "query_input_mask": torch.LongTensor([f.query_input_mask for f in features]),
            # [data_len, max_query_num, sample_num, max_query_length]
            "sample_input_ids": torch.LongTensor([f.sample_input_ids for f in features]),
            "sample_token_type_ids": torch.LongTensor([f.sample_token_type_ids for f in features]),
            "sample_input_mask": torch.LongTensor([f.sample_input_mask for f in features]),
            # [data_len, max_query_num]
            "query_mask": torch.LongTensor([f.query_mask for f in features]),
            "labels": torch.LongTensor([f.labels for f in features]),
            # [data_len]
            "start_index": torch.LongTensor([f.start_index for f in features])
        }
        _start_index = inputs["start_index"][0]
        for x in _start_index:
            assert x == _start_index
        return inputs


def _generate_single_input(sequence, max_seq_length, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    truncate = 0
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[-(max_seq_length - 2):]
        truncate += 1
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    type_ids = [0] * len(input_ids)
    mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        type_ids.append(0)
        mask.append(0)
    return input_ids, type_ids, mask, truncate


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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
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
    parser.add_argument("--fix_utterance_encoder",
                        action='store_true',
                        help="Do not train BERT utterance encoder")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_query_length",
                        default=32,
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
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--attn_head',
                        type=int,
                        default=4,
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
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    args = parser.parse_args()

    # check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        tb_file_name = args.output_dir.split('/')[1]
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    # Logger
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt" % (tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

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

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        dev_examples = processor.get_dev_examples(args.data_dir)

        # Training utterances
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, args.max_query_length, tokenizer)
        train_data = NRSDataset(train_features)
        num_train_steps = int(len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Training Statistics *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

        # Dev utterances
        dev_features = convert_examples_to_features(dev_examples, args.max_seq_length, args.max_query_length, tokenizer)
        dev_data = NRSDataset(dev_features)
        num_dev_steps = int(len(dev_data) / args.dev_batch_size * args.num_train_epochs)

        logger.info("***** Validation Statistics *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt == 'rnn':
        from BeliefTrackerSlotQueryMultiSlot_F import BeliefTracker
    elif args.nbt == 'transformer':
        from BeliefTrackerSlotQueryMultiSlotTransformer import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    model = BeliefTracker(args, num_labels, device)
    if args.fp16:
        model.half()
    model.to(device)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get domain-slot-type embeddings
    slot_token_ids, slot_len = \
        get_label_embedding(processor.target_slot, args.max_label_length, tokenizer, device)

    ## Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

    # Data parallelize when use multi-gpus
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters

        if n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
        logger.info(optimizer)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch

                # Forward
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    if n_gpu == 1:
                        for i, slot in enumerate(processor.target_slot):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i], global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i], global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    # if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    # else:
                    #     if summary_writer is not None:
                    #         summary_writer.add_scalar("Train/LearningRate", optimizer.get_lr()[0], global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if n_gpu == 1:
                        loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
                    else:
                        loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples

            if n_gpu == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                if n_gpu == 1:
                    for i, slot in enumerate(processor.target_slot):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'), dev_loss_slot[i] / nb_dev_examples,
                                                  global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i], global_step)

            dev_loss = round(dev_loss, 6)
            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info(
                    "*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (last_update, best_loss, best_acc))
            else:
                logger.info(
                    "*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (epoch, dev_loss, dev_acc))

            if last_update + args.patience <= epoch:
                break

    ###############################################################################
    # Evaluation
    ###############################################################################
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model = BeliefTracker(args, num_labels, device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # in the case that slot and values are different between the training and evaluation
    ptr_model = torch.load(output_model_file)

    if n_gpu == 1:
        state = model.state_dict()
        state.update(ptr_model)
        model.load_state_dict(state)
    else:
        print("Evaluate using only one device!")
        model.module.load_state_dict(ptr_model)

    model.to(device)

    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                      'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0}

        for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                label_ids = label_ids.unsuqeeze(0)

            with torch.no_grad():
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, n_gpu)
                else:
                    loss, _, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, n_gpu)
                    nbatch = label_ids.size(0)
                    nslot = pred_slot.size(3)
                    pred_slot = pred_slot.view(nbatch, -1, nslot)

            accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

            nb_eval_ex = (label_ids[:, :, 0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1

            if n_gpu == 1:
                eval_loss += loss.item() * nb_eval_ex
                eval_accuracy += acc.item() * nb_eval_ex
                if eval_loss_slot is None:
                    eval_loss_slot = [l * nb_eval_ex for l in loss_slot]
                    eval_acc_slot = acc_slot * nb_eval_ex
                else:
                    for i, l in enumerate(loss_slot):
                        eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                    eval_acc_slot += acc_slot * nb_eval_ex
            else:
                eval_loss += sum(loss) * nb_eval_ex
                eval_accuracy += sum(acc) * nb_eval_ex

        eval_loss = eval_loss / nb_eval_examples
        eval_accuracy = eval_accuracy / nb_eval_examples
        if n_gpu == 1:
            eval_acc_slot = eval_acc_slot / nb_eval_examples

        loss = tr_loss / nb_tr_steps if args.do_train else None

        if n_gpu == 1:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss,
                      'eval_loss_slot': '\t'.join([str(val / nb_eval_examples) for val in eval_loss_slot]),
                      'eval_acc_slot': '\t'.join([str((val).item()) for val in eval_acc_slot])
                      }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss
                      }

        out_file_name = 'eval_results'
        if args.target_slot == 'all':
            out_file_name += '_all'
        output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
            f.write(
                'joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
            f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
                (accuracies['joint7'] / accuracies['num_turn']).item(),
                (accuracies['slot7'] / accuracies['num_slot7']).item(),
                (accuracies['joint5'] / accuracies['num_turn']).item(),
                (accuracies['slot5'] / accuracies['num_slot5']).item(),
                (accuracies['joint_rest'] / accuracies['num_turn']).item(),
                (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
            ))


def eval_all_accs(pred_slot, labels, accuracies):
    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, labels)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:, :, 18:25], labels[:, :, 18:25])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    pred_slot5 = torch.cat((pred_slot[:, :, 0:3], pred_slot[:, :, 8:]), 2)
    label_slot5 = torch.cat((labels[:, :, 0:3], labels[:, :, 8:]), 2)

    # 5 domains (excluding bus and hotel domain)
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    return accuracies


if __name__ == "__main__":
    main()
