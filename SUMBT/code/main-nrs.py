import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
from typing import List, Dict, Any
import json, pickle

from allennlp.training.metrics import CategoricalAccuracy, Average
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
    def __init__(self, guid, dialog, query, label, samples, end_index):
        self.guid = guid
        self.dialog = dialog
        self.query = query
        self.label = label
        self.samples = samples
        self.end_index = end_index


class InputFeatures(object):
    def __init__(self, dialog_input_ids, dialog_token_type_ids, dialog_input_mask, dialog_mask,
                 query_input_ids, query_token_type_ids, query_input_mask, label,
                 sample_input_ids, sample_token_type_ids, sample_input_mask, end_index):
        self.dialog_input_ids = dialog_input_ids
        self.dialog_token_type_ids = dialog_token_type_ids
        self.dialog_input_mask = dialog_input_mask
        self.dialog_mask = dialog_mask

        self.query_input_ids = query_input_ids
        self.query_token_type_ids = query_token_type_ids
        self.query_input_mask = query_input_mask
        self.label = label

        self.sample_input_ids = sample_input_ids
        self.sample_token_type_ids = sample_token_type_ids
        self.sample_input_mask = sample_input_mask

        self.end_index = end_index


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
            examples.append(InputExample(
                guid=f'd#{dialog_idx}',
                dialog=dialog_examples['dialog'],
                query=dialog_examples['query'],
                label=dialog_examples['response_id'],
                samples=dialog_examples['samples'],
                end_index=dialog_examples['end_index']
            ))
        return examples


def convert_examples_to_features(examples: List[InputExample], max_seq_length, max_query_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    max_turns = 0
    sample_num = len(examples[0].samples)
    # start_index = examples[0].start_index
    for example in examples:
        max_turns = max(max_turns, len(example.dialog))
        # max_query_num = max(max_query_num, len(example.queries))
        # for sample in example.samples:
        assert sample_num == len(example.samples)
        # assert start_index == example.start_index
    token_padding = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    token_type_padding = [0, 1]
    token_mask_padding = [1, 1]
    pair_padding = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[SEP]'])
    pair_type_padding = [0, 0, 1]
    pair_mask_padding = [1, 1, 1]
    truncate_s = 0
    truncate_q = 0
    for example_idx, example in enumerate(tqdm(examples, desc='Converting features...')):
        # Dialog
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_input_mask = []
        dialog_mask = []
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
            dialog_mask.append(1)
        while len(dialog_input_ids) < max_turns:
            dialog_input_ids.append(pair_padding + [0] * (max_seq_length - 3))
            dialog_token_type_ids.append(pair_type_padding + [0] * (max_seq_length - 3))
            dialog_input_mask.append(pair_mask_padding + [0] * (max_seq_length - 3))
            dialog_mask.append(0)

        query_input_ids, query_token_type_ids, query_input_mask, truncates = _generate_single_input(example.query,
                                                                                                    max_query_length,
                                                                                                    tokenizer)
        truncate_q += truncates

        # Samples
        sample_input_ids = []
        sample_token_type_ids = []
        sample_input_mask = []
        for sample in example.samples:
            input_ids, type_ids, mask, truncates = _generate_single_input(sample, max_query_length, tokenizer)
            sample_input_ids.append(input_ids)
            sample_token_type_ids.append(type_ids)
            sample_input_mask.append(mask)
            truncate_s += truncates

        features.append(InputFeatures(dialog_input_ids=dialog_input_ids, dialog_token_type_ids=dialog_token_type_ids,
                                      dialog_input_mask=dialog_input_mask, dialog_mask=dialog_mask,
                                      query_input_ids=query_input_ids, query_token_type_ids=query_token_type_ids,
                                      query_input_mask=query_input_mask, sample_input_ids=sample_input_ids,
                                      sample_token_type_ids=sample_token_type_ids, sample_input_mask=sample_input_mask,
                                      label=example.label, end_index=example.end_index))

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
            # [data_len, max_turns]
            "dialog_mask": torch.LongTensor([f.dialog_mask for f in features]),
            # [data_len, max_query_length]
            "query_input_ids": torch.LongTensor([f.query_input_ids for f in features]),
            "query_token_type_ids": torch.LongTensor([f.query_token_type_ids for f in features]),
            "query_input_mask": torch.LongTensor([f.query_input_mask for f in features]),
            # [data_len, sample_num, max_query_length]
            "sample_input_ids": torch.LongTensor([f.sample_input_ids for f in features]),
            "sample_token_type_ids": torch.LongTensor([f.sample_token_type_ids for f in features]),
            "sample_input_mask": torch.LongTensor([f.sample_input_mask for f in features]),
            # [data_len]
            "end_index": torch.LongTensor([f.end_index for f in features]),
            "label": torch.LongTensor([f.label for f in features])
        }
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
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--model_id', type=int, default=1)

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

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        dev_examples = processor.get_dev_examples(args.data_dir)

        # Training utterances
        cached_file_name = f'{os.path.join(args.data_dir, "train_nrs.json")}-{args.max_seq_length}-' \
                           f'{args.max_query_length}-NSR'
        try:
            with open(cached_file_name, 'rb') as f:
                train_features = pickle.load(f)
        except FileNotFoundError:
            train_features = convert_examples_to_features(train_examples, args.max_seq_length, args.max_query_length,
                                                          tokenizer)
            with open(cached_file_name, 'wb') as f:
                pickle.dump(train_features, f)

        train_data = NRSDataset(train_features)
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Training Statistics *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=8)

        # Dev utterances
        dev_features = convert_examples_to_features(dev_examples, args.max_seq_length, args.max_query_length, tokenizer)
        dev_data = NRSDataset(dev_features)
        num_dev_steps = int(len(dev_data) / args.dev_batch_size * args.num_train_epochs)

        logger.info("***** Validation Statistics *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size, num_workers=8)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt == 'rnn':
        if args.model_id == 1:
            from BeliefTrackerNRSPretrain import BeliefTracker
        elif args.model_id == 2:
            from BeliefTrackerNRSPretrain2 import BeliefTracker
        else:
            raise RuntimeError()
    # elif args.nbt == 'transformer':
    #     from BeliefTrackerSlotQueryMultiSlotTransformer import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    model = BeliefTracker(args, device)
    # if args.fp16:
    #     model.half()
    model.to(device)

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

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

        if args.fp16:
            # try:
            #     from apex.optimizers import FP16_Optimizer
            #     from apex.optimizers import FusedAdam
            # except ImportError:
            #     raise ImportError(
            #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            #
            # optimizer = FusedAdam(optimizer_grouped_parameters,
            #                       lr=args.learning_rate,
            #                       bias_correction=False,
            #                       max_grad_norm=1.0)
            # if args.loss_scale == 0:
            #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            # else:
            #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info(optimizer)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None
        eval_accuracy = CategoricalAccuracy()
        eval_loss = Average()

        for epoch in range(int(args.num_train_epochs)):
            # Train
            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                batch = {k: v.to(device) for k, v in batch.items()}

                model_output = model(**batch)
                loss = model_output['loss']
                # Forward
                if n_gpu > 1:
                    # average to multi-gpus
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    # if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    # else:
                    #     if summary_writer is not None:
                    #         summary_writer.add_scalar("Train/LearningRate", optimizer.get_lr()[0], global_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                        summary_writer.add_scalar("Train/Loss", loss.item(), global_step)

            # Perform evaluation on validation dataset
            model.eval()

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation", dynamic_ncols=True)):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    model_output = model(**batch)
                    eval_loss(model_output['loss'].item())
                    eval_accuracy(model_output['logits'], batch['label'])

            # tensorboard logging
            dev_loss = eval_loss.get_metric(reset=True)
            dev_acc = eval_accuracy.get_metric(reset=True)
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)

            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                model_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                # if args.do_train:
                #     if n_gpu == 1:
                #         torch.save(model.state_dict(), output_model_file)
                #     else:
                #         torch.save(model.module.state_dict(), output_model_file)
                torch.save(model_to_save, output_model_file)

                last_update = epoch
                best_loss = dev_loss

                logger.info(
                    "*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (
                        last_update, best_loss, dev_acc))
            else:
                logger.info(
                    "*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (
                        epoch, dev_loss, dev_acc))

            if last_update + args.patience <= epoch:
                break

    # ###############################################################################
    # # Evaluation
    # ###############################################################################
    # # Load a trained model that you have fine-tuned
    # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    # model = BeliefTracker(args, num_labels, device)
    #
    # if args.local_rank != -1:
    #     try:
    #         from apex.parallel import DistributedDataParallel as DDP
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #
    # # in the case that slot and values are different between the training and evaluation
    # ptr_model = torch.load(output_model_file)
    #
    # if n_gpu == 1:
    #     state = model.state_dict()
    #     state.update(ptr_model)
    #     model.load_state_dict(state)
    # else:
    #     print("Evaluate using only one device!")
    #     model.module.load_state_dict(ptr_model)
    #
    # model.to(device)

    # Evaluation
    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):


if __name__ == "__main__":
    main()
