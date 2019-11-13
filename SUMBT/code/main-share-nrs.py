import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
import json
from typing import List, Any
import pickle

import numpy as np
import torch
from torch.nn import Parameter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from allennlp.training.metrics import CategoricalAccuracy, Average

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, query, options, answer):
        self.guid = guid
        self.context = context
        self.query = query
        self.options = options
        self.answer = answer


class InputFeatures(object):
    def __init__(self, dialog_input_ids, dialog_token_type_ids, dialog_input_mask, dialog_mask,
                 option_input_ids, option_token_type_ids, option_input_mask,
                 label=1):
        self.dialog_input_ids = dialog_input_ids
        self.dialog_token_type_ids = dialog_token_type_ids
        self.dialog_input_mask = dialog_input_mask
        self.dialog_mask = dialog_mask

        self.option_input_ids = option_input_ids
        self.option_token_type_ids = option_token_type_ids
        self.option_input_mask = option_input_mask

        self.label = label


class Processor:
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self):
        super(Processor, self).__init__()

    def get_train_examples(self, train_file):
        return self._create_examples(train_file)

    def get_dev_examples(self, dev_file):
        return self._create_examples(dev_file)

    def get_test_examples(self, test_file):
        return self._create_examples(test_file)

    @staticmethod
    def _create_examples(file):
        data = json.load(open(file, 'r'))
        examples: List[InputExample] = []
        for instance_id, instance in tqdm(enumerate(data), desc="Reading examples"):
            example = InputExample(
                guid=f"d#{instance_id}",
                context=instance["context"],
                query=instance["query"],
                options=instance["options"],
                answer=instance["answer"]
            )
            examples.append(example)
        return examples


def convert_examples_to_features(examples: List[InputExample], max_seq_length: int, max_sample_length: int, tokenizer: BertTokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    max_turn = max(map(lambda x: len(x.context) + 1, examples))

    features: List[InputFeatures] = []
    pair_padding = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[SEP]']) + [0] * (max_seq_length - 3)
    pair_type_padding = [0, 0, 1] + [0] * (max_seq_length - 3)
    pair_mask_padding = [1, 1, 1] + [0] * (max_seq_length - 3)
    sample_truncate = 0
    for example in tqdm(examples, desc='Converting examples to features', total=len(examples)):
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_input_mask = []
        dialog_mask = []

        for turn_id, turn in enumerate(example.context):
            sys_utt, usr_utt = turn[0], turn[1]
            sys_tokens = tokenizer.tokenize(sys_utt)
            usr_tokens = tokenizer.tokenize(usr_utt)
            _truncate_seq_pair(usr_tokens, sys_tokens, max_seq_length - 3)
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + usr_tokens + ['[SEP]'] + sys_tokens + ['[SEP]'])
            type_ids = [0] * (len(usr_tokens) + 2) + [1] * (len(sys_tokens) + 1)
            mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                type_ids.append(0)
                mask.append(0)
            dialog_input_ids.append(input_ids)
            dialog_token_type_ids.append(type_ids)
            dialog_input_mask.append(mask)
            dialog_mask.append(1)

        query_tokens = tokenizer.tokenize(example.query)
        # query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        if len(query_tokens) > max_seq_length - 3:
            query_tokens = query_tokens[:(max_seq_length - 3)]
        query_input_ids = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'] + query_tokens + ['[SEP]'])
        query_token_type_ids = [0] * 2 + [1] * (len(query_tokens) + 1)
        query_input_mask = [1] * len(query_input_ids)
        while len(query_input_ids) < max_seq_length:
            query_input_ids.append(0)
            query_token_type_ids.append(0)
            query_input_mask.append(0)

        dialog_input_ids.append(query_input_ids)
        dialog_token_type_ids.append(query_token_type_ids)
        dialog_input_mask.append(query_input_mask)
        dialog_mask.append(1)

        while len(dialog_mask) < max_turn:
            dialog_input_ids.append(pair_padding)
            dialog_token_type_ids.append(pair_type_padding)
            dialog_input_mask.append(pair_mask_padding)
            dialog_mask.append(0)

        sample_input_ids = []
        sample_token_type_ids = []
        sample_input_mask = []
        for sample in example.options:
            input_ids, type_ids, mask, truncate = _generate_single_input(sample, max_sample_length, tokenizer)
            sample_input_ids.append(input_ids)
            sample_token_type_ids.append(type_ids)
            sample_input_mask.append(mask)
            sample_truncate += truncate

        answer_input_ids, answer_type_ids, answer_mask, truncate = _generate_single_input(example.answer, max_sample_length, tokenizer)
        sample_input_ids = [answer_input_ids] + sample_input_ids
        sample_token_type_ids = [answer_type_ids] + sample_token_type_ids
        sample_input_mask = [answer_mask] + sample_input_mask
        sample_truncate += truncate

        features.append(InputFeatures(
            dialog_input_ids=dialog_input_ids,
            dialog_token_type_ids=dialog_token_type_ids,
            dialog_input_mask=dialog_input_mask,
            dialog_mask=dialog_mask,
            option_input_ids=sample_input_ids,
            option_token_type_ids=sample_token_type_ids,
            option_input_mask=sample_input_mask,
            label=0
        ))

    logger.info(f"Truncate {sample_truncate} options and answers in total.")
    return features


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


class NRSDataset(Dataset):
    def __init__(self, input_features: List[InputFeatures]):
        self.inputs = self.generate_inputs(input_features)
        self.length = self.inputs["dialog_input_ids"].size(0)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.inputs.items()}
        return item

    def __len__(self):
        return self.length

    @staticmethod
    def generate_inputs(features: List[InputFeatures]):
        unpack_f = lambda name: torch.LongTensor([f.__getattribute__(name) for f in features])

        inputs = {
            "dialog_input_ids": unpack_f("dialog_input_ids"),
            "dialog_token_type_ids": unpack_f("dialog_token_type_ids"),
            "dialog_input_mask": unpack_f("dialog_input_mask"),
            "dialog_mask": unpack_f("dialog_mask"),
            "option_input_ids": unpack_f("option_input_ids"),
            "option_token_type_ids": unpack_f("option_token_type_ids"),
            "option_input_mask": unpack_f("option_input_mask"),
            "label": unpack_f("label")
        }

        data_num, max_turn, seq_len = inputs["dialog_input_ids"].size()
        assert inputs["dialog_token_type_ids"].size() == inputs["dialog_input_mask"].size() == (data_num, max_turn, seq_len)
        assert inputs["dialog_mask"].size() == (data_num, max_turn)
        _, sample_num, sample_len = inputs["option_input_ids"].size()
        assert sample_num == 6
        assert inputs["option_input_mask"].size() == inputs["option_token_type_ids"].size() == (data_num, sample_num, sample_len)
        assert inputs["label"].size(-1) == data_num

        return inputs


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
    parser.add_argument("--target_slot",
                        default='all',
                        type=str,
                        required=True,
                        help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'")
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

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_sample_length", type=int, default=32)
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
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
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
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_id', type=int, default=1)
    parser.add_argument('--use_query', default=False, action='store_true')
    parser.add_argument('--fix_bert', default=False, action='store_true')
    parser.add_argument('--reduce_layers', default=0, type=int)

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
    processor = Processor()

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(train_file=args.train_file)
        dev_examples = processor.get_dev_examples(dev_file=args.dev_file)

        cached_train_features_file = f"{args.train_file}-{args.max_seq_length}-{args.max_sample_length}"
        try:
            with open(cached_train_features_file, 'rb') as f:
                train_features = pickle.load(f)
        except FileNotFoundError:
            train_features = convert_examples_to_features(train_examples, tokenizer=tokenizer,
                                                          max_seq_length=args.max_seq_length, max_sample_length=args.max_sample_length)
            with open(cached_train_features_file, 'wb') as f:
                pickle.dump(train_features, f)

        num_train_features = len(train_features)
        num_train_steps = int(num_train_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_data = NRSDataset(train_features)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        cached_dev_features_file = f"{args.dev_file}-{args.max_seq_length}-{args.max_sample_length}"
        try:
            with open(cached_dev_features_file, 'rb') as f:
                dev_features = pickle.load(f)
        except FileNotFoundError:
            dev_features = convert_examples_to_features(dev_examples, tokenizer=tokenizer,
                                                        max_seq_length=args.max_seq_length, max_sample_length=args.max_sample_length)
            with open(cached_dev_features_file, 'wb') as f:
                pickle.dump(dev_features, f)
        num_dev_steps = int(len(dev_features) / args.dev_batch_size * args.num_train_epochs)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        logger.info("  Num steps = %d", num_dev_steps)

        dev_data = NRSDataset(dev_features)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt == 'rnn':
        logger.info("Use rnn as neural belief tracker")
        from BeliefTrackerShare import BeliefTracker
    elif args.nbt == 'transformer':
        logger.info("Use transformer as neural belief tracker")
        from BeliefTrackerShareTFNRS import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    model = BeliefTracker(args, device)

    if args.pretrain is not None:
        logger.info(f'Loading pre-trained model from {args.pretrain}')
        pre_train = torch.load(args.pretrain)
        pre_train_state_dict = get_pretrain(model, pre_train)
        model.load_state_dict(pre_train_state_dict)

    model.to(device)

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

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info(optimizer)

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

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        last_loss_update = None
        eval_accuracy = CategoricalAccuracy()
        eval_loss = Average()
        best_loss = None
        best_acc = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
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
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)

                    if summary_writer is not None:
                        summary_writer.add_scalar("Epoch", epoch, global_step)
                        summary_writer.add_scalar("Train/Loss", loss.item(), global_step)
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

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

            if last_update is None or dev_acc > best_acc:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_acc = dev_acc

                logger.info(
                    "*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (last_update, dev_loss, best_acc))
            else:
                logger.info(
                    "*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (epoch, dev_loss, dev_acc))

            if last_loss_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model_loss.bin")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_loss_update = epoch
                best_loss = dev_loss

                logger.info(
                    "*** Lowest Loss Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (
                        last_loss_update, best_loss, dev_acc))
            else:
                logger.info(
                    "*** Lowest Loss Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (
                        epoch, dev_loss, dev_acc))

            if last_update + args.patience <= epoch:
                break


if __name__ == "__main__":
    main()
