import argparse
import collections
import csv
import json
import logging
import os
import random
import time
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
import numpy as np
from typing import Dict
import torch
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from torch.nn import Parameter
from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FullyShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed, note_best_checkpoint, initialize_optimizer, \
    load_and_cache_examples, if_cancel_sync

from data.default import Processor, convert_examples_to_features, get_label_embedding

logger: logging.Logger


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


def forward_step(model, inputs: Dict[str, torch.Tensor], cfg, scaler, return_outputs: bool = False):
    if cfg.fp16:
        # with torch.cuda.amp.autocast(dtype=(torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16)):
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs["loss"]

    if cfg.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
    if cfg.gradient_accumulation_steps > 1:
        loss = loss / cfg.gradient_accumulation_steps

    if cfg.fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if return_outputs:
        return loss.item(), outputs

    return loss.item()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    This script fix the undefined values in ontology. Please search "FIX_UNDEFINED" to find the difference with `main-multislot.py`

    For undefined values, we address `undefined` as the last possible value for each slot and remove its embedding.
    Therefore we can mask their label ids as the ignore_index while computing loss,
    so the undefined value will not have an influence on gradient and while computing
    accuracy, they will be computed as wrong for the model never 'seen' undefined value.
    """
    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--train_file", default=None, type=str)
    # parser.add_argument("--dev_file", default=None, type=str)
    # parser.add_argument("--test_file", default=None, type=str)
    # parser.add_argument("--ontology", default=None, type=str)
    # parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                          "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                          "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
    #                     type=str, required=False,
    #                     help="The directory of the pretrained BERT model")
    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
    #                          "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument('--predict_dir', default=None, type=str)
    # parser.add_argument("--target_slot",
    #                     default='all',
    #                     type=str,
    #                     required=True,
    #                     help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'")
    # parser.add_argument('--train_single', default='all', type=str)
    # parser.add_argument('--domain_list', default=None, type=str)
    # parser.add_argument("--tf_dir",
    #                     default='tensorboard',
    #                     type=str,
    #                     required=False,
    #                     help="Tensorboard directory")
    # parser.add_argument("--nbt",
    #                     default='rnn',
    #                     type=str,
    #                     required=True,
    #                     help="nbt type: rnn or transformer")
    # parser.add_argument("--fix_utterance_encoder",
    #                     action='store_true',
    #                     help="Do not train BERT utterance encoder")
    #
    # ## Other parameters
    # parser.add_argument("--max_seq_length",
    #                     default=64,
    #                     type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    # parser.add_argument("--max_slot_length", type=int, default=8)
    # parser.add_argument("--max_label_length",
    #                     default=18,
    #                     type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    # parser.add_argument("--max_turn_length",
    #                     default=22,
    #                     type=int,
    #                     help="The maximum total input turn length. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    # parser.add_argument('--hidden_dim',
    #                     type=int,
    #                     default=100,
    #                     help="hidden dimension used in belief tracker")
    # parser.add_argument('--num_rnn_layers',
    #                     type=int,
    #                     default=1,
    #                     help="number of RNN layers")
    # parser.add_argument('--zero_init_rnn',
    #                     action='store_true',
    #                     help="set initial hidden of rnns zero")
    # parser.add_argument('--attn_head',
    #                     type=int,
    #                     default=6,
    #                     help="the number of heads in multi-headed attention")
    # parser.add_argument("--do_train",
    #                     action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval",
    #                     action='store_true',
    #                     help="Whether to run eval on the test set.")
    # parser.add_argument("--do_lower_case",
    #                     action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--distance_metric",
    #                     type=str,
    #                     default="cosine",
    #                     help="The metric for distance between label embeddings: cosine, euclidean.")
    # parser.add_argument("--train_batch_size",
    #                     default=4,
    #                     type=int,
    #                     help="Total dialog batch size for training.")
    # parser.add_argument("--dev_batch_size",
    #                     default=1,
    #                     type=int,
    #                     help="Total dialog batch size for validation.")
    # parser.add_argument("--eval_batch_size",
    #                     default=16,
    #                     type=int,
    #                     help="Total dialog batch size for evaluation.")
    # parser.add_argument("--learning_rate",
    #                     default=5e-5,
    #                     type=float,
    #                     help="The initial learning rate for BertAdam.")
    # parser.add_argument("--num_train_epochs",
    #                     default=3.0,
    #                     type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument('--per_eval_steps', type=int, default=2000, help='Per steps for evaluation')
    # parser.add_argument("--patience",
    #                     default=10.0,
    #                     type=float,
    #                     help="The number of epochs to allow no further improvement.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    # parser.add_argument("--no_cuda",
    #                     action='store_true',
    #                     help="Whether not to use CUDA when available")
    # parser.add_argument("--local_rank",
    #                     type=int,
    #                     default=-1,
    #                     help="local_rank for distributed training on gpus")
    # parser.add_argument('--seed',
    #                     type=int,
    #                     default=42,
    #                     help="random seed for initialization")
    # parser.add_argument('--gradient_accumulation_steps',
    #                     type=int,
    #                     default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--fp16_opt_level', type=str, default='O1')
    # parser.add_argument('--loss_scale',
    #                     type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    # parser.add_argument('--max_loss_scale', type=float, default=None)
    # parser.add_argument("--do_not_use_tensorboard",
    #                     action='store_true',
    #                     help="Whether to run eval on the test set.")
    # parser.add_argument('--pretrain', type=str, default=None)
    # parser.add_argument('--model_id', type=int, default=1)
    # parser.add_argument('--use_query', default=False, action='store_true')
    # parser.add_argument('--fix_bert', default=False, action='store_true')
    # parser.add_argument('--reduce_layers', default=0, type=int)
    # parser.add_argument('--sa_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--sa_add_residual', default=False, action='store_true')
    # parser.add_argument('--ss_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--across_slot', default=False, action='store_true')
    # parser.add_argument('--override_attn', default=False, action='store_true')
    # parser.add_argument('--share_position_weight', default=False, action='store_true')
    # parser.add_argument('--slot_attention_type', default=-1, type=int)
    # parser.add_argument('--share_type', type=str, default='full_share', help="full_share, share_weight, copy_weight")
    # parser.add_argument('--key_type', type=int, default=0)
    # parser.add_argument('--self_attention_type', type=int, default=0)
    # parser.add_argument('--cls_type', default=0, type=int)
    # parser.add_argument('--mask_self', default=False, action='store_true')
    # parser.add_argument('--remove_some_slot', default=False, action='store_true')
    # parser.add_argument('--extra_dropout', type=float, default=-1.)
    # parser.add_argument('--reverse', default=False, action='store_true')
    # parser.add_argument('--weighted_cls', default=False, action='store_true')
    # parser.add_argument('--use_flow', default=False, action='store_true')
    # parser.add_argument('--flow_layer', default=100, type=int)
    # parser.add_argument('--flow_head', default=None, type=int)
    # parser.add_argument('--hie_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--hie_residual', default=False, action='store_true')
    # parser.add_argument('--hie_add_sup', default=0., type=float)
    # parser.add_argument('--max_grad_norm', default=1.0, type=float)
    # parser.add_argument('--hie_wd_residual', default=False, action='store_true')
    # parser.add_argument('--hie_wd_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--hie_wd_add_output', default=False, action='store_true')
    # parser.add_argument('--gate_type', default=0, type=int)
    # parser.add_argument('--cls_loss_weight', default=1., type=float)
    # parser.add_argument('--hidden_output', default=False, action='store_true')
    # parser.add_argument('--dropout', default=None, type=float)
    # parser.add_argument('--sa_no_position_embedding', default=False, action='store_true')
    #
    # parser.add_argument('--use_context', default=False, action='store_true')
    # parser.add_argument('--pre_turn', default=2, type=int)
    #
    # parser.add_argument('--use_pooling', default=False, action='store_true')
    # parser.add_argument('--pooling_head_num', default=1, type=int)
    # parser.add_argument('--use_mt', default=False, action='store_true')
    # parser.add_argument('--inter_domain', default=False, action='store_true')
    #
    # parser.add_argument('--extra_nbt', default=False, action='store_true')
    # parser.add_argument('--extra_nbt_attn_head', default=6, type=int)
    # parser.add_argument('--graph_add_sup', default=0., type=float)
    # parser.add_argument('--graph_value_sup', default=0., type=float)
    # parser.add_argument('--graph_attn_head', default=1, type=int)
    # parser.add_argument('--graph_add_output', default=False, action='store_true')
    # parser.add_argument('--graph_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--graph_add_residual', default=False, action='store_true')
    # parser.add_argument('--graph_hard_attn', default=False, action='store_true')
    # parser.add_argument('--use_rl', default=False, action='store_true')
    # parser.add_argument('--value_attn_head', default=1, type=int)
    # parser.add_argument('--value_attn_output', default=False, action='store_true')
    # parser.add_argument('--value_attn_layer_norm', default=False, action='store_true')
    # parser.add_argument('--value_attn_residual', default=False, action='store_true')
    #
    # parser.add_argument('--detach', default=False, action='store_true')
    #
    # parser.add_argument('--override_attn_extra', default=False, action='store_true')
    # parser.add_argument('--fusion_no_transform', default=False, action='store_true')
    # parser.add_argument('--fusion_act_fn', default='gelu', type=str)
    #
    # parser.add_argument('--context_agg', default=False, action='store_true')
    # parser.add_argument('--context_agg_fusion', default=False, action='store_true')
    # parser.add_argument('--fuse_type', default=0, type=int)
    # parser.add_argument('--diag_attn_hidden_scale', default=2.0, type=float)
    # parser.add_argument('--diag_attn_act', default=None, type=str)
    # parser.add_argument('--diag_attn_act_fn', default='relu', type=str)
    #
    # parser.add_argument('--context_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--context_add_residual', default=False, action='store_true')
    # parser.add_argument('--ff_hidden_size', type=int, default=1536)
    # parser.add_argument('--ff_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--ff_add_residual', default=False, action='store_true')
    # parser.add_argument('--query_layer_norm', default=False, action='store_true')
    # parser.add_argument('--query_residual', default=False, action='store_true')
    # parser.add_argument('--context_override_attn', default=False, action='store_true')
    #
    # parser.add_argument('--multi_view_diag_attn_hidden_scale', default=1.0, type=float)
    #
    # parser.add_argument('--value_embedding_type', default='cls', type=str)
    # parser.add_argument('--sa_fuse_act_fn', default='gelu', type=str)
    # parser.add_argument('--transfer_sup', default=0, type=float)
    # parser.add_argument('--save_gate', default=False, action='store_true')
    # parser.add_argument('--slot_res', default=None, type=str)
    # parser.add_argument('--key_add_value', default=False, action='store_true')
    # parser.add_argument('--key_add_value_pro', default=False, action='store_true')
    # parser.add_argument('--add_relu', default=False, action='store_true')
    # parser.add_argument('--add_weight', default=False, action='store_true')
    # parser.add_argument('--num_layers', default=0, type=int)
    # parser.add_argument('--hop_update_self', default=False, action='store_true')
    # parser.add_argument('--graph_attn_type', default=0, type=int)
    # parser.add_argument('--graph_dropout', default=0.1, type=float)
    # parser.add_argument('--sa_act_1', default=None, type=str)
    #
    # parser.add_argument('--sa_fuse_type', default='gate', type=str)
    # parser.add_argument('--fuse_add_layer_norm', default=False, action='store_true')
    # parser.add_argument('--pre_cls_sup', default=1.0, type=float)
    #
    # parser.add_argument('--mask_top_k', type=int, default=0)
    # parser.add_argument('--test_mode', default=-1, type=int)
    #
    # parser.add_argument('--remove_unrelated', default=False, action='store_true')
    #
    # args = parser.parse_args()

    # check output_dir

    # CUDA setup
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        dist.init_process_group(backend='nccl')
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    # Tensorboard logging
    if cfg.local_rank in [-1, 0]:
        _dir_splits = cfg.output_dir.split('/')
        _log_dir = '/'.join([_dir_splits[0], 'runs'] + _dir_splits[1:])
        tb_writer = SummaryWriter(log_dir=_log_dir)
        # tb_helper = hydra.utils.instantiate(cfg.summary_helper,
        #                                     writer=tb_writer) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_writer = None
        # tb_helper = None

    if os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir) and cfg.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(cfg.output_dir))
    if cfg.do_train:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Set random seed
    set_seed(cfg)

    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(cfg)
    label_list = processor.get_labels()
    # num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type
    """ 
    FIX_UNDEFINED: Reduce 1 for 'undefined' label to avoid that the shape of initialized embedding is not compatible 
    with the shape of that defined in __init__() method of model.
    """
    num_labels = [len(labels) - 1 for labels in label_list]  # number of slot-values in each slot-type

    #############################################################################
    # Load model
    #############################################################################
    if cfg.local_rank not in [-1, 0]:
        dist.barrier()

    if cfg.pretrain:
        pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
    else:
        pretrain_state_dict = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict, num_labels_ls=num_labels)

    if cfg.local_rank == 0:
        dist.barrier()

    # if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
    model.to(cfg.device)

    num_train_steps = None
    accumulation = False
    t_total = -1
    num_warmup_steps = -1
    if cfg.do_train:
        train_examples = processor.get_train_examples(cfg.data_dir, accumulation=accumulation, train_file=cfg.train_file)
        dev_examples = processor.get_dev_examples(cfg.data_dir, accumulation=accumulation, dev_file=cfg.dev_file)

        cfg.train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
        train_data = convert_examples_to_features(train_examples, label_list, cfg.max_seq_length, tokenizer, cfg.max_turn_length)

        if cfg.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cfg.train_batch_size,
                                      pin_memory=True, num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor)

        cfg.dev_batch_size = cfg.per_gpu_dev_batch_size * max(1, cfg.n_gpu)
        dev_data = convert_examples_to_features(dev_examples, label_list, cfg.max_seq_length, tokenizer, cfg.max_turn_length)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=cfg.dev_batch_size,
                                    pin_memory=True, num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor)

        logger.info("Loaded data!")

        # Initialize slot and value embeddings before distributed initialization
        label_emb_inputs = []
        for labels in label_list:
            label_emb_inputs.append(get_label_embedding(labels, tokenizer, device))
            # label_token_ids.append(token_ids)
            # label_len.append(lens)

        # Get domain-slot-type embeddings
        slot_emb_inputs = get_label_embedding(processor.target_slot, tokenizer, device)

        # Initialize slot and value embeddings
        model.initialize_slot_value_lookup(label_emb_inputs, slot_emb_inputs)

        if cfg.local_rank != -1:
            model.to(torch.device("cpu"))  # For FullyShardedDDP, place the model on cpu first.
            dist.barrier()

        if cfg.max_steps > 0:
            t_total = cfg.max_steps
            cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

        num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

        optimizer = scheduler = None
        # Prepare optimizer and schedule (linear warmup and decay)
        if cfg.local_rank == -1:
            optimizer = initialize_optimizer(cfg, model=model)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

        if cfg.fp16:
            if cfg.local_rank != -1:
                scaler = ShardedGradScaler()
            else:
                from torch.cuda.amp.grad_scaler import GradScaler

                scaler = GradScaler()
        else:
            scaler = None

        # Distributed training (should be after apex fp16 initialization)
        if cfg.local_rank != -1:
            model = hydra.utils.instantiate(cfg.fairscale_config, model=model, device=cfg.device)
            optimizer = initialize_optimizer(cfg, model=model)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

        logger.info(optimizer)

        ###############################################################################
        # Training code
        ###############################################################################

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", cfg.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    cfg.train_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if cfg.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Warmup steps = %d", num_warmup_steps)

        logger.info("Training...")

        global_step = 0
        last_update = None
        last_loss_update = None
        best_loss = None
        best_acc = 0
        train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
        set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)
        model.zero_grad()

        tr_loss = 0.0
        logging_loss = 0.0

        for epoch in train_iterator:
            # Train
            nb_tr_examples = 0
            nb_tr_steps = 0

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
            if cfg.local_rank != -1:
                train_dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(epoch_iterator):
                model.train()

                batch = batch_to_device(batch, device)
                # input_ids, token_type_ids, input_mask, answer_type_ids, label_ids = batch

                # Forward
                # with torch.autograd.set_detect_anomaly(True):
                if if_cancel_sync(cfg, step):
                    with model.no_sync():
                        loss, outputs = forward_step(model, batch, cfg, scaler, return_outputs=True)
                else:
                    loss, outputs = forward_step(model, batch, cfg, scaler, return_outputs=True)

                # if n_gpu == 1:
                #     loss, loss_slot, acc, _, acc_slot, _, _ = \
                #         model(input_ids, token_type_ids, input_mask, answer_type_ids, label_ids, n_gpu)
                # else:
                #     loss, _, acc, _, acc_slot, _, _ = \
                #         model(input_ids, token_type_ids, input_mask, answer_type_ids, label_ids, n_gpu)
                #
                #     # average to multi-gpus
                #     loss = loss.mean()
                #     acc = acc.mean()
                #     acc_slot = acc_slot.mean(0)

                loss_slot, acc, acc_slot = outputs[1], outputs[2], outputs[4]

                tr_loss += loss
                nb_tr_examples += batch["input_ids"].size(0)
                nb_tr_steps += 1
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.fp16:
                        scaler.unscale_(optimizer)

                    if cfg.max_grad_norm and not ("optimizer" in cfg and cfg.optimizer and "lamb" in cfg.optimizer):
                        if hasattr(optimizer, "clip_grad_norm"):
                            optimizer.clip_grad_norm(cfg.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            model.clip_grad_norm_(cfg.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                    if cfg.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad(set_to_none=True)
                    global_step += 1

                    if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                        tb_writer.add_scalar("Epoch", epoch, global_step)
                        tb_writer.add_scalar("Train/Loss", (tr_loss - logging_loss) / cfg.logging_steps, global_step, global_step)
                        tb_writer.add_scalar("Train/JointAcc", acc, global_step)
                        tb_writer.add_scalar("Train/LearningRate", scheduler.get_lr()[0], global_step)
                        logging_loss = tr_loss

                        # if n_gpu == 1:
                        #     for i, slot in enumerate(processor.target_slot):
                        #         summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i],
                        #                                   global_step)
                        #         summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i],
                        #                                   global_step)
                        #     if hasattr(model, "get_metric"):
                        #         metric = model.get_metric(reset=False)
                        #         for k, v in metric.items():
                        #             summary_writer.add_scalar(f"Train/{k}", v, global_step)

                    if cfg.evaluate_during_training and global_step % cfg.eval_steps == 0:
                        state_dict = model.state_dict()

                        if cfg.local_rank not in [-1, 0]:
                            dist.barrier()
                            continue

                        # Perform evaluation on validation dataset
                        model.eval()
                        dev_loss = 0
                        dev_acc = 0
                        dev_type_acc = 0
                        dev_loss_slot, dev_acc_slot, dev_acc_slot_type = None, None, None
                        nb_dev_examples, nb_dev_steps = 0, 0

                        for _, eval_batch in enumerate(tqdm(dev_dataloader, desc="Validation", dynamic_ncols=True)):
                            eval_batch = batch_to_device(eval_batch, device)
                            # input_ids, token_type_ids, input_mask, answer_type_ids, label_ids = eval_batch
                            # batch_size = input_ids.size(0)
                            # if input_ids.dim() == 2:
                            #     input_ids = input_ids.unsqueeze(0)
                            #     token_type_ids = token_type_ids.unsqueeze(0)
                            #     input_mask = input_mask.unsqueeze(0)
                            #     answer_type_ids = answer_type_ids.unsqueeze(0)
                            #     label_ids = label_ids.unsuqeeze(0)
                            batch_size = eval_batch["input_ids"].size(0)
                            answer_type_ids = eval_batch["answer_type_ids"]

                            with torch.no_grad():
                                with torch.cuda.amp.autocast():
                                    outputs = model(**eval_batch)
                            loss, loss_slot, acc, type_acc, acc_slot, type_acc_slot, _ = outputs

                            num_valid_turn = torch.sum(answer_type_ids[:, :, 0].view(-1) > -1,
                                                       0).item()  # valid turns for all current batch
                            # dev_loss += loss.item() * num_valid_turn
                            dev_acc += acc * num_valid_turn
                            dev_loss += loss.item() * batch_size
                            dev_type_acc += type_acc * num_valid_turn
                            # dev_acc += acc.item()

                            if cfg.n_gpu == 1:
                                if dev_loss_slot is None:
                                    # dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                                    dev_acc_slot = acc_slot * num_valid_turn
                                    dev_acc_slot_type = type_acc_slot * num_valid_turn
                                    dev_loss_slot = [l * batch_size for l in loss_slot]
                                    # dev_acc_slot = acc_slot
                                else:
                                    for i, l in enumerate(loss_slot):
                                        # dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                                        dev_loss_slot[i] = dev_loss_slot[i] + l * batch_size
                                    dev_acc_slot += acc_slot * num_valid_turn
                                    dev_acc_slot_type += type_acc_slot * num_valid_turn
                                    # dev_acc_slot += acc_slot

                            nb_dev_examples += num_valid_turn

                        # dev_loss = dev_loss / nb_dev_examples
                        dev_loss = dev_loss / len(dev_data)
                        dev_acc = dev_acc / nb_dev_examples
                        dev_type_acc = dev_type_acc / nb_dev_examples

                        if cfg.n_gpu == 1:
                            dev_acc_slot = dev_acc_slot / nb_dev_examples
                            dev_acc_slot_type = dev_acc_slot_type / nb_dev_examples

                        # tensorboard logging
                        if tb_writer is not None:
                            tb_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                            tb_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                            tb_writer.add_scalar("Validate/Cls_Acc", dev_type_acc, global_step)
                            if cfg.n_gpu == 1:
                                for i, slot in enumerate(processor.target_slot):
                                    tb_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'),
                                                         dev_loss_slot[i] / len(dev_data), global_step)
                                    tb_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i],
                                                         global_step)
                                    tb_writer.add_scalar("Validate/Cls_Acc_%s" % slot.replace(' ', '_'), dev_acc_slot_type[i],
                                                         global_step)
                                if hasattr(model, "get_metric"):
                                    metric = model.get_metric(reset=True)
                                    for k, v in metric.items():
                                        tb_writer.add_scalar(f"Validate/{k}", v, global_step)

                        dev_loss = round(dev_loss, 6)
                        if last_update is None or dev_acc > best_acc:
                            if cfg.local_rank == 0:
                                unwrap_model(model).save_pretrained(cfg.output_dir, state_dict=state_dict)
                            else:
                                model.save_pretrained(cfg.output_dir)

                            tokenizer.save_pretrained(cfg.output_dir)
                            OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

                            last_update = global_step
                            best_acc = dev_acc

                            logger.info(
                                "Model Updated: Global Step=%d, Validation Loss=%.6f, Validation Acc=%.6f" % (
                                    global_step, dev_loss, best_acc))
                        else:
                            logger.info(
                                "Model NOT Updated: Global Step=%d, Validation Loss=%.6f, Validation Acc=%.6f" % (
                                    global_step, dev_loss, dev_acc))

                        if last_loss_update is None or dev_loss < best_loss:
                            # Save a trained model
                            output_model_dir = os.path.join(cfg.output_dir, "best_loss")
                            if not os.path.exists(output_model_dir):
                                os.makedirs(output_model_dir)

                            if cfg.local_rank == 0:
                                unwrap_model(model).save_pretrained(output_model_dir, state_dict=state_dict)
                            else:
                                model.save_pretrained(output_model_dir)

                            tokenizer.save_pretrained(output_model_dir)
                            OmegaConf.save(cfg, os.path.join(output_model_dir, "training_config.yaml"))

                            last_loss_update = global_step
                            best_loss = dev_loss

                            logger.info(
                                "Lowest Loss Model Updated: Global Step=%d, Validation Loss=%.6f, Validation Acc=%.6f" % (
                                    global_step, best_loss, dev_acc))
                        else:
                            logger.info(
                                "Lowest Loss Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f" % (
                                    global_step, dev_loss, dev_acc))

                        if cfg.local_rank == 0:
                            dist.barrier()

                        if last_update and last_update + cfg.patience * cfg.eval_steps <= global_step:
                            break

            if last_update and last_update + cfg.patience * cfg.eval_steps <= global_step:
                break

    ###############################################################################
    # Evaluation
    ###############################################################################
    # Load a trained model that you have fine-tuned
    predict_dir = cfg.predict_dir if cfg.predict_dir is not None else cfg.output_dir
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir, exist_ok=True)
    for state_name in [cfg.output_dir, os.path.join(cfg.output_dir, "best_loss")]:
        # for state_name in ['pytorch_model.bin']:
        if not os.path.exists(state_name):
            continue
        # model = BeliefTracker(args, num_labels, device)
        # model = hydra.utils.call(cfg.model, state_name)
        logger.info(f'Loading saved model from {state_name}')
        # output_model_file = os.path.join(args.output_dir, state_name)

        # in the case that slot and values are different between the training and evaluation
        # ptr_model = torch.load(output_model_file)
        # Initialize slot value look up to avoid mismatch
        # for k in list(ptr_model.keys()):
        #     if 'slot_lookup' in k or 'value_lookup' in k:
        #         ptr_model.pop(k)

        state_dict = torch.load(os.path.join(state_name, "pytorch_model.bin"), map_location="cpu")
        # for k in state_dict.keys():
        #     if 'slot_lookup' in k or 'value_lookup' in k:
        #         state_dict.pop(k)
        model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=state_dict, num_labels=num_labels)

        # if n_gpu == 1:
        #     state = model.state_dict()
        #     state.update(ptr_model)
        #     state = get_pretrain(model, state)
        #     model.load_state_dict(state)
        # else:
        #     print("Evaluate using only one device!")
        #     model.module.load_state_dict(ptr_model)

        model.to(device)

        # Get slot-value embeddings
        label_emb_inputs = []
        for labels in label_list:
            label_emb_inputs.append(get_label_embedding(labels, tokenizer, device))
            # label_token_ids.append(token_ids)
            # label_len.append(lens)

        # Get domain-slot-type embeddings
        slot_emb_inputs = get_label_embedding(processor.target_slot, tokenizer, device)

        # Initialize slot and value embeddings
        model.initialize_slot_value_lookup(label_emb_inputs, slot_emb_inputs)

        # Evaluation
        if cfg.do_eval and (cfg.local_rank == -1 or dist.get_rank() == 0):

            eval_examples = processor.get_test_examples(cfg.data_dir, accumulation=accumulation, test_file=cfg.test_file)
            # all_input_ids, all_input_len, all_answer_type_ids, all_label_ids = convert_examples_to_features(
            #     eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
            # all_token_type_ids, all_input_mask = make_aux_tensors(all_input_ids, all_input_len)
            eval_data = convert_examples_to_features(eval_examples, label_list, cfg.max_seq_length, tokenizer, cfg.max_turn_length)
            # all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device),
            # all_label_ids.to(device)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", cfg.eval_batch_size)

            # eval_data = TensorDataset(all_input_ids, all_token_type_ids, all_input_mask, all_answer_type_ids,
            #                           all_label_ids)
            cfg.eval_batch_size = cfg.per_gpu_dev_batch_size * max(1, cfg.n_gpu)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=cfg.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            eval_loss_slot, eval_acc_slot = None, None
            nb_eval_steps, nb_eval_examples = 0, 0

            accuracies = {'joint5': 0, 'joint_type5': 0, 'slot5': 0, 'slot_type5': 0, 'num_slot5': 0, 'num_turn': 0,
                          'joint_rest': 0, 'joint_type_rest': 0, 'slot_rest': 0, 'slot_type_rest': 0,
                          'num_slot_rest': 0,
                          'joint_taxi': 0, 'joint_type_taxi': 0, 'slot_taxi': 0, 'slot_type_taxi': 0,
                          'num_slot_taxi': 0,
                          'joint_hotel': 0, 'joint_type_hotel': 0, 'slot_hotel': 0, 'slot_type_hotel': 0,
                          'num_slot_hotel': 0,
                          'joint_attraction': 0, 'joint_type_attraction': 0, 'slot_attraction': 0,
                          'slot_type_attraction': 0,
                          'num_slot_attraction': 0,
                          'joint_train': 0, 'joint_type_train': 0, 'slot_train': 0, 'slot_type_train': 0,
                          'num_slot_train': 0}
            predictions = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # input_ids = input_ids.to(device)
                # token_type_ids = token_type_ids.to(device)
                # input_mask = input_mask.to(device)
                # answer_type_ids = answer_type_ids.to(device)
                # label_ids = label_ids.to(device)
                # if input_ids.dim() == 2:
                #     input_ids = input_ids.unsqueeze(0)
                #     token_type_ids = token_type_ids.unsqueeze(0)
                #     input_mask = input_mask.unsqueeze(0)
                #     answer_type_ids = answer_type_ids.unsqueeze(0)
                #     label_ids = label_ids.unsuqeeze(0)
                batch = batch_to_device(batch, device)
                with torch.no_grad():
                    # if cfg.n_gpu == 1:
                    #     loss, loss_slot, acc, type_acc, acc_slot, type_acc_slot, pred_slot \
                    #         = model(input_ids, token_type_ids, input_mask, answer_type_ids, label_ids, cfg.n_gpu)
                    # else:
                    #     loss, _, acc, type_acc, acc_slot, type_acc_slot, pred_slot \
                    #         = model(input_ids, token_type_ids, input_mask, answer_type_ids, label_ids, cfg.n_gpu)
                    #     nbatch = label_ids.size(0)
                    #     nslot = pred_slot.size(3)
                    #     pred_slot = pred_slot.view(nbatch, -1, nslot)
                    with torch.cuda.amp.autocast():
                        loss, loss_slot, acc, type_acc, acc_slot, type_acc_slot, pred_slot = model(**batch)
                answer_type_ids = batch["answer_type_ids"]
                label_ids = batch["labels"]

                accuracies = eval_all_accs(pred_slot, answer_type_ids, label_ids, accuracies)
                predictions.extend(get_predictions(pred_slot, answer_type_ids, label_ids, processor,
                                                   gate=model.get_gate_metric(reset=True) if cfg.save_gate else None,
                                                   value_scores=model.get_value_scores(reset=True) if cfg.save_gate else None,
                                                   graph_scores=model.get_graph_scores(reset=True) if cfg.save_gate else None))

                nb_eval_ex = (answer_type_ids[:, :, 0].view(-1) != -1).sum().item()
                nb_eval_examples += nb_eval_ex
                nb_eval_steps += 1

                if cfg.n_gpu == 1:
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
            if cfg.n_gpu == 1:
                eval_acc_slot = eval_acc_slot / nb_eval_examples

            # loss = tr_loss / nb_tr_steps if cfg.do_train else None

            if cfg.n_gpu == 1:
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          # 'loss': loss,
                          'eval_loss_slot': '\t'.join([str(val / nb_eval_examples) for val in eval_loss_slot]),
                          'eval_acc_slot': '\t'.join([str((val).item()) for val in eval_acc_slot])
                          }
            else:
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          # 'loss': loss
                          }

            out_file_name = f'eval_results_{state_name}'
            if cfg.target_slot == 'all':
                out_file_name += '_all'
            output_eval_file = os.path.join(predict_dir, "%s.txt" % out_file_name)

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            with open(os.path.join(predict_dir, f"predictions_{state_name}.json"), 'w') as f:
                json.dump(predictions, f, indent=2)

            if hasattr(model, "get_metric"):
                with open(os.path.join(predict_dir, f"eval_metric_{state_name}.json"), 'w') as f:
                    json.dump(model.get_metric(reset=False), f, indent=2)

            out_file_name = f'eval_all_accuracies_{state_name}'
            with open(os.path.join(predict_dir, "%s.txt" % out_file_name), 'w') as f:
                f.write(
                    'joint acc (5 domain) : %.5f \t slot acc (5 domain) : %.5f \n'
                    'joint acc type (5 domain) : %.5f \t slot acc type (5 domain) : %.5f \n'

                    'joint restaurant : %.5f \t slot acc restaurant : %.5f \n'
                    'joint restaurant type : %.5f \t slot acc restaurant type : %.5f \n'

                    'joint taxi : %.5f \t slot acc taxi : %.5f \n'
                    'joint taxi type : %.5f \t slot acc taxi type : %.5f \n'

                    'joint hotel : %.5f \t slot acc hotel : %.5f \n'
                    'joint hotel type : %.5f \t slot acc hotel type : %.5f \n'

                    'joint attraction : %.5f \t slot acc attraction : %.5f \n'
                    'joint attraction type : %.5f \t slot acc attraction type : %.5f \n'

                    'joint train : %.5f \t slot acc train %.5f \n'
                    'joint train type : %.5f \t slot acc train type %.5f \n' % (
                        (accuracies['joint5'] / accuracies['num_turn']).item(),
                        (accuracies['slot5'] / accuracies['num_slot5']).item(),
                        (accuracies['joint_type5'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type5'] / accuracies['num_slot5']).item(),

                        (accuracies['joint_rest'] / accuracies['num_turn']).item(),
                        (accuracies['slot_rest'] / accuracies['num_slot_rest']).item(),
                        (accuracies['joint_type_rest'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type_rest'] / accuracies['num_slot_rest']).item(),

                        (accuracies['joint_taxi'] / accuracies['num_turn']).item(),
                        (accuracies['slot_taxi'] / accuracies['num_slot_taxi']).item(),
                        (accuracies['joint_type_taxi'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type_taxi'] / accuracies['num_slot_taxi']).item(),

                        (accuracies['joint_hotel'] / accuracies['num_turn']).item(),
                        (accuracies['slot_hotel'] / accuracies['num_slot_hotel']).item(),
                        (accuracies['joint_type_hotel'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type_hotel'] / accuracies['num_slot_hotel']).item(),

                        (accuracies['joint_attraction'] / accuracies['num_turn']).item(),
                        (accuracies['slot_attraction'] / accuracies['num_slot_attraction']).item(),
                        (accuracies['joint_type_attraction'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type_attraction'] / accuracies['num_slot_attraction']).item(),

                        (accuracies['joint_train'] / accuracies['num_turn']).item(),
                        (accuracies['slot_train'] / accuracies['num_slot_train']).item(),
                        (accuracies['joint_type_train'] / accuracies['num_turn']).item(),
                        (accuracies['slot_type_train'] / accuracies['num_slot_train']).item()
                    ))


def eval_all_accs(pred_slot, answer_type_ids, labels, accuracies):
    answer_type_pred = pred_slot[:, :, :, 0]
    pred_slot = pred_slot[:, :, :, 1]

    def _eval_acc(_answer_type_pred, _pred_slot, _answer_type_ids, _labels):
        slot_dim = _labels.size(-1)
        classify_mask = ((_answer_type_ids != -1) * (_answer_type_ids != 2)).view(-1, slot_dim)
        value_accuracy = (_pred_slot == _labels).view(-1, slot_dim).masked_fill(classify_mask, 1)
        answer_type_accuracy = (_answer_type_pred == _answer_type_ids).view(-1, slot_dim)
        accuracy = value_accuracy * answer_type_accuracy
        # accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_answer_type_ids[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_answer_type_ids > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        joint_acc_type = sum(torch.sum(answer_type_accuracy, dim=1) // slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        slot_acc_type = torch.sum(answer_type_accuracy).float()
        return joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data

    if labels.size(-1) == 30:  # Full slots
        # restaurant domain
        joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
            answer_type_pred[:, :, 13:20], pred_slot[:, :, 13:20], answer_type_ids[:, :, 13:20], labels[:, :, 13:20])
        accuracies['joint_rest'] += joint_acc
        accuracies['joint_type_rest'] += joint_acc_type
        accuracies['slot_rest'] += slot_acc
        accuracies['slot_type_rest'] += slot_acc_type
        accuracies['num_slot_rest'] += num_data

        # taxi domain
        joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
            answer_type_pred[:, :, 20:24], pred_slot[:, :, 20:24], answer_type_ids[:, :, 20:24], labels[:, :, 20:24])
        accuracies['joint_taxi'] += joint_acc
        accuracies['joint_type_taxi'] += joint_acc_type
        accuracies['slot_taxi'] += slot_acc
        accuracies['slot_type_taxi'] += slot_acc_type
        accuracies['num_slot_taxi'] += num_data

        # attraction
        joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
            answer_type_pred[:, :, 0:3], pred_slot[:, :, 0:3], answer_type_ids[:, :, 0:3], labels[:, :, 0:3])
        accuracies['joint_attraction'] += joint_acc
        accuracies['joint_type_attraction'] += joint_acc_type
        accuracies['slot_attraction'] += slot_acc
        accuracies['slot_type_attraction'] += slot_acc_type
        accuracies['num_slot_attraction'] += num_data

        # hotel
        joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
            answer_type_pred[:, :, 3:13], pred_slot[:, :, 3:13], answer_type_ids[:, :, 3:13], labels[:, :, 3:13])
        accuracies['joint_hotel'] += joint_acc
        accuracies['joint_type_hotel'] += joint_acc_type
        accuracies['slot_hotel'] += slot_acc
        accuracies['slot_type_hotel'] += slot_acc_type
        accuracies['num_slot_hotel'] += num_data

        # train
        joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
            answer_type_pred[:, :, 24:], pred_slot[:, :, 24:], answer_type_ids[:, :, 24:], labels[:, :, 24:])
        accuracies['joint_train'] += joint_acc
        accuracies['joint_type_train'] += joint_acc_type
        accuracies['slot_train'] += slot_acc
        accuracies['slot_type_train'] += slot_acc_type
        accuracies['num_slot_train'] += num_data
    else:
        accuracies['num_slot_rest'] = torch.tensor(1)
        accuracies['num_slot_taxi'] = torch.tensor(1)
        accuracies['num_slot_attraction'] = torch.tensor(1)
        accuracies['num_slot_hotel'] = torch.tensor(1)
        accuracies['num_slot_train'] = torch.tensor(1)

    # 5 domains (excluding bus and hotel domain)
    joint_acc, joint_acc_type, slot_acc, slot_acc_type, num_turn, num_data = _eval_acc(
        answer_type_pred, pred_slot, answer_type_ids, labels)
    accuracies['num_turn'] += num_turn
    accuracies['joint5'] += joint_acc
    accuracies['joint_type5'] += joint_acc_type
    accuracies['slot5'] += slot_acc
    accuracies['slot_type5'] += slot_acc_type
    accuracies['num_slot5'] += num_data

    return accuracies


def get_predictions(pred_slots, answer_type_ids, labels, processor: Processor, gate=None,
                    value_scores=None, graph_scores=None):
    """
    :param pred_slots:
    :param answer_type_ids:
    :param labels:
    :param processor:
    :param gate: [slot_dim, ds, ts - 1, 1]
    :return:
    """
    answer_type_pred = pred_slots[:, :, :, 0]
    pred_slots = pred_slots[:, :, :, 1]
    type_vocab = ['none', 'do not care', 'value']
    predictions = []
    ds = pred_slots.size(0)
    ts = pred_slots.size(1)
    slot_dim = pred_slots.size(-1)
    if gate is not None:
        assert gate.size() == (slot_dim, ds, ts - 1)
    if value_scores is not None:
        # assert value_scores.size()[:-1] == (slot_dim, ds, ts)
        value_prob, value_idx = value_scores
        assert value_prob.size() == (slot_dim, ds, ts)
    if graph_scores is not None:
        assert graph_scores.size() == (ds, ts - 1, slot_dim, slot_dim)
        # graph_scores = graph_scores.reshape(ds, ts - 1, slot_dim, slot_dim)
    for i in range(ds):
        dialog_pred = []
        for j in range(ts):
            if answer_type_ids[i, j, 0] == -1:
                break
            slot_pred = {}
            for slot_idx in range(slot_dim):
                slot = processor.target_slot[slot_idx]
                pred_answer_type = type_vocab[answer_type_pred[i, j, slot_idx]]
                gold_answer_type = type_vocab[answer_type_ids[i, j, slot_idx]]
                pred_label = -1 if pred_slots[i, j, slot_idx] == -1 else processor.ontology[slot][
                    pred_slots[i][j][slot_idx]]
                gold_label = -1 if labels[i, j, slot_idx] == -1 else processor.ontology[slot][labels[i][j][slot_idx]]
                # dialog_pred.append({
                #     "turn": j,
                #     "slot": slot,
                #     "predict_value": pred_label,
                #     "gold_label": gold_label,
                #     "pred_answer_type": pred_answer_type,
                #     "gold_answer_type": gold_answer_type
                # })
                slot_pred[slot] = {
                    "predict_value": pred_label,
                    "gold_label": gold_label,
                    "predict_answer_type": pred_answer_type,
                    "gold_answer_type": gold_answer_type
                }

                if gate is not None:
                    if j == 0:
                        slot_pred[slot]['gate'] = 0
                    else:
                        slot_pred[slot]['gate'] = gate[slot_idx, i, j - 1].item()
                if value_scores is not None:
                    slot_pred[slot]['value_prob'] = value_prob[slot_idx, i, j].item()
                    slot_pred[slot]['value_str'] = processor.ontology[slot][value_idx[slot_idx, i, j]]
                if graph_scores is not None:
                    if j == 0:
                        slot_pred[slot]['graph_scores'] = [0] * slot_dim
                    else:
                        slot_pred[slot]['graph_scores'] = graph_scores[i, j - 1, slot_idx].tolist()
            dialog_pred.append(slot_pred)
        predictions.append(dialog_pred)
    return predictions


if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()
