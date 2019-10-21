from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from allennlp.training.metrics import CategoricalAccuracy, Average
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from util.general_util import read_json_config
from util.logger import setting_logger
from reader import from_params as reader_from_params
from model import from_params as model_from_params


def batch_to_device(batch, device):
    output = []
    for t in batch:
        output.append(t.to(device))
    # output.append(batch[-1])
    return output


def save(output_dir, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info('New model saved.')


def main(args):
    global logger
    logger = setting_logger(args.output_dir)
    logger.info('================== Program start. ========================')

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_predict:
        os.makedirs(args.predict_dir, exist_ok=True)

    reader = reader_from_params(args)

    num_train_steps = None
    if args.do_train:
        train_examples = reader.read(**args.read_train)
        num_train_steps = int(
            len(train_examples[0]) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    if args.pretrain:
        model_state_dict = torch.load(args.pretrain)
        args.model.update({'state_dict': model_state_dict})
    model = model_from_params(args)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

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

    eval_examples = reader.read(**args.read_eval)
    eval_tensors = reader.data2tensor(*eval_examples)
    eval_data = TensorDataset(*eval_tensors)
    eval_sampler = SequentialSampler(eval_data)
    eval_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    if args.do_train:
        logger.info('Start Training...')

        train_loss = Average()
        train_acc = CategoricalAccuracy()

        eval_loss = Average()
        eval_acc = CategoricalAccuracy()
        best_acc = 0

        summary_writer = SummaryWriter(log_dir=args.output_dir)
        global_step = 0

        train_tensors = reader.data2tensor(*train_examples)
        train_data = TensorDataset(*train_tensors)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        logger.info(f'****** Num train steps: {t_total}  ******')
        logger.info(f'****** Num train epochs: {args.num_train_epochs}   ******')
        logger.info(f'****** Train Batch Size: {args.train_batch_size}   ******')
        logger.info(f'****** Gradient accumulation steps: {args.gradient_accumulation_steps}   ******')

        for epoch in trange(int(args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(tqdm(train_data_loader, desc='Training')):
                if n_gpu == 1:
                    batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
                output = model.forward(*batch)
                loss = output['loss']
                scores = output['scores']
                train_acc(scores, batch[-1])

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        train_loss(loss.item())
                        if not args.fp16:
                            summary_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        summary_writer.add_scalar('train/loss', train_loss.get_metric(), global_step)
            summary_writer.add_scalar('train/accuracy', train_acc.get_metric(reset=True), epoch)

            model.eval()
            for eval_step, batch in enumerate(tqdm(eval_data_loader, desc='Evaluating')):
                if n_gpu == 1:
                    batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
                with torch.no_grad():
                    output = model(*batch)
                    loss, scores = output['loss'], output['scores']
                    if n_gpu > 1:
                        loss = loss.mean()
                    eval_loss(loss.item())
                eval_acc(scores, batch[-1])

            epoch_eval_loss = eval_loss.get_metric(reset=True)
            accuracy = eval_acc.get_metric(reset=True)
            summary_writer.add_scalar('eval/accuracy', accuracy, epoch)
            summary_writer.add_scalar('eval/loss', epoch_eval_loss, epoch)

            if best_acc < accuracy:
                best_acc = accuracy
                save(args.output_dir, model)
            logger.info('Epoch: %d, Loss: %f, Accuracy: %f (Best Accuracy: %f)' % (epoch, epoch_eval_loss, accuracy, best_acc))

        summary_writer.close()

    # Loading trained model.
    # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    # model_state_dict = torch.load(output_model_file, map_location='cuda:0')
    # model = initialize_model_by_name(state_dict=model_state_dict, **args.model)
    # model.to(device)
    #
    # if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_acc = CategoricalAccuracy()
    #     model.eval()
    #     format_output = dict()
    #     data_ids = reader.data_ids
    #     for eval_step, batch in enumerate(tqdm(eval_data_loader, desc='Predicting')):
    #         if n_gpu == 1:
    #             batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
    #         inputs = reader.batch2input(batch)
    #         with torch.no_grad():
    #             output = model(*inputs)
    #             loss, scores, predictions = output['loss'], output['scores'], output['predictions']
    #
    #         format_output.update(generate_format_predictions(batch[-1], predictions, eval_data_ids))
    #         eval_acc(scores, inputs['fluency'])
    #     with open(os.path.join(args.predict_dir, 'best_predictions.json'), 'w') as f:
    #         json.dump(format_output, f, indent=2)
    #     logger.info(f'Accuracy: {eval_acc.get_metric(reset=True)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file in Json format.', required=True)
    arg = parser.parse_args()
    config = read_json_config(arg.config)
    main(config)