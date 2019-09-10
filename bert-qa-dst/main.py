import argparse
import json
import os
import random

import numpy as np
import torch
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from reader import from_params
from train.trainer import BertQADst
from util.config import get_config
from util.data_instance import State
from util.general_util import AverageMeter
from util.logger import setting_logger


def set_seed(_config):
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    if _config["n_gpu"] > 0:
        torch.cuda.manual_seed_all(_config["seed"])


def move(ls, device):
    return [x.to(device) for x in ls]


def train(_config, train_meta_data, train_data_loader, eval_meta_data, eval_data_loader, model):
    summary_writer = SummaryWriter(log_dir=_config["output_dir"])

    num_train_epochs = _config["num_train_epochs"]
    gradient_accumulation_steps = _config["gradient_accumulation_steps"]
    t_total = num_train_epochs * len(train_data_loader) // gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": _config["weight_decay"]
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=_config["learning_rate"], eps=_config["adam_epsilon"])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=_config["warmup_steps"], t_total=t_total)
    if _config["fp16"]:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=_config["fp16_opt_level"])

    if _config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # train_examples = train_meta_data["examples"]
    # Train
    logger.info("***** Running training *****")
    logger.info(f" Num input features: {len(train_data_loader)}")
    logger.info(f" Num epochs: {num_train_epochs}")
    logger.info(f" Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f" Total optimization steps: {t_total}")

    global_step = 0
    eval_cnt = 0
    train_loss = AverageMeter()
    best_metric = 0
    model.zero_grad()
    set_seed(_config)
    for epoch in range(int(num_train_epochs)):
        logger.info(f"=============== Running at epoch {epoch} ===============")
        for step, batch in enumerate(tqdm(train_data_loader, desc="Training", dynamic_ncols=True)):
            model.train()
            # batch = tuple(t.to(_config["device"]) for t in batch)
            batch_output = model(*batch)
            loss = batch_output["loss"]

            if _config["n_gpu"] > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if _config["fp16"]:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), _config["max_grad_norm"])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _config["max_grad_norm"])

            train_loss.update(loss.item(), n=1)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                summary_writer.add_scalar(f"train/loss", train_loss.avg, global_step)
                summary_writer.add_scalar(f"lr", scheduler.get_lr()[0], global_step)

                if global_step % _config["per_eval_step"] == 0 or \
                        (global_step + 1) == len(train_data_loader) // gradient_accumulation_steps:
                    eval_metric = evaluate(_config, model, eval_meta_data, eval_data_loader)
                    logger.info(f"Eval at {eval_cnt} times:")
                    for k, v in eval_metric:
                        summary_writer.add_scalar(f"eval/{k}", scalar_value=v, global_step=eval_cnt)
                        logger.info(f"{k}: {v}")

                    if eval_metric[_config["save_metric"]] > best_metric:
                        output_model_file = os.path.join(_config["output_dir"], f"saved_model.bin")
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("New model saved.")
                        best_metric = eval_metric[_config["save_metric"]]


def evaluate(_config, eval_meta_data, eval_data_loader, model):
    if not os.path.exists(_config["output_dir"]):
        os.makedirs(_config["output_dir"])

    logger.info("***** Running evaluation *****")
    logger.info(f" Num input features: {len(eval_data_loader)}")

    eval_loss = AverageMeter()

    for batch in tqdm(eval_data_loader, desc="evaluating", dynamic_ncols=True):
        model.eval()
        batch = tuple(t.to(_config["device"]) for t in batch)
        with torch.no_grad():
            model_output = model(*batch)
            if "loss" in model_output:
                eval_loss.update(model_output["loss"].item(), n=1)

    eval_metric = model.get_metric(reset=True)
    eval_metric["loss"] = eval_loss.avg
    return eval_metric


def main(_config):
    if os.path.exists(_config["output_dir"]) and os.listdir(_config["output_dir"]) and _config["do_train"]:
        raise ValueError(f"Output directory ({_config['output_dir']}) already exists and is not empty.")

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not _config["no_cuda"] else "cpu")
    _config["n_gpu"] = torch.cuda.device_count()
    _config["device"] = device

    global logger
    logger = setting_logger(_config["output_dir"])

    set_seed(_config)

    reader = from_params(_config)
    model = BertQADst(_config)
    model.to(device)

    if _config["do_train"]:
        _config["read_train"]["state"] = State.TRAIN
        _config["read_eval"]["state"] = State.VALIDATE
        train_meta_data, train_data_loader = reader.read(**_config["read_train"])
        eval_meta_data, eval_data_loader = reader.read(**_config["read_eval"])
        model.get_value_embedding(move(train_meta_data["value_input_ids"], device),
                                  move(train_meta_data["value_input_mask"], device),
                                  move(train_meta_data["value_token_type_ids"], device))
        train(_config, train_meta_data, train_data_loader, eval_meta_data, eval_data_loader, model)

    if _config["do_predict"]:
        _config["read_test"]["state"] = State.TEST
        test_meta_data, test_data_loader = reader.read(**_config["read_test"])
        test_metric = evaluate(_config, test_meta_data, test_data_loader, model)
        logger.info(json.dumps(test_metric, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = get_config(json.load(open(args.config, 'r')))

    main(config)
