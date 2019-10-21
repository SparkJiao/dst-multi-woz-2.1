import json
from typing import Dict


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def save(self):
        return {
            'val': self.val,
            'avg': self.avg,
            'sum': self.sum,
            'count': self.count
        }

    def load(self, value: dict):
        if value is None:
            self.reset()
        self.val = value['val'] if 'val' in value else 0
        self.avg = value['avg'] if 'avg' in value else 0
        self.sum = value['sum'] if 'sum' in value else 0
        self.count = value['count'] if 'count' in value else 0


class Config(object):
    def __init__(self, opt: Dict = None):
        self.train_batch_size = 32
        self.predict_batch_size = 32
        self.learning_rate = 5e-5
        self.num_train_epochs = 2.0
        self.warmup_proportion = 0.1
        self.no_cuda = False
        self.seed = 42
        self.gradient_accumulation_steps = 1
        self.do_lower_case = True
        self.local_rank = -1
        self.fp16 = False
        self.loss_scale = 0
        self.do_train = False
        self.do_predict = False
        self.eval_step = 1000000

        self.__dict__.update(opt)


def read_json_config(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return Config(config)
