default_config = {
    "seed": 42,
    "do_train": True,
    "no_cuda": False,
    "fp16": False,
    "gradient_accumulation_steps": 1,
    "adam_epsilon": 1e-6,
    "fp16_opt_level": "O1",
    "save_metric": "joint_acc"
}


def get_config(_config):
    default_config.update(_config)
    return default_config
