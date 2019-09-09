default_config = {
    "seed": 42,
    "do_train": True,
    "no_cuda": False,
    "fp16": False,
    "gradient_accumulation_steps": 1
}


def get_config(_config):
    default_config.update(_config)
    return default_config
