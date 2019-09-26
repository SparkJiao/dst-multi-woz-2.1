import argparse

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

_parser = argparse.ArgumentParser()

_parser.add_argument('--task', type=str, default='dst', help='dst, pre-train')

args = _parser.parse_args()
