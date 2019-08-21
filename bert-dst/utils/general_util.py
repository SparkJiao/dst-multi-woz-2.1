import csv


def read_tsv(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
    lines = []
    for line in reader:
        if len(line) > 0 and line[0][0] == '#':  # Remove comment
            lines.append(line)
    return lines


def is_whitespace(ch):
    if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
        return True
    return False


def get_char_to_tokens_map(sentence: str):
    char_to_tok_map = []
    tokens = []
    prev_is_whitespace = True
    for ch in sentence:
        if is_whitespace(ch):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(ch)
            else:
                tokens[-1] += ch
            prev_is_whitespace = False
        char_to_tok_map.append(len(tokens) - 1)
    return tokens, char_to_tok_map


def truncate_seq_pair(tokens_a, tokens_b, max_length):
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


class AverageMeter(object):
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

