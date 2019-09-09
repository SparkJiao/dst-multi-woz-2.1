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