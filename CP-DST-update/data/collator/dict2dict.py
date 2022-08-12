from typing import Dict

from torch import Tensor
from torch.utils.data import Dataset


class DictTensorDataset(Dataset):
    def __init__(self, data: Dict[str, Tensor]):
        self.data = data
        self.keys = list(self.data.keys())
        for v in self.data.values():
            # assert v.size(0) == self.data[self.keys[0]].size(0)
            assert len(v) == self.data[self.keys[0]].size(0)
            # print(v.size())

    def __len__(self):
        return self.data[self.keys[0]].size(0)

    def __getitem__(self, idx):
        res = {k: v[idx] for k, v in self.data.items()}
        # if "index" not in res:
        #     res["index"] = torch.LongTensor([idx])
        return res
