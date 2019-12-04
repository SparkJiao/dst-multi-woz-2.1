import torch
from torch import nn
from torch.nn import functional as F
from allennlp.nn.util import masked_log_softmax


class ProductSimilarity(nn.Module):
    def __init__(self, input_size):
        super(ProductSimilarity, self).__init__()

    def forward(self, x1, x2):
        return x1.bmm(x2.transpose(1, 2))



# ===============================
# Function
# ===============================


def masked_log_softmax_fp16(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    initial_dtype = vector.dtype
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector.float() + (mask + 1e-45).log()
        vector = vector.to(dtype=initial_dtype)
    return F.log_softmax(vector, dim=dim)
