import torch
from torch import nn
from torch.nn import functional as F
from allennlp.nn.util import masked_log_softmax
from torch.nn import HingeEmbeddingLoss


class ProductSimilarity(nn.Module):
    def __init__(self, input_size):
        super(ProductSimilarity, self).__init__()

    def forward(self, x1, x2):
        return x1.bmm(x2.transpose(1, 2))


class MultiClassHingeLoss(nn.Module):
    def __init__(self, margin, ignore_index=-1):
        super(MultiClassHingeLoss, self).__init__()
        self.margin = margin
        self.ignore_index = ignore_index

    def forward(self, x, target, do_mean: bool = False):
        """
        x: (batch, *, C)
        target: (batch, *)
        """
        x = x.to(dtype=torch.float)  # avoid half precision
        assert target.size() == x.size()[:-1]
        ignore_mask = (target == -1)
        target.masked_fill_(ignore_mask, 0)
        target_score = x.gather(index=target.unsqueeze(-1), dim=-1)
        target_mask = F.one_hot(target, num_classes=x.size(-1)).to(dtype=torch.uint8)
        masked_target = x.masked_fill(target_mask, -40000.0)
        second_max, _ = masked_target.max(dim=-1)
        loss = self.margin - target_score + second_max
        loss[loss < 0] = 0
        loss[ignore_mask] = 0
        loss = loss.sum()
        if do_mean:
            loss = loss / x.size(0)
        return loss


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
