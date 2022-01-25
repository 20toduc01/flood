from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# Code from kornia
def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:    
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = (
        -alpha*torch.pow(probs_neg, gamma)*target*F.logsigmoid(input) 
        - (1 - alpha)*torch.pow(probs_pos, gamma)*(1.0 - target)*F.logsigmoid(-input)
    )

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, reduction)