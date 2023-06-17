"""
Implements Focal loss

MIT License

Copyright (c) 2020 Adeel Hassan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
  """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

  def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0., reduction: str = 'mean'):
    """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError('Reduction must be one of: "mean", "sum", "none".')

    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

    self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none')

  def __repr__(self):
    arg_keys = ['alpha', 'gamma', 'reduction']
    arg_vals = [self.__dict__[k] for k in arg_keys]
    arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
    arg_str = ', '.join(arg_strs)
    return f'{type(self).__name__}({arg_str})'

  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    if x.ndim > 2:
      # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
      c = x.shape[1]
      x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
      # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
      y = y.view(-1)

    # compute weighted cross entropy term: -alpha * log(pt)
    # (alpha is already part of self.nll_loss)
    log_p = F.log_softmax(x, dim=-1)
    ce = self.nll_loss(log_p, y)

    # get true class column from each row
    all_rows = torch.arange(len(x), device=y.device)
    log_pt = log_p[all_rows, y]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt)**self.gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce

    if self.reduction == 'mean':
      loss = loss.mean()
    elif self.reduction == 'sum':
      loss = loss.sum()

    return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
  """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
  if alpha is not None:
    if not isinstance(alpha, Tensor):
      alpha = torch.tensor(alpha)
    alpha = alpha.to(device=device, dtype=dtype)

  fl = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
  return fl
