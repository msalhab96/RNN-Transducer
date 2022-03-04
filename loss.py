import torch
import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, phi_idx: int, eps: float) -> None:
        super().__init__()
        self.phi_idx = phi_idx
        self.eps = eps

    def forward(
            self,
            probs: Tensor,
            target: Tensor,
            target_lengths: Tensor,
            ):
        pass

    def get_score_matrix(
            self, batch_size: int, n_chars: int, n_phis: int
            ) -> Tensor:
        return torch.zeros(batch_size, n_chars, n_phis)
