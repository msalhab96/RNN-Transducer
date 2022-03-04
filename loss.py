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

    def get_phi_probs(self, probs: Tensor, c: int, p: int) -> Tensor:
        return probs[:, c + p - 1, self.phi_idx]

    def get_chars_probs(
            self, probs: Tensor, target: Tensor, c: int, p: int
            ) -> Tensor:
        all_seqs = probs[:, p + c - 1]
        result = torch.index_select(all_seqs, dim=-1, index=target[:, c - 1])
        return result[0, :]
