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
            ) -> Tensor:
        batch_size, max_length, *_ = probs.shape
        n_chars = target.shape[1]
        n_nulls = max_length - n_chars
        # initializing the scores matrix
        scores = self.get_score_matrix(batch_size, n_chars, n_nulls)
        # going over all possible alignment paths
        for c in range(n_chars + 1):
            for p in range(n_nulls + 1):
                if c == 0 and p == 0:
                    # keeping scores[:, c, p] zeros
                    continue
                scores = self.update_scores(scores, probs, target, p, c)
        return self.calc_loss(scores, target_lengths)

    def calc_loss(self, scores: Tensor, target_lengths: Tensor) -> Tensor:
        """Calculates the loss from the given loglikelhood of all paths

        Args:
            scores (Tensor): The score matrix
            target_lengths (Tensor): A tensor contains the lengths of
            the true target

        Returns:
            Tensor: The loss
        """
        # should we normalize by the number of paths ?
        loss = torch.diagonal(torch.index_select(
            scores[:, :, -1], dim=1, index=target_lengths
            ))
        loss *= -1
        return loss.mean()

    def get_score_matrix(
            self, batch_size: int, n_chars: int, n_nulls: int
            ) -> Tensor:
        """Returns a zeros matrix with (B, n_chars, n_nulls) shape

        Args:
            batch_size (int): the target batch size
            n_chars (int): the number of maximum length of chars
            n_nulls (int): the number of nulls to be added to reach the
            maximum length

        Returns:
            Tensor: Zeros matrix with (B, n_chars, n_nulls) shape
        """
        return torch.zeros(batch_size, n_chars + 1, n_nulls + 1)

    def update_scores(
            self, scores: Tensor, probs: Tensor, target: Tensor, p: int, c: int
            ) -> Tensor:
        """Updates the given scores matrix based on the values of p and c

        Args:
            scores (Tensor): The scores matrix
            probs (Tensor): The probabilities scores out of the model
            target (Tensor): The target values
            p (int): The location on the nulls dimension in the scores
            matrix
            c (int): The location on the characters dimension in the scores
            matrix

        Returns:
            Tensor: The updated scores matrix
        """
        if p == 0:
            chars_probs = self.get_chars_probs(probs, target, c, p)
            scores[:, c, p] = self.log(chars_probs) + scores[:, c - 1, p]
            return scores
        elif c == 0:
            phi_probs = self.get_phi_probs(probs, c, p)
            scores[:, c, p] = self.log(phi_probs) + scores[:, c, p - 1]
            return scores
        chars_probs = self.get_chars_probs(probs, target, c, p)
        phi_probs = self.get_phi_probs(probs, c, p)
        scores[:, c, p] = scores[:, c, p - 1] + self.log(phi_probs)
        scores[:, c, p] += scores[:, c - 1, p] + self.log(chars_probs)
        return scores

    def get_phi_probs(self, probs: Tensor, c: int, p: int) -> Tensor:
        return probs[:, c + p - 1, self.phi_idx]

    def get_chars_probs(
            self, probs: Tensor, target: Tensor, c: int, p: int
            ) -> Tensor:
        all_seqs = probs[:, p + c - 1]
        result = torch.index_select(all_seqs, dim=-1, index=target[:, c - 1])
        return torch.diagonal(result)

    def log(self, input: Tensor) -> Tensor:
        return torch.log(self.eps + input)
