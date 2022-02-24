from typing import Tuple
from torch import Tensor
import torch.nn as nn


class PredNet(nn.Module):
    """Implements the functionalities of the 
    predict network in the model arcticture

    Attributes
    ----------
    emb : nn.Module
        The network's embedding layer
    lstm : nn.Module
        The network's RNN layer
    """
    def __init__(
            self,
            vocab_size: int,
            emb_dim: int, 
            pad_idx: int,
            hidden_size: int,
            n_layers: int,
            dropout: float
            ) -> None:
        """Constructs all the necessary attributes

        Args:
            vocab_size (int): The number of vocabularies in the dataset, used for the embedding layer
            emb_dim (int): The embedding layer dimensionality
            pad_idx (int): the padding index 
            hidden_size (int): The RNN's hidden layer size
            n_layers (int): the number of stacked LSTM layers in the network
            dropout (float): the dropout rate for each RNN layer
        """
        super().__init__()
        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx
            )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(
            self, 
            x: Tensor, 
            hn: Tensor, 
            cn: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.emb(x)
        out, (hn, cn) = self.lstm(out)
        return out, hn, cn


class TransNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass

class JoinNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass