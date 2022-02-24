from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn


class PredNet(nn.Module):
    """Implements the functionalities of the
    predict network in the model architecture

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
            vocab_size (int): The number of vocabularies in the dataset,
            used for the embedding layer
            emb_dim (int): The embedding layer dimensionality
            pad_idx (int): the padding index
            hidden_size (int): The RNN's hidden layer size
            n_layers (int): the number of stacked LSTM layers in the network
            dropout (float): the dropout rate for each RNN layer
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
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
        self.validate_dims(x, hn, cn)
        out = self.emb(x)
        out, (hn, cn) = self.lstm(out, (hn, cn))
        return out, hn, cn

    def validate_dims(
            self,
            x: Tensor,
            hn: Tensor,
            cn: Tensor
            ) -> None:
        assert hn.shape[0] == self.n_layers, \
            'The hidden state should match the number of layers'
        assert hn.shape[2] == self.hidden_size, \
            'The hidden state should match the hiden size'
        assert cn.shape[0] == self.n_layers, \
            'The cell state should match the number of layers'
        assert cn.shape[2] == self.hidden_size, \
            'The cell state should match the hiden size'


class TransNet(nn.Module):
    """Implements the functionalities of the
    transcription network in the model architecture, where
    the input is the speech features and projects it to high
    level feature representation.

    Attributes
    ----------
    lstm : nn.Module
        The network's RNN layer
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            n_layers: int,
            dropout: float,
            is_bidirectional: bool
            ) -> None:
        """
        Args:
            input_size (int): The number of input features per time step,
            hidden_size (int): The RNN's hidden layer size
            n_layers (int): the number of stacked LSTM layers in the network
            dropout (float): the dropout rate for each RNN layer
            is_bidirectional (bool): if the RNN layers are bidirectional or not
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=is_bidirectional
        )

    def forward(self, x: Tensor) -> Tensor:
        out, *_ = self.lstm(x)
        return out


class JoinNet(nn.Module):
    """Implements the functionalities of the
    Join network in the model architecture, where
    the inputs are high speech features at time tn and the prediction at step u
    and predicts the next character or phi based on that, there are two
    moods of operations additive and multiplicative mood.

    Attributes
    ----------
    MOODES: dict
        Maps the mode to a function
    fc : nn.Module
        The network's fully connected layer that maps the
        features into vocabulary distribution
    join_mood: Callable
        The mood of operation
    """
    MODES = {
        'multiplicative': lambda f, g: f * g,
        'mul': lambda f, g: f * g,
        'additive': lambda f, g: f + g,
        'add': lambda f, g: f + g
    }

    def __init__(
            self,
            input_size: int,
            vocab_size: int,
            mode: str
            ) -> None:
        """
        Args:
            input_size (int): The dimension of each timesteps features
            vocab_size (int): The number of vocabulary in the corpus
            mode (str): The mode of operations, either mul
            for multiplicative or add for additive
        """
        super().__init__()
        self.join_mood = self.MODES[mode]
        self.fc = nn.Linear(
            in_features=input_size,
            out_features=vocab_size
        )

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        """performs forward propagation step

        Args:
            f (Tensor): The transcription vector at time t of shape (B, 1, h)
            g (Tensor): The prediction vector at step u of shape (B, 1, h)

        Returns:
            Tensor: vocabulary distribution
        """
        out = self.join_mood(f, g)
        out = self.fc(out)
        return torch.softmax(out, dim=-1)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass
