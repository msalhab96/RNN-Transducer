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

    def get_zeros_hidden_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros((self.n_layers, batch_size, self.hidden_size)),
            torch.zeros((self.n_layers, batch_size, self.hidden_size))
        )

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
    """Implements the full RNN-T model which consists of
        - prediction network
        - transcription network
        - join network
    Attributes
    ----------
    prednet : nn.Module
        The prediction network
    transnet: nn.Module
        The transcrption network
    joinnet: nn.Module
        The join network
    device: str
        The device to do the operations on.
        default to cuda.
    phi_idx: int
        The index of phi symbol
    pad_idx: int
        The index of the padding symbol
    sos_idx: int
        The index of the start of sentence symbol
    """
    def __init__(
            self,
            prednet_params: dict,
            transnet_params: dict,
            joinnet_params: dict,
            phi_idx: int,
            pad_idx: int,
            sos_idx: int,
            device='cuda'
            ) -> None:
        super().__init__()
        self.prednet = PredNet(**prednet_params).to(device)
        self.transnet = TransNet(**transnet_params).to(device)
        self.joinnet = JoinNet(**joinnet_params).to(device)
        self.prednet_hidden_size = prednet_params['hidden_size']
        self.device = device
        self.phi_idx = phi_idx
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx

    def forward(
            self,
            x: Tensor,
            max_length: int
            ) -> Tensor:
        # TODO: Code Refactoring and documentation
        batch_size, T, *_ = x.shape
        counter = self.get_counter_start(batch_size, T)
        counter_ceil = counter + T - 1
        term_state = torch.zeros(batch_size)
        trans_result = self.feed_into_transnet(x)
        # reshaping the results (B, T, F) -> (B * T, F)
        trans_result = trans_result.reshape(batch_size * T, -1)
        h, c = self.prednet.get_zeros_hidden_state(batch_size)
        h = h.to(self.device)
        c = c.to(self.device)
        gu = self.get_sos_seed(batch_size)
        t = 0
        while True:
            t += 1
            out, h, c = self.prednet(gu, h, c)
            fy = trans_result[counter, :].unsqueeze(dim=1)
            preds = self.joinnet(fy, out)
            if t == 1:
                result = preds
            else:
                result = torch.concat([result, preds], dim=1)
            gu = torch.argmax(preds, dim=-1)
            counter += (gu.cpu() == self.phi_idx).squeeze()
            counter, update_mask = self.clip_counter(counter, counter_ceil)
            term_state = self.update_termination_state(
                term_state, update_mask, t
                )
            if (update_mask.sum().item() == batch_size) or (max_length == t):
                break

    def get_sos_seed(
            self, batch_size: int
            ) -> Tensor:
        return torch.LongTensor([[self.sos_idx]] * batch_size).to(self.device)

    def feed_into_transnet(self, x: Tensor) -> Tensor:
        return self.transnet(x)

    def feed_into_prednet(
            self, yu: Tensor, h: Tensor, c: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.transnet(yu, h, c)

    def get_counter_start(
            self, batch_size: int, max_size: int
            ) -> Tensor:
        return torch.arange(0, batch_size * max_size, max_size)

    def clip_counter(
            self, counter: Tensor, ceil_vector: Tensor
            ) -> Tuple[Tensor, Tensor]:
        """Clips the counter to the ceil values,
        if the value at index i in the counter
        exceeded teh value at index i at the ceil_vector
        it will be assigned to the ceil_vector[i]

        Args:
            counter (Tensor): The counter vector to be updated
            ceil_vector (Tensor): The maximum value at each index of the
                counter values

        Returns:
            Tuple[Tensor, Tensor]: A tuple of the updated counter
            and a boolean tensor where indicates where the values
            are exceeded the limit
        """
        update_mask = counter >= ceil_vector
        upper_bounded = update_mask * ceil_vector
        kept_counts = (counter < ceil_vector) * counter
        return upper_bounded + kept_counts, update_mask

    def update_termination_state(
            self,
            term_state: Tensor,
            update_mask: Tensor,
            last_index: int
            ) -> Tensor:
        """Updates the termination state, where the
        it stores if an example m reached the end of transcription or not

        Args:
            term_state (Tensor): The latest termination state of (N,) shape
            where the value at index i eitehr 0 or number if 0 it's not
            terminated yet otherwise the number indicates the latest position
            to consider.
            update_mask (Tensor): The update mask tensor resulted from the
            clip operation.
            last_index (int): The last index reached in the iteration loop.

        Returns:
            Tensor: The updated term_state tensor
        """
        is_unended = term_state == 0
        to_update = is_unended & update_mask
        return term_state + to_update * last_index
