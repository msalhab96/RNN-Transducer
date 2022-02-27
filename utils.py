from abc import ABC, abstractmethod
from typing import Union, Tuple
from datetime import datetime
from torch.nn import Module
from pathlib import Path
import torchaudio
import torch
import json


class IPipeline(ABC):
    @abstractmethod
    def run():
        """Used to run all the callables functions sequantially
        """
        pass


def save_json(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_audio(file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    x, sr = torchaudio.load(file_path, normalize=True)
    return x, sr


def get_formated_date() -> str:
    """Used to generate time stamp
    Returns:
        str: a formated string represnt the current time stap
    """
    t = datetime.now()
    return f'{t.year}{t.month}{t.day}-{t.hour}{t.minute}{t.second}'


def load_stat_dict(model: Module, model_path: Union[str, Path]) -> None:
    """Used to load the weigths for the given model
    Args:
        model (Module): the model to load the weights into
        model_path (Union[str, Path]): tha path of the saved weigths
    """
    model.load_state_dict(torch.load(model_path))
