from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class BaseHyperparameters:
    train_steps: int = 100000  # number of train_loop steps
    burn_in: int = 1000  # how many steps to loop for before starting training
    train_every: int = 1  # how many steps per train call
    evaluate_every: int = 10_000  # how many steps per evaluation call
    evaluate_episodes: int = 5  # how many episodes we complete each evaluation call
    batch_size: int = 32  # batch size for training
    buffer_size_gathered: int = 100000  # buffer size for gathered data
    buffer_size_dataset: int = (
        100000  # buffer size for the provided data i.e. how much provided data to use
    )
    gather_every: int = 1  # how often we collect transition data
    gather_n: int = 1  # how many transitions we collect at once