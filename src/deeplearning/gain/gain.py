import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass, field


class Generator(nn.Module):
    """
    Input:
        x: Tabular data
        random: randomly generated values
        missing mask: binary mask
    Out:
        x: Imputed Values
    """

    # TODO:
    # - Define Model for imputation
    # Maybe something simple?
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, random_value: torch.Tensor, missing_mask: torch.Tensor):
        return x


class Discriminator(nn.Module):
    """
    Input:
        x: Tabular data
    Out:
        m: Binary Mask if a value is generated or real
    """

    # TODO:
    # - Define Model that judges
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x > 0


@dataclass
class Gain:
    seed: int = 42
    rng: torch.Generator = field(init=False)
    generator: Generator = field(init=False)
    discriminator: Discriminator = field(init=False)

    def __post_init__(self):
        self.rng = torch.Generator().manual_seed(self.seed)

    def fit(self, x: pd.DataFrame):
        nrows, ncols = x.shape

    def transform(self, x: pd.DataFrame):
        nrows, ncols = x.shape
        mm = x.isna()
        random_values = torch.rand(nrows, ncols, generator=self.rng)
        x = x.fillna(0)
        return self.generator(
            torch.tensor(x.values),
            random_values,
            torch.tensor(mm.values)
        )
