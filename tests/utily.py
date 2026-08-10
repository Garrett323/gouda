import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def missing_data():
    missing_rate = 0.3
    data = np.random.rand(500, 5)
    missing_data = data.copy()
    missing_data[data < missing_rate] = np.nan
    return data, missing_data, data < missing_rate


@pytest.fixture
def missing_cat_data():
    # Reproducibility
    rng = np.random.default_rng(42)

    n_rows = 20

    df = pd.DataFrame({
        "Color": pd.Categorical(
            rng.choice(["Red", "Green", "Blue"], size=n_rows)
        ),
        "Animal": pd.Categorical(
            rng.choice(["Cat", "Dog", "Rabbit", "Bird"], size=n_rows)
        ),
        "Size": pd.Categorical(
            rng.choice(["Small", "Medium", "Large"], size=n_rows),
            categories=["Small", "Medium", "Large"],
            ordered=True,
        ),
        "Region": pd.Categorical(
            rng.choice(["North", "South", "East", "West"], size=n_rows)
        ),
    })
    missing_fraction = 0.20

    mask = rng.random(df.shape) < missing_fraction
    missing = df.mask(mask, pd.NA)
    return df, missing, mask


@pytest.fixture
def missing_mixed_data():
    # Reproducibility
    rng = np.random.default_rng(42)

    n_rows = 20

    df = pd.DataFrame({
        "Color": pd.Categorical(
            rng.choice(["Red", "Green", "Blue"], size=n_rows)
        ),
        "Animal": pd.Categorical(
            rng.choice(["Cat", "Dog", "Rabbit", "Bird"], size=n_rows)
        ),
        "Size": pd.Categorical(
            rng.choice(["Small", "Medium", "Large"], size=n_rows),
            categories=["Small", "Medium", "Large"],
            ordered=True,
        ),
        "Region": pd.Categorical(
            rng.choice(["North", "South", "East", "West"], size=n_rows)
        ),
    })
    missing_fraction = 0.20

    mask = rng.random(df.shape) < missing_fraction
    missing = df.mask(mask, pd.NA)
    return df, missing, mask
