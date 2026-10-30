"""Fixtures partagées. ``src/`` n'est pas un package : on l'ajoute au path,
comme le fait le notebook 01."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import eda  # noqa: E402
import features as F  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return F.load_config()


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return F.load_model_frame()


@pytest.fixture(scope="session")
def split(frame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    return F.temporal_split(frame, cfg)


@pytest.fixture(scope="session")
def xy_train(split) -> tuple[pd.DataFrame, pd.Series]:
    return F.split_xy(split[0])


@pytest.fixture(scope="session")
def raw_csv() -> pd.DataFrame:
    """CSV tel que ``pd.read_csv`` le lit : ce que recevra l'API du bloc 4."""
    return pd.read_csv(eda.DEFAULT_DATA_PATH)


V2_CONFIG_PATH = F.PROJECT_ROOT / "configs" / "bloc2_v2.yaml"


@pytest.fixture(scope="session")
def cfg_v2() -> dict:
    return F.load_config(V2_CONFIG_PATH)
