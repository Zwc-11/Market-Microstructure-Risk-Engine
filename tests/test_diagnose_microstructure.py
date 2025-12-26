from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.diagnose_microstructure import _feature_missingness, _forward_returns, _ic_table


def test_forward_returns():
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="1min", tz="UTC")
    bars = pd.DataFrame({"mid_close": [100.0, 110.0, 121.0]}, index=idx)

    fwd = _forward_returns(bars, horizons=[1, 2], price_col="mid_close")
    assert np.isclose(fwd[1].iloc[0], 0.1)
    assert np.isclose(fwd[2].iloc[0], 0.21)


def test_ic_table_and_missingness():
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="1min", tz="UTC")
    features = pd.DataFrame({"x": [1.0, 2.0, np.nan]}, index=idx)
    returns = {1: pd.Series([1.0, 2.0, 3.0], index=idx)}

    ic = _ic_table(features, returns, method="pearson")
    assert np.isclose(ic["1"]["x"], 1.0)

    missing = _feature_missingness(features)
    assert np.isclose(missing["overall"]["x"], 1.0 / 3.0)
