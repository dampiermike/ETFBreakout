#!/usr/bin/env python3
"""Shared research context: loads the panel once and exposes the wide frames."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

import data
import engine

START = "2010-03-11"


class Context:
    def __init__(self, start: str = START, min_dollar_volume: float = 1e6):
        self.panel = data.load_cached()
        self.start = start
        close = data.to_wide(self.panel, "close")
        self.index = close.index
        self.f: Dict[str, pd.DataFrame] = {
            field: data.to_wide(self.panel, field)
            for field in ("open", "high", "low", "close", "volume")
        }
        self.eligible = data.eligibility_mask(self.panel, min_dollar_volume=min_dollar_volume)
        self.bar_returns = engine.bar_return_frame(self.panel, self.index)
        self._forward = None

    def forward(self, horizons=(1, 3, 5, 10, 20)) -> Dict[int, pd.DataFrame]:
        if self._forward is None or set(horizons) - set(self._forward):
            self._forward = engine.forward_returns(self.panel, self.index, horizons)
        return self._forward

    def slice_start(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[self.start:]


def fmt_stats(row: Dict[str, float]) -> str:
    return (f"CAGR {row['cagr']:7.2%}  vol {row['vol']:6.2%}  Sharpe {row['sharpe']:5.2f}  "
            f"maxDD {row['max_dd']:7.2%}  Calmar {row['calmar']:5.2f}")
