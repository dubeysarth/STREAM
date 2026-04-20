from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Fixed daily split policy for the Ohio pilot."""

    warmup_start: str = "1998-01-01"
    warmup_end: str = "1998-12-31"
    analysis_start: str = "1999-01-01"
    analysis_end: str = "2019-12-31"
    train_start: str = "1999-01-01"
    train_end: str = "2009-12-31"
    val_start: str = "2010-01-01"
    val_end: str = "2014-12-31"
    test_start: str = "2015-01-01"
    test_end: str = "2019-12-31"

    def build(self) -> dict[str, dict[str, str]]:
        return {
            "warmup": {"start": self.warmup_start, "end": self.warmup_end},
            "analysis": {"start": self.analysis_start, "end": self.analysis_end},
            "train": {"start": self.train_start, "end": self.train_end},
            "val": {"start": self.val_start, "end": self.val_end},
            "test": {"start": self.test_start, "end": self.test_end},
        }

    def split_mask(self, dates: Iterable[pd.Timestamp], split: str) -> pd.Series:
        series = pd.Series(pd.to_datetime(list(dates)))
        bounds = self.build()[split]
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        return (series >= start) & (series <= end)

    def target_indices(
        self,
        dates: Iterable[pd.Timestamp],
        split: str,
        history_length: int,
    ) -> list[int]:
        series = pd.Series(pd.to_datetime(list(dates)))
        start = pd.Timestamp(self.build()[split]["start"])
        end = pd.Timestamp(self.build()[split]["end"])
        warmup_start = pd.Timestamp(self.warmup_start)
        valid = []
        for idx, date in enumerate(series):
            if date < start or date > end:
                continue
            history_start_idx = idx - history_length
            if history_start_idx < 0:
                continue
            if series.iloc[history_start_idx] < warmup_start:
                continue
            valid.append(idx)
        return valid
