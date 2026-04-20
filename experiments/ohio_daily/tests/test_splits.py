import pandas as pd

from devp_ohio.splits import SplitSpec


def test_split_build_and_history_indices() -> None:
    spec = SplitSpec()
    dates = pd.date_range("1998-01-01", "1999-01-10", freq="D")
    indices = spec.target_indices(dates, split="train", history_length=5)
    assert indices
    assert pd.Timestamp(dates[indices[0]]) >= pd.Timestamp(spec.train_start)
