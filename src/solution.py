import numpy as np
import pandas as pd


def ops_ts_argmin(input_path: str, window: int = 20) -> np.ndarray:
    df = (
        pd.read_parquet(input_path, columns=["symbol", "date", "hhmm", "Close"])
        .sort_values(["symbol", "date", "hhmm"], kind="mergesort")
    )

    ts_argmin = df.groupby("symbol", sort=False)["Close"].transform(
        lambda s: s.rolling(window, min_periods=1).apply(np.argmin, raw=True).add(1.0)
    )

    return ts_argmin.to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)