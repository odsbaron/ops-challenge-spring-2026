import numpy as np
import pandas as pd
import numba as nb
# TESTS: 1.0
@nb.njit
def roll_argmin_numba(arr, window):
    n = len(arr)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        min_idx = start
        min_val = arr[start]
        for j in range(start + 1, i + 1):
            if arr[j] < min_val:
                min_val = arr[j]
                min_idx = j
        res[i] = (min_idx - start) + 1.0
    return res

def ops_ts_argmin(input_path: str, window: int = 20) -> np.ndarray:
    df = pd.read_parquet(input_path, columns=["symbol", "date", "hhmm", "Close"])
    df = df.sort_values(["symbol", "date", "hhmm"], kind="mergesort")
    
    ts_argmin = df.groupby("symbol", sort=False)["Close"].transform(
        lambda s: roll_argmin_numba(s.to_numpy(), window)
    )
    return ts_argmin.to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)