
import numpy as np


def _np_ffill_1d(arr: np.ndarray) -> np.ndarray:
    """Perform forward fill on a 1D NumPy array."""
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    idx = np.where(~mask, np.arange(len(arr)), 0)

    if mask[0] and np.any(~mask):
        first_valid_idx = np.min(np.where(~mask)[0])
        idx[0] = first_valid_idx
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def _np_bfill_1d(arr: np.ndarray) -> np.ndarray:
    """Perform backward fill on a 1D NumPy array."""
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    idx = np.where(~mask, np.arange(len(arr)), len(arr) - 1)

    if mask[-1] and np.any(~mask):
        last_valid_idx = np.max(np.where(~mask)[0])
        idx[-1] = last_valid_idx
    idx = np.minimum.accumulate(idx[::-1])[::-1]
    return arr[idx]

def numpy_ffill_bfill_channelwise(arr_3d: np.ndarray) -> np.ndarray:
    """
    Performs a 2D spatial ffill and then bfill channel-wise on a 3D NumPy array (C, H, W).
    This version correctly handles spatial data by filling along rows and then columns.
    """
    C, H, W = arr_3d.shape
    filled_arr = np.copy(arr_3d)

    for c in range(C):
        channel_2d = filled_arr[c, :, :]

        if not np.isnan(channel_2d).any():
            continue


        for i in range(H):
            channel_2d[i, :] = _np_ffill_1d(channel_2d[i, :])

        for j in range(W):
            channel_2d[:, j] = _np_ffill_1d(channel_2d[:, j])


        for i in range(H):
            channel_2d[i, :] = _np_bfill_1d(channel_2d[i, :])

        for j in range(W):
            channel_2d[:, j] = _np_bfill_1d(channel_2d[:, j])

        filled_arr[c, :, :] = channel_2d

    return filled_arr