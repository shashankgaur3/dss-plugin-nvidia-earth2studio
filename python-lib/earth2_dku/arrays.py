"""Helpers for (de)serializing numpy arrays as .npy bytes."""
from __future__ import annotations

import io

import numpy as np


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def load_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def read_npy_header(data: bytes) -> tuple[tuple[int, ...], np.dtype]:
    """Return (shape, dtype) of .npy bytes without loading the array."""
    buf = io.BytesIO(data)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(buf)
    else:
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(buf)
    return tuple(shape), dtype
