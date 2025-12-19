#!/usr/bin/env python3
"""
Generate small .npy fixtures for Clojure reader tests.

Outputs to: test-resources/npy-fixtures/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


OUTDIR = Path("npy-fixtures")

SHAPES = [
    (1,),
    (10,),
    (1, 10),
    (2, 3),
    (10, 1),
    (10, 12),
]

FORTRAN_SHAPES = [
    (2, 3),
    (10, 12),
]

DTYPES = {
    "u1": np.dtype("<u1"),
    "u2": np.dtype("<u2"),
    "u4": np.dtype("<u4"),
    # NB no u8 since that requires BigInteger in java

    "i1": np.dtype("<i1"),
    "i2": np.dtype("<i2"),
    "i4": np.dtype("<i4"),
    "i8": np.dtype("<i8"),

    "f4": np.dtype("<f4"),
    "f8": np.dtype("<f8"),
}

BIG_ENDIAN_DTYPES = {
    "i2": np.dtype(">i2"),
    "u4": np.dtype(">u4"),
    "f8": np.dtype(">f8"),
}

BIG_ENDIAN_SHAPES = [
    (2, 3),
]


def make_values(shape: tuple[int, ...],
                dtype: np.dtype) -> np.ndarray:
    """
    Deterministic contents that are:
    - nontrivial (not all zeros)
    - safe for integer ranges
    - stable across runs
    """
    n = int(np.prod(shape))

    # Use a simple arithmetic progression then reshape.
    # For floats, include a fractional component.
    if dtype.kind in ("f",):
        base = (np.arange(n, dtype=np.float64) * 1.25) - 3.0
        arr = base.astype(dtype)
    elif dtype.kind in ("u", "i"):
        base = (np.arange(n, dtype=np.int64) * 17) - 5
        # For unsigned, wrap naturally via astype.
        arr = base.astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype kind: {dtype}")

    arr = arr.reshape(shape, order="C")
    return arr


def save_fixture(arr: np.ndarray,
                 name: str,
                 order: str = "C") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.npy"

    # Ensure order
    if order == "F":
        arr_o = np.asfortranarray(arr)
    elif order == "C":
        arr_o = np.ascontiguousarray(arr)
    else:
        raise ValueError(f"Unsupported order: {order}")

    np.save(path, arr_o, allow_pickle=False)
    print(
        f"Wrote {path}  dtype={arr_o.dtype}  shape={arr_o.shape}  "
        f"C={arr_o.flags['C_CONTIGUOUS']} F={arr_o.flags['F_CONTIGUOUS']}"
    )


def main() -> None:
    # 1) Every dtype on a single shape (2,3)
    shape = (2, 3)
    for key, dt in DTYPES.items():
        arr = make_values(shape, dt)
        save_fixture(arr, f"shape_2x3__dtype_{key}")

    # 2) Every shape on a single dtype u4
    dt = DTYPES["u4"]
    for shape in SHAPES:
        arr = make_values(shape, dt)
        # Make filename readable
        if len(shape) == 1:
            shape_tag = f"{shape[0]}_"
        else:
            shape_tag = "x".join(str(s) for s in shape)
        save_fixture(arr, f"shape_{shape_tag}__dtype_u4")

    # Optional: write one “known small” array you can eyeball in tests
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=DTYPES["i4"])
    save_fixture(arr, "manual_2x3__dtype_i4")

    # 3) Every shape on a single dtype u4, but Fortran order
    dt = DTYPES["u4"]
    for shape in FORTRAN_SHAPES:
        arr = make_values(shape, dt)
        if len(shape) == 1:
            shape_tag = f"{shape[0]}_"
        else:
            shape_tag = "x".join(str(s) for s in shape)
        save_fixture(arr, f"shape_{shape_tag}__dtype_u4__order_F", order="F")

    # 4) Big-endian fixtures (C-order), a few representative dtypes
    for shape in BIG_ENDIAN_SHAPES:
        for key, dt in BIG_ENDIAN_DTYPES.items():
            arr = make_values(shape, dt)  # NOTE: dtype carries endianness
            shape_tag = "x".join(str(s) for s in shape)
            save_fixture(arr, f"shape_{shape_tag}__dtype_{key}__endian_B")


if __name__ == "__main__":
    main()
