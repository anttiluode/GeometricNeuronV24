"""The V24 lens: a scale and an address produce one scalar measurement.

The hidden image is treated as a piecewise-constant function on an ``n x n``
pixel grid covering the unit square.  A lens of side ``r`` partitions the same
square into ``r x r`` cells.  Reading one address returns the area average over
that cell.  Unlike OpenCV resize, the overlap weights are explicit, testable,
and have a known transpose (the software adjoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class Address:
    """One addressed scalar read: resolution, row, and column."""

    scale: int
    row: int
    col: int


def area_overlap_1d(n: int, scale: int) -> np.ndarray:
    """Return the exact 1-D area-average operator, shape ``(scale, n)``.

    Pixel values are constant over source intervals of width ``1/n``.  Every
    output row sums to one, including non-divisor scales such as 15 on a
    32-pixel axis.
    """

    if n < 1 or scale < 1:
        raise ValueError("n and scale must be positive")
    if scale > n:
        raise ValueError("scale may not exceed the source side")

    source_left = np.arange(n, dtype=np.float64) / n
    source_right = np.arange(1, n + 1, dtype=np.float64) / n
    target_left = np.arange(scale, dtype=np.float64) / scale
    target_right = np.arange(1, scale + 1, dtype=np.float64) / scale

    overlap = np.maximum(
        0.0,
        np.minimum(target_right[:, None], source_right[None, :])
        - np.maximum(target_left[:, None], source_left[None, :]),
    )
    weights = scale * overlap
    if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-14):
        raise ArithmeticError("area weights do not conserve a constant image")
    return weights


def lens_matrix(n: int, scale: int) -> np.ndarray:
    """Return all ``scale**2`` addressed box-average masks.

    Rows are ordered row-major by lens address.  Images are likewise flattened
    row-major, so row ``i*scale+j`` is the mask at ``Address(scale, i, j)``.
    """

    p = area_overlap_1d(n, scale)
    # (row, col, source_y, source_x) -> (address, flattened source image)
    return np.einsum("iy,jx->ijyx", p, p, optimize=True).reshape(
        scale * scale, n * n
    )


def lens_mask(n: int, address: Address) -> np.ndarray:
    """Return one addressed mask as an ``n x n`` array."""

    if not (0 <= address.row < address.scale):
        raise IndexError("row is outside the addressed lens")
    if not (0 <= address.col < address.scale):
        raise IndexError("col is outside the addressed lens")
    h = lens_matrix(n, address.scale)
    return h[address.row * address.scale + address.col].reshape(n, n)


def stack_lenses(n: int, scales: Sequence[int]) -> np.ndarray:
    """Stack every address from each requested scale."""

    if not scales:
        return np.empty((0, n * n), dtype=np.float64)
    return np.vstack([lens_matrix(n, int(scale)) for scale in scales])


def numerical_rank(matrix: np.ndarray, rtol: float = 1e-10) -> int:
    """Numerical rank with a relative threshold tied to the largest singular value."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular > rtol * singular[0]))


def orthonormal_row_basis(matrix: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Return columns spanning the row space of ``matrix``.

    The result has shape ``(n_features, rank)`` and is convenient for projecting
    hidden images onto what a set of pulses can identify.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return np.empty((matrix.shape[1], 0), dtype=np.float64)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.empty((matrix.shape[1], 0), dtype=np.float64)
    rank = int(np.count_nonzero(singular > rtol * singular[0]))
    return vh[:rank].T.copy()


def minimum_norm_reconstruction(
    matrix: np.ndarray, measurements: np.ndarray, rcond: float = 1e-10
) -> np.ndarray:
    """Reconstruct the observable component without inventing a hidden prior."""

    solution, *_ = np.linalg.lstsq(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(measurements, dtype=np.float64),
        rcond=rcond,
    )
    return solution


def adjoint_gradient(
    matrix: np.ndarray, estimate: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Gradient of ``0.5 * ||H estimate - observed||^2``.

    This is a software transpose/backprojection.  It is not, by itself, a
    physical same-device adjoint experiment.
    """

    h = np.asarray(matrix, dtype=np.float64)
    residual = h @ np.asarray(estimate, dtype=np.float64) - np.asarray(
        observed, dtype=np.float64
    )
    return h.T @ residual


def pair_rank_prediction(scale_a: int, scale_b: int, n: int | None = None) -> int:
    """Predict the rank of two complete square partition lenses.

    In the continuous aligned-partition model, the intersection contains the
    ``gcd(scale_a, scale_b) x gcd(scale_a, scale_b)`` common coarse partition.
    Hence ``rank = a^2 + b^2 - gcd(a,b)^2``.  ``n`` optionally caps the result
    at the number of discrete source pixels.
    """

    a = int(scale_a)
    b = int(scale_b)
    if a < 1 or b < 1:
        raise ValueError("scales must be positive")
    predicted = a * a + b * b - gcd(a, b) ** 2
    if n is not None:
        predicted = min(predicted, int(n) ** 2)
    return predicted


def measurements_for_addresses(
    image: np.ndarray, addresses: Iterable[Address]
) -> np.ndarray:
    """Convenience reference implementation for a sequence of scalar reads."""

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")
    n = image.shape[0]
    flat = image.reshape(-1)
    return np.asarray(
        [float(lens_mask(n, address).reshape(-1) @ flat) for address in addresses]
    )
