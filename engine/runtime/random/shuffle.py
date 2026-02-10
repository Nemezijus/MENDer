from __future__ import annotations

from typing import Union, Any
import numpy as np


__all__ = ["shuffle_simple_vector"]


def shuffle_simple_vector(
    vec: Any,
    rng: Union[None, int, np.random.Generator] = None,
) -> np.ndarray:
    """
    Return a shuffled copy of a 1D vector.

    Parameters
    ----------
    vec : array-like of shape (n,)
        1D vector to shuffle (NumPy array, list, tuple, or pandas Series).
    rng : None | int | numpy.random.Generator, optional
        Random generator or seed.
        - If int: a new Generator is created with that seed (reproducible).
        - If Generator: it will be used directly.
        - If None: uses np.random.default_rng().

    Returns
    -------
    np.ndarray of shape (n,)
        Shuffled copy of `vec`. The original is not modified.

    Raises
    ------
    ValueError
        If `vec` is not 1D.
    """
    arr = np.asarray(vec)
    if arr.ndim != 1:
        raise ValueError(
            f"shuffle_simple_vector expects a 1D vector, got shape {arr.shape}."
        )

    # Create/use RNG
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # permutation returns a shuffled copy (not in-place)
    shuffled = gen.permutation(arr)

    # Ensure contiguous memory and preserve dtype
    return np.ascontiguousarray(shuffled)