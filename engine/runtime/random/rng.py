from __future__ import annotations
import hashlib
import numpy as np
from numpy.random import Generator

class RngManager:
    """
    Single source of truth for randomness.
    Creates named, order-independent child seeds/streams by hashing:
      child_seed(name)       -> stable int seed
      child_generator(name)  -> np.random.Generator seeded from that int
    """
    def __init__(self, seed: int | None):
        # Keep a small, well-defined representation
        self._root = 0 if seed is None else int(seed) & 0xFFFFFFFF

    def _mix(self, name: str) -> int:
        # Stable across runs and Python versions
        h = hashlib.sha256(f"{self._root}:{name}".encode("utf-8")).digest()
        # Use 32 bits for compatibility with libraries expecting uint32 seeds
        return int.from_bytes(h[:4], "little", signed=False)

    def child_seed(self, name: str) -> int:
        return self._mix(name)

    def child_generator(self, name: str) -> Generator:
        return np.random.default_rng(self._mix(name))

    def shuffle_generators(self, n: int, base_name: str = "shuffle") -> list[Generator]:
        return [self.child_generator(f"{base_name}_{i}") for i in range(n)]

    def shuffle_seeds(self, n: int, base_name: str = "shuffle") -> list[int]:
        return [self.child_seed(f"{base_name}_{i}") for i in range(n)]
