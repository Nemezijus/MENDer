from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class Registry(Generic[K, V]):
    """Minimal registry mapping keys to values.

    This is intentionally small and dependency-free.

    Typical usage:
        REG = Registry[str, Callable[..., Any]]()

        @REG.register("foo")
        def make_foo(...):
            ...

        make = REG.get("foo")

    Values are usually callables, but can be any object.
    """

    _items: Dict[K, V] = field(default_factory=dict)
    _name: str = "registry"

    def register(self, key: K) -> Callable[[V], V]:
        def deco(value: V) -> V:
            self._items[key] = value
            return value

        return deco

    def get(self, key: K) -> V:
        if key not in self._items:
            raise KeyError(f"{self._name}: unknown key {key!r}")
        return self._items[key]

    def try_get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._items.get(key, default)

    def keys(self) -> Iterable[K]:
        return self._items.keys()

    def items(self) -> Iterable[tuple[K, V]]:
        return self._items.items()

    def __contains__(self, key: K) -> bool:  # pragma: no cover
        return key in self._items

    def __iter__(self) -> Iterator[K]:  # pragma: no cover
        return iter(self._items)
