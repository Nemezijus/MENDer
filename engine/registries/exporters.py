from __future__ import annotations

from typing import Callable

from engine.registries.base import Registry

from engine.components.exporters.data_export import DataExporter

ExporterFactory = Callable[[], DataExporter]

_EXPORTERS: Registry[str, ExporterFactory] = Registry(_name="exporters")

_BUILTINS_LOADED = False


def register_export_format(fmt: str) -> Callable[[ExporterFactory], ExporterFactory]:
    return _EXPORTERS.register(fmt.lower())


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from engine.registries.builtins import exporters as _  # noqa: F401
    _BUILTINS_LOADED = True


def make_exporter(fmt: str = "csv") -> DataExporter:
    _ensure_builtins()
    fmt_norm = fmt.lower()
    factory = _EXPORTERS.try_get(fmt_norm)
    if factory is None:
        raise ValueError(f"Unsupported export format: {fmt}")
    return factory()


def list_export_formats() -> list[str]:
    _ensure_builtins()
    return sorted(list(_EXPORTERS.keys()))
