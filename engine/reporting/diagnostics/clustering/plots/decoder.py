from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np
from .utils import histogram_payload


def add_decoder_payload(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Decoder-style per-sample payloads (confidence / likelihood / noise trend)."""
    if np is None:
        return

    if ctx.per_sample is None:
        return

    dec: Dict[str, Any] = {}

    try:
        if ctx.per_sample.get("max_membership_prob") is not None:
            v = np.asarray(ctx.per_sample.get("max_membership_prob")).reshape(-1).astype(float)
            if v.size == ctx.n:
                conf_vals = v[ctx.idx]
                dec["confidence"] = {"values": [float(x) for x in conf_vals.tolist()]}
                h = histogram_payload(conf_vals, bins=30)
                if h is not None:
                    dec["confidence_hist"] = dict(h)

        if ctx.per_sample.get("log_likelihood") is not None:
            v = np.asarray(ctx.per_sample.get("log_likelihood")).reshape(-1).astype(float)
            if v.size == ctx.n:
                ll_vals = v[ctx.idx]
                dec["log_likelihood"] = {"values": [float(x) for x in ll_vals.tolist()]}
                h = histogram_payload(ll_vals, bins=30)
                if h is not None:
                    dec["log_likelihood_hist"] = dict(h)

        if ctx.per_sample.get("is_noise") is not None:
            v = np.asarray(ctx.per_sample.get("is_noise")).reshape(-1).astype(bool)
            if v.size == ctx.n:
                x = np.arange(v.size, dtype=int)
                cum = np.cumsum(v.astype(int))
                frac = cum / np.maximum(1, (x + 1))
                dec["noise_trend"] = {
                    "x": [int(i) for i in x[ctx.idx].tolist()],
                    "y": [float(z) for z in frac[ctx.idx].tolist()],
                }

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.decoder_payload", exc=e)
        return

    if dec:
        out["decoder"] = dec
