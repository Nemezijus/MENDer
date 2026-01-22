from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..common.json_safety import safe_float_list, safe_float_scalar


def normalize_confusion(confusion_payload: Optional[Dict[str, Any]]) -> Tuple[List[Any], List[List[Any]], List[Any], Any, Any, Any]:
    """Normalize confusion payload from utils.metrics into JSON-friendly pieces.

    Returns: (labels_out, matrix_out, per_class, overall, macro_avg, weighted_avg)
    """
    if confusion_payload is None:
        return [], [], [], None, None, None

    labels_arr = confusion_payload.get("labels")
    matrix_arr = confusion_payload.get("matrix")

    if hasattr(labels_arr, "tolist"):
        labels_base = labels_arr.tolist()
    else:
        labels_base = list(labels_arr) if labels_arr is not None else []

    labels_out = [str(l) if not isinstance(l, (int, float)) else l for l in labels_base]

    if hasattr(matrix_arr, "tolist"):
        cm_mat = matrix_arr.tolist()
    else:
        cm_mat = matrix_arr if matrix_arr is not None else []

    per_class = confusion_payload.get("per_class") or []
    overall = confusion_payload.get("global") or None
    macro_avg = confusion_payload.get("macro_avg") or None
    weighted_avg = confusion_payload.get("weighted_avg") or None

    return labels_out, cm_mat, per_class, overall, macro_avg, weighted_avg


def normalize_roc(roc_raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize ROC payload produced by metrics computer into frontend schema."""
    if roc_raw is None:
        return None

    # Binary ROC
    if "pos_label" in roc_raw:
        auc_val = safe_float_scalar(roc_raw.get("auc"))
        curve = {
            "label": roc_raw.get("pos_label"),
            "fpr": safe_float_list(roc_raw.get("fpr")),
            "tpr": safe_float_list(roc_raw.get("tpr")),
            "thresholds": safe_float_list(roc_raw.get("thresholds")),
            "auc": auc_val,
        }
        return {
            "kind": "binary",
            "curves": [curve],
            "labels": None,
            "macro_auc": auc_val,
            "positive_label": roc_raw.get("pos_label"),
        }

    # Multiclass one-vs-rest ROC
    if "per_class" in roc_raw:
        labels_arr = roc_raw.get("labels")
        if hasattr(labels_arr, "tolist"):
            labels_list = labels_arr.tolist()
        else:
            labels_list = list(labels_arr) if labels_arr is not None else None

        curves: List[Dict[str, Any]] = []
        for entry in roc_raw.get("per_class", []):
            curves.append(
                {
                    "label": entry.get("label"),
                    "fpr": safe_float_list(entry.get("fpr")),
                    "tpr": safe_float_list(entry.get("tpr")),
                    "thresholds": safe_float_list(entry.get("thresholds")),
                    "auc": safe_float_scalar(entry.get("auc")),
                }
            )

        roc_macro_avg = roc_raw.get("macro_avg") or {}
        macro_auc = safe_float_scalar(roc_macro_avg.get("auc")) if "auc" in roc_macro_avg else None

        macro_fpr = roc_macro_avg.get("fpr")
        macro_tpr = roc_macro_avg.get("tpr")
        if macro_fpr is not None and macro_tpr is not None:
            curves.append(
                {
                    "label": "macro",
                    "fpr": safe_float_list(macro_fpr),
                    "tpr": safe_float_list(macro_tpr),
                    "thresholds": [],
                    "auc": macro_auc if macro_auc is not None else 0.0,
                }
            )

        roc_micro_avg = roc_raw.get("micro_avg") or {}
        micro_auc: Optional[float] = None
        micro_fpr = roc_micro_avg.get("fpr")
        micro_tpr = roc_micro_avg.get("tpr")
        micro_thresholds = roc_micro_avg.get("thresholds")

        if micro_fpr is not None and micro_tpr is not None and micro_thresholds is not None:
            micro_auc = safe_float_scalar(roc_micro_avg.get("auc"))
            curves.append(
                {
                    "label": "micro",
                    "fpr": safe_float_list(micro_fpr),
                    "tpr": safe_float_list(micro_tpr),
                    "thresholds": safe_float_list(micro_thresholds),
                    "auc": micro_auc,
                }
            )

        payload: Dict[str, Any] = {
            "kind": "multiclass",
            "curves": curves,
            "labels": labels_list,
            "macro_auc": macro_auc,
        }
        if micro_auc is not None:
            payload["micro_auc"] = micro_auc
        return payload

    return None
