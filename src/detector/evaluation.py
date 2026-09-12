"""Layer 3, part 4: evaluation.

Accuracy is deliberately NOT the headline metric (spec #8) - on an imbalanced
phishing corpus a "predict benign" stub scores high. We report precision,
recall, F1, the confusion matrix, ROC-AUC and PR-AUC, and a threshold sweep so
the precision/recall trade-off is visible (spec #8, #9).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_at_threshold(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"benign": int((y_true == 0).sum()), "phishing": int((y_true == 1).sum())},
    }
    # AUCs are threshold-independent but need both classes present.
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:  # pragma: no cover - guarded for tiny/degenerate eval sets
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def threshold_sweep(y_true, y_prob, thresholds) -> list[dict]:
    return [evaluate_at_threshold(y_true, y_prob, t) for t in thresholds]


def format_confusion_matrix(cm: dict) -> str:
    """Exactly the layout the spec asks for."""
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    w = max(len(str(v)) for v in (tn, fp, fn, tp, 999999))
    return (
        "                     Predicted\n"
        "                 Benign    Phishing\n"
        f"Actual Benign    {tn:>{w}}      {fp:>{w}}\n"
        f"Actual Phishing  {fn:>{w}}      {tp:>{w}}\n"
    )


def format_report(metrics: dict, *, title: str = "Evaluation") -> str:
    cm = metrics["confusion_matrix"]
    lines = [
        f"=== {title} (threshold = {metrics['threshold']:.2f}) ===",
        f"  Precision : {metrics['precision']:.4f}   "
        "(of URLs flagged phishing, the share truly phishing)",
        f"  Recall    : {metrics['recall']:.4f}   "
        "(of real phishing URLs, the share we caught)",
        f"  F1        : {metrics['f1']:.4f}",
        f"  ROC-AUC   : {metrics['roc_auc'] if metrics['roc_auc'] is None else round(metrics['roc_auc'], 4)}",
        f"  PR-AUC    : {metrics['pr_auc'] if metrics['pr_auc'] is None else round(metrics['pr_auc'], 4)}",
        f"  Accuracy  : {metrics['accuracy']:.4f}   (NOT the primary metric)",
        "",
        format_confusion_matrix(cm),
        f"  False positives (benign flagged phishing): {cm['fp']}  "
        "-> user friction, blocked legit sites",
        f"  False negatives (phishing missed)        : {cm['fn']}  "
        "-> user exposed to the attack",
    ]
    return "\n".join(lines)


def per_source_metrics(y_true, y_prob, sources, threshold: float = 0.5) -> dict:
    """Metrics broken down by dataset source.

    This is a leakage check, not a vanity table. If the model scores near-perfectly
    on `tranco` vs `phishtank` but poorly within a source, it has learned to tell
    the *feeds* apart (Tranco entries are bare `http://domain/`, PhishTank URLs are
    long and messy) rather than learning phishing. Large per-source gaps are the
    signal to look at how the corpus was assembled.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    sources = np.asarray([str(s) for s in sources])

    out: dict = {}
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        if not mask.any():
            continue
        subset_true = y_true[mask]
        subset_prob = y_prob[mask]
        pred = (subset_prob >= threshold).astype(int)
        out[source] = {
            "n": int(mask.sum()),
            "positives": int((subset_true == 1).sum()),
            "mean_probability": float(subset_prob.mean()),
            "accuracy_within_source": float((pred == subset_true).mean()),
        }
        if len(np.unique(subset_true)) == 2:
            out[source]["roc_auc"] = float(roc_auc_score(subset_true, subset_prob))
        else:
            # Single-class source: AUC is undefined, and that itself is the
            # interesting fact (this source contributes only one label).
            out[source]["roc_auc"] = None
            out[source]["single_class"] = True
    return out


def format_per_source(rows: dict) -> str:
    if not rows:
        return "  (no source information available)"
    head = f"  {'source':<14} | {'n':>6} | {'pos':>6} | {'mean P':>7} | {'acc':>6} | {'ROC-AUC':>7}"
    lines = ["  Per-source breakdown (a large gap here suggests the model is "
             "separating FEEDS, not phishing):", head, "  " + "-" * (len(head) - 2)]
    for source, m in rows.items():
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        lines.append(
            f"  {source:<14} | {m['n']:>6} | {m['positives']:>6} | "
            f"{m['mean_probability']:>7.4f} | {m['accuracy_within_source']:>6.4f} | {auc:>7}"
        )
    return "\n".join(lines)


def format_threshold_sweep(rows: list[dict]) -> str:
    head = f"  {'thr':>4} | {'precision':>9} | {'recall':>7} | {'f1':>6} | {'fp':>5} | {'fn':>5}"
    sep = "  " + "-" * (len(head) - 2)
    body = [
        f"  {r['threshold']:>4.2f} | {r['precision']:>9.4f} | {r['recall']:>7.4f} "
        f"| {r['f1']:>6.4f} | {r['confusion_matrix']['fp']:>5} | {r['confusion_matrix']['fn']:>5}"
        for r in rows
    ]
    return "\n".join(["  Threshold sweep (choose on validation, never on test):",
                      head, sep, *body])
