"""Layer 3 tests - evaluation metrics and threshold sweep."""

from __future__ import annotations

import numpy as np

from detector.evaluation import (
    evaluate_at_threshold,
    format_confusion_matrix,
    format_report,
    threshold_sweep,
)


def test_metrics_on_known_arrays():
    # 4 phishing (1), 4 benign (0)
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    # probs: 3 phishing caught, 1 missed; 1 benign false-positive
    y_prob = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1]

    m = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
    # predicted phishing: idx 0,1,2 (real) + idx 4 (benign) -> tp=3, fp=1, fn=1, tn=3
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == 3 / 4
    assert m["recall"] == 3 / 4
    assert m["f1"] == 0.75
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_threshold_sweep_monotonic_recall():
    rng = np.random.default_rng(1)
    y_true = np.array([1] * 50 + [0] * 50)
    y_prob = np.concatenate([rng.uniform(0.4, 1.0, 50), rng.uniform(0.0, 0.6, 50)])
    rows = threshold_sweep(y_true, y_prob, [0.3, 0.5, 0.7, 0.9])
    recalls = [r["recall"] for r in rows]
    # raising the threshold can only keep or lower recall
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))
    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.7, 0.9]


def test_confusion_matrix_format_layout():
    txt = format_confusion_matrix({"tn": 10, "fp": 2, "fn": 3, "tp": 40})
    assert "Predicted" in txt
    assert "Actual Benign" in txt and "Actual Phishing" in txt
    # benign row shows tn then fp; phishing row shows fn then tp
    lines = [l for l in txt.splitlines() if l.strip()]
    assert lines[-2].split()[-2:] == ["10", "2"]
    assert lines[-1].split()[-2:] == ["3", "40"]


def test_format_report_mentions_fp_fn_tradeoff():
    m = evaluate_at_threshold([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    report = format_report(m, title="X")
    assert "False positives" in report and "False negatives" in report
    assert "Precision" in report and "Recall" in report


def test_single_class_eval_set_does_not_crash():
    m = evaluate_at_threshold([1, 1, 1], [0.9, 0.8, 0.2], 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["recall"] == 2 / 3
