"""Layer 3 tests - dataset loading, cleaning, dedup, and leakage-safe splitting.

Uses tiny local fixtures under tests/fixtures/ - never downloads live feeds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from detector.config import MLConfig
from detector.dataset import (
    class_distribution,
    clean_dataset,
    load_many,
    load_normalized_csv,
    load_openphish,
    load_phishtank,
    load_tranco,
    time_based_split,
)

FIX = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# 1. dataset loading (per source -> common url,label,source,date schema)
# --------------------------------------------------------------------------
def test_load_phishtank():
    df = load_phishtank(FIX / "phishtank_sample.csv")
    assert list(df.columns) == ["url", "label", "source", "date"]
    assert (df["label"] == 1).all()
    assert (df["source"] == "phishtank").all()
    assert df["date"].notna().sum() >= 3           # submission_time parsed
    # the empty-url row is dropped by the loader
    assert not (df["url"].str.strip() == "").any()


def test_load_openphish_has_no_dates():
    df = load_openphish(FIX / "openphish_sample.txt")
    assert (df["label"] == 1).all()
    assert df["date"].isna().all()                 # feed carries no timestamps
    assert not df["url"].str.startswith("#").any() # comment line skipped


def test_load_tranco_builds_benign_urls():
    df = load_tranco(FIX / "tranco_sample.csv", list_date="2026-01-01")
    assert (df["label"] == 0).all()
    assert df["url"].iloc[0].startswith("http://")
    assert df["date"].notna().all()               # whole-list date applied


def test_load_many_concatenates():
    df = load_many([
        {"kind": "phishtank", "path": str(FIX / "phishtank_sample.csv")},
        {"kind": "tranco", "path": str(FIX / "tranco_sample.csv")},
        {"kind": "majestic", "path": str(FIX / "majestic_sample.csv")},
    ])
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["source"].unique()) == {"phishtank", "tranco", "majestic"}


# --------------------------------------------------------------------------
# 2 & 3. cleaning + duplicate removal + stats
# --------------------------------------------------------------------------
def test_clean_dataset_stats_and_dedup():
    df = load_normalized_csv(FIX / "normalized_sample.csv")
    clean, st = clean_dataset(df, ml_config=MLConfig(max_urls_per_registered_domain=None),
                              verbose=False)

    assert st.total_in == 6
    assert st.removed_missing_label == 1           # the blank-label row
    assert st.removed_duplicate_urls == 1          # dup.example/ appears twice
    assert st.invalid_urls_kept == 1              # "not a url" kept + flagged
    assert st.total_out == st.phishing + st.benign
    assert "Removed duplicates" in st.render()
    # nothing silently vanished: in - removed == out (+ kept invalids)
    removed = (st.removed_missing_url + st.removed_missing_label
               + st.removed_bad_label + st.removed_duplicate_urls
               + st.removed_domain_cap + st.empty_host_dropped)
    assert st.total_in - removed == st.total_out


def test_clean_dataset_per_domain_cap():
    rows = [{"url": f"http://campaign-site.com/path{i}", "label": 1,
             "source": "x", "date": f"2025-01-{i+1:02d}T00:00:00+00:00"}
            for i in range(20)]
    rows += [{"url": "http://other-site.com/", "label": 0, "source": "x",
              "date": "2025-01-01T00:00:00+00:00"}]
    df = pd.DataFrame(rows)
    clean, st = clean_dataset(df, ml_config=MLConfig(max_urls_per_registered_domain=5),
                              verbose=False)
    assert st.removed_domain_cap == 15
    assert (clean["registered_domain"] == "campaign-site.com").sum() == 5


def test_class_distribution():
    df = pd.DataFrame({"label": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]})
    d = class_distribution(df)
    assert d["phishing"] == 3 and d["benign"] == 7
    assert d["ratio_phishing_to_benign"] == pytest.approx(3 / 7)


# --------------------------------------------------------------------------
# 6 & 7. time-based split (older -> train, newer -> test)
# --------------------------------------------------------------------------
def _dated_frame():
    rows = []
    # 2025 (train), early 2026 (val), late 2026 (test); distinct domains per bucket
    for i in range(30):
        rows.append({"url": f"http://train{i}.example/", "label": i % 2,
                     "registered_domain": f"train{i}.example",
                     "date": pd.Timestamp("2025-06-01", tz="UTC")})
    for i in range(10):
        rows.append({"url": f"http://val{i}.example/", "label": i % 2,
                     "registered_domain": f"val{i}.example",
                     "date": pd.Timestamp("2026-03-01", tz="UTC")})
    for i in range(10):
        rows.append({"url": f"http://test{i}.example/", "label": i % 2,
                     "registered_domain": f"test{i}.example",
                     "date": pd.Timestamp("2026-07-01", tz="UTC")})
    return pd.DataFrame(rows)


def test_time_based_split_orders_by_date():
    cfg = MLConfig(train_end="2026-01-01", val_end="2026-05-01",
                   drop_cross_split_domains=True)
    res = time_based_split(_dated_frame(), ml_config=cfg)

    assert res.method.startswith("time_based")
    assert len(res.train) == 30 and len(res.val) == 10 and len(res.test) == 10
    assert res.train["date"].max() < pd.Timestamp("2026-01-01", tz="UTC")
    assert res.test["date"].min() >= pd.Timestamp("2026-05-01", tz="UTC")
    # no domain spans train and test
    assert set(res.train["registered_domain"]).isdisjoint(res.test["registered_domain"])


def test_time_split_drops_cross_split_domains():
    df = _dated_frame()
    # inject a domain that appears in both 2025 and late 2026
    leak = pd.DataFrame([
        {"url": "http://leak.example/a", "label": 1,
         "registered_domain": "leak.example", "date": pd.Timestamp("2025-05-01", tz="UTC")},
        {"url": "http://leak.example/b", "label": 1,
         "registered_domain": "leak.example", "date": pd.Timestamp("2026-08-01", tz="UTC")},
    ])
    df = pd.concat([df, leak], ignore_index=True)
    cfg = MLConfig(train_end="2026-01-01", val_end="2026-05-01",
                   drop_cross_split_domains=True)
    res = time_based_split(df, ml_config=cfg)
    assert "leak.example" not in set(res.test["registered_domain"])
    assert res.dropped_cross_split_domains >= 1


def test_undated_rows_go_to_train_by_default():
    df = _dated_frame()
    df.loc[df.index[:5], "date"] = pd.NaT
    cfg = MLConfig(train_end="2026-01-01", val_end="2026-05-01", undated_bucket="train")
    res = time_based_split(df, ml_config=cfg)
    assert "undated_to_train" in res.method


def test_split_falls_back_to_grouped_when_no_time_span():
    rows = [{"url": f"http://d{i}-site.com/", "label": i % 2,
             "registered_domain": f"d{i}-site.com", "date": pd.NaT} for i in range(40)]
    cfg = MLConfig(undated_bucket="exclude")   # -> nothing dated -> fallback
    res = time_based_split(pd.DataFrame(rows), ml_config=cfg)
    assert res.method == "grouped_fallback"
    assert len(res.train) > 0 and len(res.test) > 0
    assert set(res.train["registered_domain"]).isdisjoint(res.test["registered_domain"])
