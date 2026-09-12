"""Layer 3 training entry point.

    python -m detector.train --sample                 # bundled synthetic sample
    python -m detector.train --spec data/raw/sources.json
    python scripts/train.py --help

Steps (spec #12): load -> clean -> Layer 1 features -> Layer 2 features (cached)
-> combine -> time-based split -> train Random Forest -> evaluate val & test ->
print metrics -> save model + schema + feature importance + training metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CACHE_CONFIG, DEFAULT_ML_CONFIG, MLConfig
from .dataset import (
    class_distribution,
    clean_dataset,
    load_many,
    load_normalized_csv,
    time_based_split,
)
from .evaluation import (
    format_per_source,
    format_report,
    format_threshold_sweep,
    per_source_metrics,
    threshold_sweep,
)
from .feature_pipeline import (
    active_feature_names,
    build_dataset,
    feature_matrix,
)
from .feature_schema import SCHEMA_VERSION
from .layer2_cache import Layer2Cache
from .model import PhishingModel

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sample"


def _load_input(args) -> pd.DataFrame:
    if args.sample:
        path = SAMPLE_DIR / "sample_dataset.csv"
        if not path.exists():
            print(f"sample not found at {path}\n"
                  f"generate it with:  python scripts/make_sample_dataset.py",
                  file=sys.stderr)
            sys.exit(2)
        print(f"Loading bundled SAMPLE dataset: {path}")
        print("  (synthetic - for pipeline verification only, NOT a benchmark)")
        return load_normalized_csv(path)

    if args.spec:
        specs = json.loads(Path(args.spec).read_text())
        print(f"Loading {len(specs)} source(s) from {args.spec}")
        return load_many(specs)

    if args.csv:
        print(f"Loading normalised CSV: {args.csv}")
        return load_normalized_csv(args.csv)

    print("nothing to load: pass --sample, --spec FILE, or --csv FILE",
          file=sys.stderr)
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Layer 3 Random Forest")
    src = ap.add_argument_group("input (choose one)")
    src.add_argument("--sample", action="store_true",
                     help="use the bundled synthetic sample dataset")
    src.add_argument("--spec", help="JSON list of {kind,path,...} source specs")
    src.add_argument("--csv", help="a single normalised url,label,source,date CSV")

    ap.add_argument("--no-layer2", action="store_true",
                    help="skip DNS/WHOIS (Layer 1 features only) - fast, offline")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of rows (after cleaning) for a quick run")
    ap.add_argument("--models-dir", default=DEFAULT_ML_CONFIG.models_dir)
    ap.add_argument("--reports-dir", default=DEFAULT_ML_CONFIG.reports_dir)
    ap.add_argument("--cache-path", default=DEFAULT_CACHE_CONFIG.layer2_path)
    ap.add_argument("--train-end", default=DEFAULT_ML_CONFIG.train_end)
    ap.add_argument("--val-end", default=DEFAULT_ML_CONFIG.val_end)
    ap.add_argument("--threshold", type=float, default=DEFAULT_ML_CONFIG.default_threshold)
    ap.add_argument("--no-domain-age-features", action="store_true",
                    help="drop domain-age features entirely (leakage experiment)")
    ap.add_argument("--seed", type=int, default=DEFAULT_ML_CONFIG.random_seed)
    args = ap.parse_args(argv)

    ml = MLConfig(
        random_seed=args.seed,
        train_end=args.train_end,
        val_end=args.val_end,
        default_threshold=args.threshold,
        include_domain_age_features=not args.no_domain_age_features,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
    np.random.seed(ml.random_seed)

    print("=" * 70)
    print("LAYER 3 TRAINING")
    print("=" * 70)

    # 1. load ---------------------------------------------------------------
    raw = _load_input(args)
    print(f"\nRaw rows: {len(raw)}")

    # 2. clean --------------------------------------------------------------
    clean, clean_stats = clean_dataset(raw, ml_config=ml, verbose=True)
    if args.limit:
        clean = (clean.sort_values('date', na_position='last', kind='stable')
                       .groupby('label', group_keys=False).head(args.limit // 2))
        print(f"--limit applied: {len(clean)} rows kept")

    dist = class_distribution(clean)
    print("Class distribution (post-clean):")
    print(f"  phishing={dist['phishing']}  benign={dist['benign']}  "
          f"ratio={dist['ratio_phishing_to_benign']:.3f}:1  "
          f"phishing_fraction={dist['phishing_fraction']:.3f}")
    print(f"  -> class_weight={ml.class_weight!r} (no row duplication)")

    # 3-5. features (Layer 1 + cached Layer 2) --------------------------
    run_l2 = not args.no_layer2
    cache = Layer2Cache(args.cache_path) if run_l2 else None
    print(f"\nBuilding features (layer2={'on' if run_l2 else 'OFF'}"
          f"{', cache=' + args.cache_path if cache is not None else ''}) ...")
    feat_df, feat_stats = build_dataset(
        clean["url"], clean["label"], sources=clean["source"], dates=clean["date"],
        ml_config=ml, run_layer2=run_l2, cache=cache, progress_every=200,
    )
    if cache is not None:
        cache.close()
    print(f"  features per URL: {feat_stats['n_features']}")
    print(f"  extraction failures: {feat_stats['feature_extraction_failures']}")
    if cache is not None:
        print(f"  layer2 cache: {feat_stats['layer2_cache']}")

    feat_df["registered_domain"] = clean["registered_domain"].values

    # 6. time-based split --------------------------------------------
    split = time_based_split(feat_df, ml_config=ml)
    print(f"\nSplit method: {split.method}")
    print(f"  sizes: {split.sizes()}")
    print(f"  date ranges: {split.date_ranges}")
    print(f"  cross-split domains removed: {split.dropped_cross_split_domains}")
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        d = class_distribution(part)
        print(f"  {name:5}: phishing={d['phishing']} benign={d['benign']}")
    if len(split.train) == 0 or len(split.test) == 0:
        print("ERROR: empty train or test after splitting - check dates/data",
              file=sys.stderr)
        return 1

    feature_names = active_feature_names(ml)
    X_train = feature_matrix(split.train, ml)
    y_train = split.train["label"].astype(int)
    X_val = feature_matrix(split.val, ml) if len(split.val) else None
    X_test = feature_matrix(split.test, ml)

    # 7. train ------------------------------------------------------
    print("\nTraining RandomForest "
          f"(n_estimators={ml.n_estimators}, max_depth={ml.max_depth}, "
          f"min_samples_leaf={ml.min_samples_leaf}, seed={ml.random_seed}) ...")
    model = PhishingModel(ml_config=ml, feature_names=feature_names)
    model.fit(X_train, y_train)
    print(f"  model_version = {model.info.model_version}")

    # 8-9. evaluate + threshold sweep (sweep on VALIDATION) --------
    reports = {}
    if X_val is not None and len(split.val) and split.val["label"].nunique() == 2:
        sweep = threshold_sweep(split.val["label"], model.predict_proba(X_val),
                                ml.eval_thresholds)
        print("\n" + format_threshold_sweep(sweep))
        reports["validation_threshold_sweep"] = sweep
        val_metrics = next(r for r in sweep if abs(r["threshold"] - args.threshold) < 1e-9)
        reports["validation"] = val_metrics
        print("\n" + format_report(val_metrics, title="VALIDATION"))
    else:
        print("\n(validation set empty or single-class - skipping sweep)")

    from .evaluation import evaluate_at_threshold
    test_prob = model.predict_proba(X_test)
    test_metrics = evaluate_at_threshold(
        split.test["label"], test_prob, args.threshold
    )
    reports["test"] = test_metrics
    print("\n" + format_report(test_metrics, title="TEST (held-out, future)"))

    # Leakage check: does the model just separate the source feeds?
    if "source" in split.test.columns:
        by_source = per_source_metrics(
            split.test["label"], test_prob, split.test["source"], args.threshold
        )
        reports["test_per_source"] = by_source
        print("\n" + format_per_source(by_source))

    # 10. feature importance -------------------------------------
    importances = model.feature_importances()
    age = model.domain_age_reliance()
    print("\nTop 15 features by importance:")
    for i, (name, imp) in enumerate(importances[:15], 1):
        tag = "  <-- DOMAIN AGE" if name in age["domain_age_features"] else ""
        print(f"  {i:2}. {name:28} {imp:.4f}{tag}")
    print(f"\nDomain-age importance share: {age['domain_age_importance_sum']:.4f} "
          f"(of 1.0 total).  Single largest feature: {age['max_single_feature_importance']:.4f}")
    if age["domain_age_importance_sum"] > ml.domain_age_reliance_warn:
        print("  WARNING: model leans heavily on domain age - check dataset bias "
              "(README 'domain-age leakage').")

    # 11. save model + schema ----------------------------------
    paths = model.save(args.models_dir)
    print(f"\nSaved model:  {paths['model_path']}")
    print(f"Saved schema: {paths['schema_path']}")

    # feature importance file
    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    imp_df = pd.DataFrame(importances, columns=["feature", "importance"])
    imp_df["is_domain_age_feature"] = imp_df["feature"].isin(age["domain_age_features"])
    imp_df.to_csv(rep_dir / "feature_importance.csv", index=False)
    (rep_dir / "feature_importance.json").write_text(imp_df.to_json(orient="records", indent=2))

    # 12. training metadata (spec #18) ------------------------
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model.info.model_version,
        "schema_hash": model.info.schema_hash,
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "random_seed": ml.random_seed,
        "model_params": model.info.params,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "domain_age_features_included": ml.include_domain_age_features,
        "dataset": {
            "input": ("sample" if args.sample else args.spec or args.csv),
            "raw_rows": len(raw),
            "clean_stats": clean_stats.as_dict(),
            "class_distribution": dist,
            "sources": sorted(clean["source"].dropna().unique().tolist()),
        },
        "split": {
            "method": split.method,
            "sizes": split.sizes(),
            "date_ranges": split.date_ranges,
            "train_end": ml.train_end,
            "val_end": ml.val_end,
            "undated_bucket": ml.undated_bucket,
            "cross_split_domains_removed": split.dropped_cross_split_domains,
            "class_distribution": {
                "train": class_distribution(split.train),
                "val": class_distribution(split.val),
                "test": class_distribution(split.test),
            },
        },
        "class_imbalance_handling": {
            "method": f"class_weight={ml.class_weight!r}",
            "note": "inverse-frequency reweighting; no row duplication / SMOTE",
        },
        "threshold": {
            "default": args.threshold,
            "swept": list(ml.eval_thresholds),
            "note": "sweep evaluated on validation only; test uses the default",
        },
        "metrics": reports,
        "feature_importance_top": [
            {"feature": n, "importance": round(v, 6)} for n, v in importances[:25]
        ],
        "domain_age_reliance": age,
        "layer2": feat_stats.get("layer2_cache") if run_l2 else {"enabled": False},
    }
    meta_path = Path(args.models_dir) / ml.metadata_filename
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    (rep_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved metadata: {meta_path}")

    print("\n" + "=" * 70)
    print("DONE.  Reminder: high accuracy != good phishing detector "
          "(README Layer 3).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
