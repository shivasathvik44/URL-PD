"""Command-line interface.

    python -m detector analyze "https://example.com/login"
    python -m detector analyze --json URL [URL ...]
    python -m detector analyze --layer1-only --no-llm URL
    python -m detector train --sample
    python -m detector evaluate --csv data/raw/my_set.csv
    python -m detector features URL           # raw Layer 1+2 features, no scoring

`analyze` prints a per-layer human-readable report by default and a single JSON
document with `--json` (a JSON array when several URLs are given), so it drops
into a shell pipeline without post-processing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config import (
    DEFAULT_CACHE_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_ML_CONFIG,
    decision_config_from_env,
)
from .observability import configure_logging

_SEVERITY_MARK = {"high": "!!", "medium": " !", "low": "  ", "info": "  "}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _fmt(value, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_analysis(result) -> str:
    """Human-readable per-layer report for one AnalysisResult."""
    l1, l2 = result.layer1, result.layer2
    d = result.decision
    out: list[str] = []

    out.append("=" * 72)
    out.append(f"URL          {l1.url}")
    out.append(f"Analysis ID  {result.analysis_id}")
    out.append("=" * 72)

    # --- Layer 1 --------------------------------------------------------
    out.append("\nLAYER 1  lexical (offline)")
    out.append(f"  registered domain .. {l1.registered_domain or '(none)'}")
    out.append(f"  hostname ........... {l1.hostname or '(none)'}")
    out.append(f"  malformed .......... {_fmt(l1.is_malformed)}")
    out.append(f"  brand in wrong place {_fmt(l1.brand_in_wrong_place)}"
               + (f" ({l1.brand_in_wrong_place_name})" if l1.brand_in_wrong_place else ""))
    out.append(f"  typosquat suspicion  {_fmt(l1.is_typosquat_suspicious)}"
               + (f" (~{l1.closest_brand}, distance {l1.levenshtein_distance})"
                  if l1.is_typosquat_suspicious else ""))
    out.append(f"  suspicious TLD ..... {_fmt(l1.is_suspicious_tld)} (.{l1.tld or '?'})")
    out.append(f"  keywords ........... {', '.join(l1.keywords_found) or '(none)'}")
    out.append(f"  entropy ............ {l1.entropy:.3f}")

    # --- Layer 2 --------------------------------------------------------
    out.append("\nLAYER 2  DNS / domain intelligence")
    if l2 is None:
        out.append("  (not run)")
    else:
        out.append(f"  resolves ........... {_fmt(l2.dns.dns_resolves)} "
                   f"(A={l2.dns.ipv4_count}, AAAA={l2.dns.ipv6_count})")
        out.append(f"  NS / MX ............ {l2.dns.ns_count} / {l2.dns.mx_count}")
        out.append(f"  domain age ......... "
                   f"{_fmt(l2.age.age_days, 'unknown')} days"
                   f"{' (NEW)' if l2.age.is_new_domain else ''}"
                   f"  [source: {_fmt(l2.age.source, 'unavailable')}]")
        if l2.tls is not None and l2.tls.checked:
            out.append(f"  TLS ................ available={_fmt(l2.tls.tls_available)} "
                       f"valid={_fmt(l2.tls.cert_valid)} "
                       f"issuer={_fmt(l2.tls.issuer, 'unknown')[:48]}")
        if l2.dns.errors:
            out.append(f"  DNS errors ......... {l2.dns.errors}")

    # --- Layer 3 --------------------------------------------------------
    out.append("\nLAYER 3  machine learning")
    if d.ml_probability is None:
        out.append("  (no model available - train one with: python -m detector train --sample)")
    else:
        out.append(f"  P(phishing) ........ {d.ml_probability:.4f}  "
                   f"[uncalibrated model output]")
        out.append(f"  model .............. {_fmt(result.metadata.get('model_version'))}")

    # --- Layer 4 --------------------------------------------------------
    out.append("\nLAYER 4  page content + LLM")
    if not result.metadata.get("layer4_attempted"):
        out.append("  (not needed - Layers 1-3 were already conclusive)")
    else:
        page = result.page
        if page is not None and page.blocked_reason:
            out.append(f"  fetch .............. BLOCKED by safety policy: {page.blocked_reason}")
        elif page is not None and not page.fetched:
            out.append(f"  fetch .............. failed: {page.fetch_error}")
        elif page is not None:
            s = page.structure
            out.append(f"  fetch .............. {page.status_code} "
                       f"({len(page.redirect_chain)} redirect(s), {page.elapsed_ms:.0f} ms)")
            out.append(f"  title .............. {_fmt(s.title, '(none)')[:60]}")
            out.append(f"  forms .............. {len(s.forms)}"
                       f"  credential={_fmt(s.has_credential_form)}"
                       f"  otp={_fmt(s.has_otp_field)}"
                       f"  payment={_fmt(s.has_payment_field)}")
            if s.cross_domain_form_targets:
                out.append(f"  cross-domain post .. {', '.join(s.cross_domain_form_targets)}")
            out.append(f"  iframes/hidden ..... {len(s.iframes)} iframe(s), "
                       f"{s.hidden_element_count} hidden element(s)")
        llm = result.llm
        if llm is None:
            out.append("  LLM ................ not run")
        elif not llm.performed:
            out.append(f"  LLM ................ unavailable: {llm.error}")
        else:
            out.append(f"  LLM risk ........... {_fmt(llm.risk_level)} "
                       f"(confidence {_fmt(llm.confidence)}, likelihood "
                       f"{_fmt(llm.llm_likelihood)})")
            out.append(f"  brand claimed ...... {_fmt(llm.brand_claimed, '(none)')}"
                       f"  impersonation={_fmt(llm.brand_impersonation)}")
            if llm.prompt_injection_attempt:
                out.append("  PROMPT INJECTION ... page tried to manipulate the analyser "
                           "(instructions were not followed)")
            if llm.schema_violations:
                out.append(f"  schema violations .. {'; '.join(llm.schema_violations[:3])}")
        if llm is not None and (llm.provider or llm.model):
            # Which backend produced (or failed to produce) the analysis above.
            out.append(f"  provider ........... {_fmt(llm.provider, '(unknown)')}")
            out.append(f"  model .............. {_fmt(llm.model, '(unknown)')}")

    # --- verdict --------------------------------------------------------
    out.append("\n" + "-" * 72)
    out.append(f"RESULT       {d.classification.upper()}")
    out.append(f"risk score   {d.risk_score:.2f}   "
               f"(ordinal severity, NOT a calibrated probability)")
    out.append(f"ml / llm     {_fmt(d.ml_probability)} / {_fmt(d.llm_likelihood)}")
    out.append(f"path         {d.decision_path}")
    out.append("\nreasons:")
    if d.reasons:
        for r in d.reasons:
            mark = _SEVERITY_MARK.get(r.severity, "  ")
            out.append(f"  {mark} [{r.severity:<6}] {r.message}")
    else:
        out.append("  (none recorded)")

    if result.errors:
        out.append("\nnotes:")
        for e in result.errors:
            out.append(f"  - {e}")

    out.append(f"\ntimings (ms): {result.timings_ms}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _read_urls(values: list[str]) -> list[str]:
    if values == ["-"]:
        return [line.strip() for line in sys.stdin if line.strip()]
    return values


def cmd_analyze(args) -> int:
    from .pipeline import analyze_url

    decision_config = decision_config_from_env(DEFAULT_DECISION_CONFIG)
    if args.no_llm:
        decision_config = replace(decision_config, enable_layer4=False)
    if args.low is not None:
        decision_config = replace(decision_config, low_threshold=args.low)
    if args.high is not None:
        decision_config = replace(decision_config, high_threshold=args.high)
    if args.no_cache:
        decision_config = replace(
            decision_config,
            cache=replace(decision_config.cache, layer2_enabled=False, page_enabled=False),
        )

    ml_config = DEFAULT_ML_CONFIG
    if args.models_dir:
        ml_config = replace(ml_config, models_dir=args.models_dir)

    results = []
    for url in _read_urls(args.urls):
        results.append(analyze_url(
            url,
            ml_config=ml_config,
            decision_config=decision_config,
            run_layer2=not args.layer1_only,
            do_tls=not args.no_tls,
            use_caches=not args.no_cache,
        ))

    if args.json:
        payload = [r.as_dict() for r in results]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2, default=str))
    else:
        print("\n\n".join(render_analysis(r) for r in results))

    # Exit code carries the verdict for scripting: 0 safe, 1 phishing, 2 uncertain.
    if any(r.classification == "phishing" for r in results):
        return 1
    if any(r.classification == "uncertain" for r in results):
        return 2
    return 0


def cmd_features(args) -> int:
    """Raw Layer 1 (+2) features. No model, no decision - the debugging view."""
    from .pipeline import analyze

    results = [
        analyze(url, run_layer2=not args.layer1_only, do_tls=args.tls).as_dict()
        for url in _read_urls(args.urls)
    ]
    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, indent=2 if args.pretty or args.json else None, default=str))
    return 0


def cmd_train(args, extra: list[str]) -> int:
    from .train import main as train_main

    return train_main(extra)


def cmd_evaluate(args) -> int:
    """Score an existing model against a labelled CSV."""

    from .dataset import clean_dataset, load_normalized_csv
    from .evaluation import (
        evaluate_at_threshold,
        format_per_source,
        format_report,
        format_threshold_sweep,
        per_source_metrics,
        threshold_sweep,
    )
    from .feature_pipeline import build_dataset, feature_matrix
    from .layer2_cache import Layer2Cache
    from .prediction import ModelNotFound, load_model

    ml_config = DEFAULT_ML_CONFIG
    if args.models_dir:
        ml_config = replace(ml_config, models_dir=args.models_dir)

    try:
        model = load_model(args.models_dir, ml_config)
    except ModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # schema mismatch and friends
        print(f"error: model cannot be used: {exc}", file=sys.stderr)
        return 2

    raw = load_normalized_csv(args.csv)
    clean, _stats = clean_dataset(raw, ml_config=ml_config, verbose=not args.json)

    cache = Layer2Cache(args.cache_path or DEFAULT_CACHE_CONFIG.layer2_path) \
        if not args.no_layer2 else None
    frame, _fstats = build_dataset(
        clean["url"], clean["label"], sources=clean["source"], dates=clean["date"],
        ml_config=ml_config, run_layer2=not args.no_layer2, cache=cache,
        progress_every=0 if args.json else 200,
    )
    if cache is not None:
        cache.close()

    X = feature_matrix(frame, ml_config)
    y = frame["label"].astype(int)
    prob = model.predict_proba(X)

    metrics = evaluate_at_threshold(y, prob, args.threshold)
    sweep = threshold_sweep(y, prob, ml_config.eval_thresholds)
    by_source = per_source_metrics(y, prob, frame["source"], args.threshold)

    if args.json:
        print(json.dumps({
            "model_version": model.info.model_version if model.info else None,
            "n_rows": int(len(frame)),
            "metrics": metrics,
            "threshold_sweep": sweep,
            "per_source": by_source,
        }, indent=2, default=str))
    else:
        print(f"\nModel: {model.info.model_version if model.info else '(unknown)'}")
        print(f"Rows evaluated: {len(frame)}")
        print("\n" + format_report(metrics, title="EVALUATION"))
        print("\n" + format_threshold_sweep(sweep))
        print("\n" + format_per_source(by_source))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detector",
        description="Four-layer phishing URL detector "
                    "(lexical -> DNS/domain -> ML -> page/LLM -> decision)",
    )
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="structured log verbosity (default: WARNING)")
    ap.add_argument("--log-json", action="store_true",
                    help="emit logs as one JSON object per line")
    sub = ap.add_subparsers(dest="command", required=True)

    # --- analyze ---------------------------------------------------------
    an = sub.add_parser("analyze", help="run all four layers and decide")
    an.add_argument("urls", nargs="+", help="one or more URLs, or '-' for stdin")
    an.add_argument("--json", action="store_true", help="machine-readable output")
    an.add_argument("--layer1-only", "--no-layer2", dest="layer1_only",
                    action="store_true", help="skip Layer 2 (no DNS/WHOIS traffic)")
    an.add_argument("--no-llm", action="store_true",
                    help="never run Layer 4, even in the uncertain band")
    an.add_argument("--no-tls", action="store_true", help="skip the TLS certificate check")
    an.add_argument("--no-cache", action="store_true", help="bypass both caches")
    an.add_argument("--low", type=float, default=None,
                    help=f"low threshold (default {DEFAULT_DECISION_CONFIG.low_threshold})")
    an.add_argument("--high", type=float, default=None,
                    help=f"high threshold (default {DEFAULT_DECISION_CONFIG.high_threshold})")
    an.add_argument("--models-dir", default=None)
    an.set_defaults(func=cmd_analyze)

    # --- features --------------------------------------------------------
    fe = sub.add_parser("features", help="dump raw Layer 1+2 features (no decision)")
    fe.add_argument("urls", nargs="+", help="one or more URLs, or '-' for stdin")
    fe.add_argument("--json", action="store_true", help="(features are always JSON)")
    fe.add_argument("--pretty", action="store_true", help="indent the JSON")
    fe.add_argument("--layer1-only", "--no-layer2", dest="layer1_only",
                    action="store_true", help="skip Layer 2")
    fe.add_argument("--tls", action="store_true", help="also run the TLS check")
    fe.set_defaults(func=cmd_features)

    # --- train -----------------------------------------------------------
    tr = sub.add_parser("train", help="train the Layer 3 model "
                                      "(remaining args go to detector.train)",
                        add_help=False)
    tr.set_defaults(func=None)

    # --- evaluate --------------------------------------------------------
    ev = sub.add_parser("evaluate", help="score a trained model against a labelled CSV")
    ev.add_argument("--csv", required=True, help="normalised url,label,source,date CSV")
    ev.add_argument("--threshold", type=float, default=DEFAULT_ML_CONFIG.default_threshold)
    ev.add_argument("--no-layer2", action="store_true", help="Layer 1 features only")
    ev.add_argument("--cache-path", default=None)
    ev.add_argument("--models-dir", default=None)
    ev.add_argument("--json", action="store_true", help="machine-readable output")
    ev.set_defaults(func=cmd_evaluate)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `train` forwards everything after it to the training entry point, which
    # has its own (large) flag set.
    if argv and argv[0] == "train":
        from .train import main as train_main

        return train_main(argv[1:])

    ap = build_parser()
    args = ap.parse_args(argv)
    configure_logging(args.log_level, json_format=args.log_json)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
