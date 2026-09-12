"""The command-line interface: subcommands, JSON output, exit codes.

The `analyze` tests run with `--layer1-only --no-llm --no-cache`, so they need
no network, no LLM and no cache files - only a trained model, which the repo
ships after `python -m detector train --sample`. Where no model is present the
CLI must still produce a well-formed "uncertain" result, and that is asserted
too.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakePredictor
from detector.cli import build_parser, main, render_analysis
from detector.pipeline import analyze_url

OFFLINE = ["--layer1-only", "--no-llm", "--no-cache"]


def _run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


# ---------------------------------------------------------------------------
# parser wiring
# ---------------------------------------------------------------------------
def test_subcommands_exist():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])                    # a subcommand is required

    for command in ("analyze", "features", "evaluate"):
        args = parser.parse_args(
            [command] + (["--csv", "x.csv"] if command == "evaluate" else ["http://x.com"])
        )
        assert args.command == command


def test_analyze_flags_parse():
    args = build_parser().parse_args(
        ["analyze", "--json", "--no-llm", "--no-cache", "--low", "0.2", "--high", "0.8",
         "http://x.com"])
    assert args.json is True and args.no_llm is True
    assert args.low == 0.2 and args.high == 0.8


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def test_analyze_json_is_a_single_parseable_document(capsys):
    _code, out = _run(["analyze", "--json", *OFFLINE,
                       "http://paypal.com-verify.tk/login"], capsys)
    payload = json.loads(out)

    assert isinstance(payload, dict)
    assert payload["classification"] in ("phishing", "safe", "uncertain")
    assert "risk_score" in payload and "ml_probability" in payload
    assert "reasons" in payload and isinstance(payload["reasons"], list)
    assert "not a calibrated probability" in payload["risk_score_note"]


def test_analyze_json_returns_an_array_for_several_urls(capsys):
    _code, out = _run(["analyze", "--json", *OFFLINE,
                       "http://paypal.com-verify.tk/x", "https://example.com/"], capsys)
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_analyze_json_reasons_carry_machine_codes(capsys):
    _code, out = _run(["analyze", "--json", *OFFLINE,
                       "https://paypal.security-check.xyz/login"], capsys)
    payload = json.loads(out)
    codes = {r["code"] for r in payload["reasons"]}
    assert "l1_brand_in_wrong_place" in codes
    assert all({"code", "message", "severity", "layer", "source"} == set(r)
               for r in payload["reasons"])


def test_json_output_redacts_credentials(capsys):
    _code, out = _run(["analyze", "--json", *OFFLINE,
                       "https://site.example.com/login?password=hunter2"], capsys)
    assert "hunter2" not in out


# ---------------------------------------------------------------------------
# human-readable output
# ---------------------------------------------------------------------------
def test_human_output_shows_every_layer(capsys):
    _code, out = _run(["analyze", *OFFLINE, "http://paypal.com-verify.tk/login"], capsys)

    assert "LAYER 1" in out and "LAYER 2" in out
    assert "LAYER 3" in out and "LAYER 4" in out
    assert "RESULT" in out
    assert "reasons:" in out


def test_human_output_states_the_score_is_not_calibrated(capsys):
    _code, out = _run(["analyze", *OFFLINE, "https://example.com/"], capsys)
    assert "NOT a calibrated probability" in out


def test_human_output_says_when_layer4_was_skipped(capsys):
    _code, out = _run(["analyze", *OFFLINE, "http://paypal.com-verify.tk/login"], capsys)
    assert "not needed" in out or "not run" in out


def test_render_handles_a_result_with_no_model():
    result = analyze_url("https://example.com/", run_layer2=False, use_caches=False,
                         predictor=FakePredictor(None, exc=RuntimeError("no model")))
    rendered = render_analysis(result)
    assert "LAYER 3" in rendered
    assert "UNCERTAIN" in rendered


# ---------------------------------------------------------------------------
# exit codes carry the verdict
# ---------------------------------------------------------------------------
def test_exit_code_one_for_phishing(capsys):
    code, _out = _run(["analyze", "--json", *OFFLINE,
                       "http://paypal.com-verify.tk/login/secure"], capsys)
    assert code == 1


def test_exit_code_zero_for_safe(capsys):
    code, _out = _run(["analyze", "--json", *OFFLINE, "https://www.wikipedia.org/"], capsys)
    assert code in (0, 2)          # 2 if no model is trained in this checkout


# ---------------------------------------------------------------------------
# features subcommand
# ---------------------------------------------------------------------------
def test_features_outputs_raw_layers_with_no_verdict(capsys):
    _code, out = _run(["features", "--layer1-only", "--pretty",
                       "https://paypal.security-check.xyz/login"], capsys)
    payload = json.loads(out)

    assert "layer1_lexical" in payload
    assert payload["layer1_lexical"]["brand_in_wrong_place"] is True
    assert "classification" not in payload      # features only, never a decision
    assert "risk_score" not in payload
