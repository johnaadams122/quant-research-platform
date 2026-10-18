import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import render_near_miss_log
from mining import trial_ledger


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["render_near_miss_log.py"] + argv)
    return render_near_miss_log.main()


def test_rejects_out_path_identical_to_ledger_path(tmp_path, monkeypatch):
    """Guards against --out resolving to the SAME path as --ledger
    (a plausible CLI typo/bad flag order, e.g.
    `py scripts/render_near_miss_log.py --out data/mining/trial_ledger.jsonl`)
    must never reach `open(args.out, "w")`, which would truncate the
    append-only ledger it just read from and overwrite it with Markdown."""
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_a", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.5)
    original_bytes = open(ledger, "rb").read()

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--ledger", ledger, "--out", ledger])

    # Proof the guard actually bites: if it were missing (or
    # merely a string-equality check bypassed by aliasing), the ledger
    # would come back truncated/overwritten with Markdown. It must be
    # byte-identical to what was there before the call.
    assert open(ledger, "rb").read() == original_bytes


def test_rejects_out_path_that_aliases_ledger_path_via_relative_segments(
        tmp_path, monkeypatch):
    """Same guard, but the two path STRINGS differ while resolving to the
    identical real file via a '..' segment -- a plain string-equality check
    would miss this; only realpath/abspath normalization catches it."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    ledger = str(real_dir / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_a", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.5)
    original_bytes = open(ledger, "rb").read()

    aliased_out = os.path.join(
        str(tmp_path), "real", "sub", "..", "ledger.jsonl")
    assert aliased_out != ledger, "test setup must use a differing string"

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--ledger", ledger, "--out", aliased_out])

    assert open(ledger, "rb").read() == original_bytes


def test_accepts_a_genuinely_different_out_path(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_a", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.5)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    exit_code = _run_main(monkeypatch, ["--ledger", ledger, "--out", out])

    assert exit_code == 0
    assert os.path.exists(out)


def test_rejects_nan_margin_instead_of_silently_flagging_everything(
        tmp_path, monkeypatch):
    """Guards against a NaN margin: --margin nan is silently accepted by
    argparse, and since NaN comparisons are always False in Python,
    `distance > margin` would never exclude a row -- every scored cut would
    (silently) report as a near miss. Must error clearly instead."""
    ledger = str(tmp_path / "ledger.jsonl")
    # A cut_dsr score far from any sane bar -- with a NaN margin accepted,
    # this would still get reported as a "near miss".
    trial_ledger.record_outcome(
        ledger, "fam_far", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.10)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    with pytest.raises(SystemExit):
        _run_main(monkeypatch,
                   ["--ledger", ledger, "--out", out, "--margin", "nan"])

    assert not os.path.exists(out)


def test_rejects_nan_threshold(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_far", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.10)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    with pytest.raises(SystemExit):
        _run_main(monkeypatch,
                   ["--ledger", ledger, "--out", out, "--threshold", "nan"])

    assert not os.path.exists(out)


def test_rejects_threshold_outside_the_dsr_probability_range(
        tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_far", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.10)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    with pytest.raises(SystemExit):
        _run_main(monkeypatch,
                   ["--ledger", ledger, "--out", out, "--threshold", "1.5"])
    assert not os.path.exists(out)


def test_rejects_margin_outside_the_dsr_probability_scale(
        tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_far", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.10)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    with pytest.raises(SystemExit):
        _run_main(monkeypatch,
                   ["--ledger", ledger, "--out", out, "--margin", "-0.1"])
    assert not os.path.exists(out)


def test_accepts_a_valid_margin_and_threshold_override(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "fam_close", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.80)
    out = str(tmp_path / "NEAR-MISS-LOG.md")

    exit_code = _run_main(
        monkeypatch,
        ["--ledger", ledger, "--out", out, "--margin", "0.20",
         "--threshold", "0.90"])

    assert exit_code == 0
    text = open(out, encoding="utf-8").read()
    assert "fam_close" in text
