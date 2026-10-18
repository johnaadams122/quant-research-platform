import json
import os
import pathlib
import re
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.join(os.path.dirname(_SRC), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from mining import edgar_data, sec_contact

_REPO = pathlib.Path(_SRC).parent
_SYNTHETIC = "Example Research Lab research-contact" + "@" + "example.org"


class _FakeResponse:
    status_code = 404

    def raise_for_status(self):
        pass

    def json(self):
        return {}


class _RecordingGet:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers=None, timeout=None):
        self.calls.append(headers or {})
        return _FakeResponse()


def test_reads_the_setting():
    assert sec_contact.sec_contact_user_agent({"SEC_EDGAR_USER_AGENT": _SYNTHETIC}) == _SYNTHETIC


@pytest.mark.parametrize("value", ["", "   ", "no contact address here"], ids=["empty", "blank", "no-at-sign"])
def test_rejects_missing_or_unusable_values(value):
    with pytest.raises(sec_contact.SecContactMissing):
        sec_contact.sec_contact_user_agent({"SEC_EDGAR_USER_AGENT": value})


def test_rejects_unset():
    with pytest.raises(sec_contact.SecContactMissing):
        sec_contact.sec_contact_user_agent({})


def test_edgar_request_sends_the_configured_contact(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", _SYNTHETIC)
    get = _RecordingGet()
    edgar_data.fetch_company_concept(1234, "SomeTag", http_get=get)
    assert get.calls == [{"User-Agent": _SYNTHETIC}]


def test_edgar_request_refuses_before_any_request_when_unset(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    get = _RecordingGet()
    with pytest.raises(sec_contact.SecContactMissing):
        edgar_data.fetch_company_concept(1234, "SomeTag", http_get=get)
    assert get.calls == []


@pytest.mark.parametrize("module_name, function_name", [
    ("build_capex_watchlist", "_fetch_company_tickers"),
    ("run_retail_inventory_coverage_check", "fetch_company_tickers"),
], ids=["capex-watchlist", "retail-coverage"])
def test_ticker_list_fetchers_send_the_contact_and_refuse_when_unset(monkeypatch, module_name, function_name):
    module = __import__(module_name)
    fetch = getattr(module, function_name)
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    get = _RecordingGet()
    with pytest.raises(sec_contact.SecContactMissing):
        fetch(http_get=get)
    assert get.calls == []
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", _SYNTHETIC)
    try:
        fetch(http_get=get)
    except Exception:
        pass  # payload parsing of the empty fake is not under test; only the header is
    assert get.calls and get.calls[0].get("User-Agent") == _SYNTHETIC


@pytest.mark.parametrize("module_name", ["run_capex_cycle_mining", "run_accrual_anomaly_mining"],
                         ids=["capex-driver", "accrual-driver"])
def test_builder_chain_sends_contact_and_fails_loudly_when_unset(monkeypatch, module_name):
    """Real builder -> coverage/series -> edgar_data chain; only HTTP is faked.

    Every function between build_fundamentals_map and fetch_company_concept
    reaches it through a module-global lookup inside mining.edgar_data (the
    scripts import only the fetch_all_* wrappers by name, and those call
    fetch_company_concept_series -> fetch_company_concept as globals of
    edgar_data), so patching the edgar_data attribute below is reached."""
    driver = __import__(module_name)
    get = _RecordingGet()
    real_fetch = edgar_data.fetch_company_concept

    def fetch_with_fake_http(*args, **kwargs):
        kwargs["http_get"] = get
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr(edgar_data, "fetch_company_concept", fetch_with_fake_http)
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", _SYNTHETIC)
    driver.build_fundamentals_map(["AAA"], cik_by_ticker={"AAA": 1234}, sleep=lambda s: None, min_interval=0.0)
    assert get.calls and all(h.get("User-Agent") == _SYNTHETIC for h in get.calls)

    get.calls.clear()
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    with pytest.raises(sec_contact.SecContactMissing):
        driver.build_fundamentals_map(["AAA"], cik_by_ticker={"AAA": 1234}, sleep=lambda s: None, min_interval=0.0)
    assert get.calls == []


def _strings(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def _assert_output_has_source_not_contact(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    strings = list(_strings(data))
    assert "env:SEC_EDGAR_USER_AGENT" in strings
    assert not any("research-contact" in s or "example.org" in s for s in strings)


_FAKE_SP500 = ["AAA", "BBB", "CCC"]
_FAKE_COMPANY_TICKERS = {
    "0": {"cik_str": 1, "ticker": "AAA", "title": "Alpha Example Corp"},
    "1": {"cik_str": 2, "ticker": "BBB", "title": "Beta Example Corp"},
    "2": {"cik_str": 3, "ticker": "CCC", "title": "Gamma Example Corp"},
}
_FAKE_HOLDINGS_JSON = json.dumps(
    {"holdings": [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]})


def _fake_capex_series(cik, *, user_agent=None, sleep=None, min_interval=0.0):
    # No usable disclosure: the smallest shape main() accepts without a
    # network call; only the written provenance field is under test.
    return None, ["FakeCapexTag"]


def _fake_fetch_tags(cik, tags):
    return {}


def test_capex_watchlist_output_records_source_not_contact(monkeypatch, tmp_path):
    import build_capex_watchlist as script
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", _SYNTHETIC)
    monkeypatch.setattr(script, "_fetch_sp500_tickers", lambda *a, **k: _FAKE_SP500)
    monkeypatch.setattr(script, "_fetch_company_tickers", lambda *a, **k: _FAKE_COMPANY_TICKERS)
    monkeypatch.setattr(script, "_build_point_in_time_series_for_ticker", _fake_capex_series)
    monkeypatch.setattr(sys, "argv", ["build_capex_watchlist.py", "--out-dir", str(tmp_path),
                                      "--date", "2026-01-02", "--min-interval", "0.000001"])
    assert script.main() == 0
    outputs = sorted(tmp_path.glob("*.json"))
    assert outputs
    for path in outputs:
        _assert_output_has_source_not_contact(path)


def test_retail_coverage_output_records_source_not_contact(monkeypatch, tmp_path):
    import run_retail_inventory_coverage_check as script
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", _SYNTHETIC)
    holdings = tmp_path / "holdings.json"
    holdings.write_text(_FAKE_HOLDINGS_JSON, encoding="utf-8")
    monkeypatch.setattr(script, "build_cik_map", lambda tickers: {t: i + 1 for i, t in enumerate(tickers)})
    monkeypatch.setattr(script, "make_live_fetch_tags_fn", lambda **k: _fake_fetch_tags)
    out = tmp_path / "coverage.json"
    assert script.main(["--holdings", str(holdings), "--output", str(out)]) == 0
    _assert_output_has_source_not_contact(out)


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_ALLOWED_DOMAINS = re.compile(r"@(example\.(org|com|net)|[A-Za-z0-9.-]*\.example|example\.invalid)\b")


def test_no_email_address_in_shipped_non_test_code():
    offenders = []
    files = sorted((_REPO / "scripts").glob("*.py")) + sorted((_REPO / "src").glob("*.py")) \
        + sorted((_REPO / "src" / "mining").rglob("*.py"))
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMAIL.finditer(line):
                if not _ALLOWED_DOMAINS.search(match.group(0)):
                    offenders.append(f"{path.relative_to(_REPO).as_posix()}:{number}")
    assert offenders == []


def test_retail_coverage_check_reraises_missing_contact_instead_of_fetch_error():
    from mining import retail_inventory_coverage

    def fetch_tags(cik, tags):
        raise sec_contact.SecContactMissing("unset")

    with pytest.raises(sec_contact.SecContactMissing):
        retail_inventory_coverage.check_ticker_coverage("AAA", 1234, fetch_tags_fn=fetch_tags)


def test_retail_coverage_check_reraises_missing_contact_on_the_revenue_fetch():
    import pandas as pd
    from mining import edgar_data, retail_inventory_coverage

    def fetch_tags(cik, tags):
        if tags == edgar_data.REVENUE_TAGS:
            raise sec_contact.SecContactMissing("unset")
        return {"inventory": "present"}

    def select(by_tag, tags, kind=None):
        return pd.Series([1.0], index=pd.to_datetime(["2020-12-31"])), "SomeInventoryTag"

    original = retail_inventory_coverage.select_and_build_annual_series
    retail_inventory_coverage.select_and_build_annual_series = select
    try:
        with pytest.raises(sec_contact.SecContactMissing):
            retail_inventory_coverage.check_ticker_coverage("AAA", 1234, fetch_tags_fn=fetch_tags)
    finally:
        retail_inventory_coverage.select_and_build_annual_series = original
