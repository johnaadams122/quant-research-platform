import os
import sys

import pytest

# Each test module inserts src/ on sys.path itself; do it here too so
# conftest-level fixtures can import the modules under test irrespective of
# which files got collected.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests import offline_guard  # noqa: E402


def pytest_configure(config):
    offline_guard.install()
    offline_guard.install_for_children()
    config.addinivalue_line(
        "markers",
        "slow: real-spawn / real-corpus tests; run at commit/CI gates, skip in "
        "the fast unit cycle with `-m 'not slow'`.")


def pytest_unconfigure(config):
    offline_guard.uninstall_for_children()
    offline_guard.uninstall()


@pytest.fixture(autouse=True)
def _isolate_abi_memo():
    """Keep `compute_abi_digest`'s process-wide memo from leaking across tests.

    `sweep_cache.compute_abi_digest` memoizes in `_ABI_MEMO` for the life of the
    process, and it hashes the SOURCE of every function in the sweep manifest.
    So any test that monkeypatches a manifest member and then triggers a digest
    computation memoizes a digest of the patched source -- which then survives
    monkeypatch teardown and is served to every later test in the session.

    For example, a test that patches `strategy_sweep.grid_points` to a lambda
    and calls `main()` recomputes the digest through `sweep_fingerprint`; if
    the memo is empty at that moment, it is filled with a digest of the
    lambda's source, and every later reader in the process sees the wrong
    digest. An ABI guard that silently reports the wrong digest is worse than
    no guard -- it can report a break that does not exist or MASK a real one.

    Clearing on both sides means every test computes from unpatched source and
    leaves nothing behind. Recomputation is a few `inspect.getsource` calls, so
    the cost is negligible; the memo exists for long sweep runs, not tests.
    """
    try:
        import sweep_cache as sc
    except Exception:          # module not importable for this selection
        yield
        return
    sc._ABI_MEMO.clear()
    try:
        yield
    finally:
        sc._ABI_MEMO.clear()


def _calendar_memos():
    """Every module-level session-calendar memo (a ``_CAL_MEMO`` dict) of an
    already-imported module. Read from each module's own namespace, so no
    module-level ``__getattr__`` hook runs."""
    memos = []
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        memo = namespace.get("_CAL_MEMO") if isinstance(namespace, dict) else None
        if isinstance(memo, dict):
            memos.append(memo)
    return memos


@pytest.fixture(autouse=True)
def _isolate_calendar_memos():
    """Keep any imported module's module-level session-calendar memo
    (``_CAL_MEMO``) from leaking across tests, the same shape as
    `_isolate_abi_memo` above. It is a no-op when no imported module carries
    one. A test that populates such a memo with a vendored session list would
    hide a later test's monkeypatched failing loader; run the other way, a
    cached None drops every calendar-path test onto the weekday fallback where
    it passes vacuously."""
    for memo in _calendar_memos():
        memo.clear()
    try:
        yield
    finally:
        for memo in _calendar_memos():
            memo.clear()
