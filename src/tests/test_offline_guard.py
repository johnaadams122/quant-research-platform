import os
import pathlib
import socket
import subprocess
import sys

import pytest

_REASON = "network access is disabled in the test suite"
_SRC = pathlib.Path(__file__).resolve().parents[1]


def test_create_connection_is_blocked():
    with pytest.raises(OSError, match=_REASON):
        socket.create_connection(("example.org", 80), timeout=1)


def test_socket_connect_and_connect_ex_are_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match=_REASON):
            s.connect(("example.org", 80))
        with pytest.raises(OSError, match=_REASON):
            s.connect_ex(("example.org", 80))
    finally:
        s.close()


def test_curl_cffi_requests_are_blocked():
    import curl_cffi.requests as curl_requests
    with pytest.raises(OSError, match=_REASON):
        curl_requests.Session().get("https://example.org")


def test_conftest_wires_the_guard_and_imports_no_private_module():
    text = (_SRC / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "offline_guard.install()" in text and "offline_guard.uninstall()" in text
    # every module the conftest imports must resolve in this tree or the environment
    import ast
    import importlib.util
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    for name in sorted(names):
        assert importlib.util.find_spec(name) is not None, name


def _attempt_connection():
    try:
        socket.create_connection(("example.org", 80), timeout=1)
        return "CONNECTED"
    except OSError as exc:
        return str(exc)


# Runs at import, i.e. during collection under the REAL conftest and the plain
# manifest command (the guard must already be active by then). Not run
# when the network is explicitly allowed: the opt-in test below imports this
# module in a child run with the guard off, and that import must not try an
# outside connection.
_OPTED_IN = os.environ.get("QUANT_TESTS_ALLOW_NETWORK") == "1"
_AT_COLLECTION = None if _OPTED_IN else _attempt_connection()


@pytest.fixture(scope="module")
def _module_scoped_attempt():
    return _attempt_connection()


def test_guard_is_active_during_collection():
    if _OPTED_IN:
        pytest.skip("network explicitly allowed for this run")
    assert _AT_COLLECTION == _REASON


def test_importing_this_module_with_the_network_allowed_attempts_no_connection():
    """With QUANT_TESTS_ALLOW_NETWORK=1 the guard is off, so the import-time
    probe above must not run at all. A child imports this module with every
    connection call replaced by a recorder (no guard, no traffic)."""
    code = ("import os, socket, sys\n"
            "os.environ['QUANT_TESTS_ALLOW_NETWORK'] = '1'\n"
            "calls = []\n"
            "def record(*a, **k):\n"
            "    calls.append(a); raise OSError('recorded')\n"
            "socket.create_connection = record\n"
            "socket.socket.connect = record\n"
            "socket.socket.connect_ex = record\n"
            f"sys.path.insert(0, {str(_SRC)!r})\n"
            "import tests.test_offline_guard as t\n"
            "print(len(calls), t._AT_COLLECTION)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=str(_SRC.parent))
    assert out.stdout.split() == ["0", "None"], out.stdout + out.stderr


def test_guard_is_active_in_module_scoped_fixtures(_module_scoped_attempt):
    assert _module_scoped_attempt == _REASON


def test_child_python_process_is_offline_too():
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('example.org', 80), timeout=1); print('CONNECTED')\n"
            "except OSError as exc:\n"
            "    print(exc)\n"
            "import curl_cffi.requests as r\n"
            "try:\n"
            "    r.Session().get('https://example.org'); print('CONNECTED')\n"
            "except OSError as exc:\n"
            "    print(exc)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.splitlines() == [_REASON, _REASON], out.stdout + out.stderr


def test_spawned_worker_process_is_offline_too():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(_attempt_connection).result(timeout=120) == _REASON


def test_curl_cffi_async_requests_are_blocked():
    import asyncio
    import curl_cffi.requests as curl_requests

    async def attempt():
        async with curl_requests.AsyncSession() as session:
            await session.get("https://example.org")

    with pytest.raises(OSError, match=_REASON):
        asyncio.run(attempt())


def test_asyncio_open_connection_is_blocked():
    import asyncio
    with pytest.raises(OSError, match=_REASON):
        asyncio.run(asyncio.open_connection("example.org", 80))


def test_child_bootstrap_is_removed_and_pythonpath_restored(monkeypatch):
    from tests import offline_guard
    monkeypatch.setenv("PYTHONPATH", "sentinel-path")
    folder = offline_guard.install_for_children()
    assert os.path.isdir(folder)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == folder
    offline_guard.uninstall_for_children()
    assert not os.path.exists(folder)
    assert os.environ["PYTHONPATH"] == "sentinel-path"


def test_a_second_copy_of_the_guard_neither_saves_nor_restores_the_guard():
    import importlib.util
    spec = importlib.util.spec_from_file_location("offline_guard_second_copy",
                                                  str(_SRC / "tests" / "offline_guard.py"))
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    copy.install()
    assert copy._saved == {}
    copy.uninstall()
    with pytest.raises(OSError, match=_REASON):
        socket.create_connection(("example.org", 80), timeout=1)


def test_loopback_plumbing_still_works():
    import asyncio
    a, b = socket.socketpair()
    a.close()
    b.close()
    assert asyncio.run(asyncio.sleep(0, result="ran")) == "ran"


def test_asyncio_refusal_comes_from_the_guard_not_loop_setup():
    import asyncio
    import traceback

    async def attempt():
        await asyncio.open_connection("192.0.2.1", 80)

    with pytest.raises(OSError, match=_REASON) as info:
        asyncio.run(attempt())
    frames = [f.name for f in traceback.extract_tb(info.value.__traceback__)]
    assert "_make_self_pipe" not in frames


def test_uninstall_restores_the_original_socket_methods():
    import importlib.util
    from tests import offline_guard
    offline_guard.uninstall()
    try:
        assert not getattr(socket.socket.connect, "_offline_guard", False)
        assert "connect" not in socket.socket.__dict__
        spec = importlib.util.spec_from_file_location("offline_guard_probe",
                                                      str(_SRC / "tests" / "offline_guard.py"))
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        assert probe._saved == {}
    finally:
        offline_guard.install()
    assert getattr(socket.socket.connect, "_offline_guard", False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_is_refused_outside_socketpair(host):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(family, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match=_REASON):
            s.connect((host, 9))
    finally:
        s.close()
    with pytest.raises(OSError, match=_REASON):
        socket.create_connection((host, 9), timeout=1)


def test_the_explicit_opt_in_variable_turns_the_guard_off_for_the_whole_run():
    """QUANT_TESTS_ALLOW_NETWORK=1 is the documented way to run tests that need
    live data; without it the guard is always on. With the guard off, the
    child run's probe makes one real TCP connection attempt to 127.0.0.1
    port 9: loopback only, so no traffic leaves the machine."""
    probe = "src/tests/test_offline_guard.py::test_loopback_is_refused_outside_socketpair[127.0.0.1]"
    env = dict(os.environ, QUANT_TESTS_ALLOW_NETWORK="1")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", probe],
                         capture_output=True, text=True, timeout=300, cwd=str(_SRC.parent), env=env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert f"Input: '{_REASON}'" not in out.stdout  # the raised error is the OS's, not the guard's
