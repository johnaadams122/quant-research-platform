"""Session-wide network block for the test suite (also usable as a pytest plugin).

Installed from conftest.pytest_configure, before collection, so module-level
code and fixtures of every scope run offline. Python sockets, asyncio event
loops and curl_cffi (which yfinance uses) refuse every connection with one
clear message. The single exception is Python's own socket pair: on Windows
socket.socketpair() connects two sockets over 127.0.0.1, and every asyncio
event loop builds its wake-up pipe that way, so a numeric loopback address is
allowed only while socket.socketpair() itself is running. DNS lookups are not
blocked; connections are. Setting QUANT_TESTS_ALLOW_NETWORK=1 is the one
explicit way to run tests that need live data: the guard then stays off, in
child processes too. It is never set by default.

Scope, stated plainly: the guard covers socket connect/connect_ex/
create_connection (and through them http.client, urllib, urllib3/requests and
ssl), asyncio create_connection/sock_connect and curl_cffi requests, in this
process and in Python child processes that inherit PYTHONPATH and run normal
site initialization (see install_for_children). It does NOT block DNS lookups
or UDP sendto, and it does not reach a child started with -I, -E or -S or a
non-Python executable. No test in the suite starts such a child.
"""
import ipaddress
import os
import shutil
import socket
import threading

REASON = "network access is disabled in the test suite"
ALLOW_ENV = "QUANT_TESTS_ALLOW_NETWORK"  # set to 1 to run tests that need live data; unset by default


def _allowed_by_owner():
    return os.environ.get(ALLOW_ENV) == "1"
_saved = {}
_child_bootstraps = []  # (folder, previous PYTHONPATH or None), newest last
_inside_socketpair = threading.local()


def _is_loopback(host):
    """True only for a numeric loopback address; names such as "localhost"
    are refused because their resolution is not checked here."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _allowed(host):
    return getattr(_inside_socketpair, "active", False) and _is_loopback(host)


def _host_of(address):
    if isinstance(address, (tuple, list)) and address:
        return address[0]
    return address


def _guard(original, host_from_call):
    """Wrap `original` so it runs only inside socket.socketpair() on loopback."""
    def wrapper(*args, **kwargs):
        if _allowed(host_from_call(args, kwargs)):
            return original(*args, **kwargs)
        raise OSError(REASON)
    wrapper._offline_guard = True
    return wrapper


def _refuse(*args, **kwargs):
    raise OSError(REASON)


async def _refuse_async(*args, **kwargs):
    raise OSError(REASON)


_refuse._offline_guard = True
_refuse_async._offline_guard = True


def _socket_address(args, kwargs):            # socket.connect(self, address)
    return _host_of(args[1] if len(args) > 1 else kwargs.get("address"))


def _create_connection_address(args, kwargs):  # socket.create_connection(address, ...)
    return _host_of(args[0] if args else kwargs.get("address"))


def _loop_sock_connect_address(args, kwargs):  # loop.sock_connect(self, sock, address)
    return _host_of(args[2] if len(args) > 2 else kwargs.get("address"))


def _loop_create_connection_host(args, kwargs):  # loop.create_connection(self, factory, host, port, ...)
    return args[2] if len(args) > 2 else kwargs.get("host")


def _guard_socketpair(original, _unused):
    """Allow numeric loopback connects only while socket.socketpair() runs."""
    def wrapper(*args, **kwargs):
        previous = getattr(_inside_socketpair, "active", False)
        _inside_socketpair.active = True
        try:
            return original(*args, **kwargs)
        finally:
            _inside_socketpair.active = previous
    wrapper._offline_guard = True
    return wrapper


def _targets():
    import asyncio.base_events
    import asyncio.proactor_events
    import asyncio.selector_events
    import curl_cffi.requests as curl_requests
    return [
        (socket.socket, "connect", _guard, _socket_address),
        (socket.socket, "connect_ex", _guard, _socket_address),
        (socket, "create_connection", _guard, _create_connection_address),
        (socket, "socketpair", _guard_socketpair, None),
        (asyncio.base_events.BaseEventLoop, "create_connection", _guard, _loop_create_connection_host),
        (asyncio.proactor_events.BaseProactorEventLoop, "sock_connect", _guard, _loop_sock_connect_address),
        (asyncio.selector_events.BaseSelectorEventLoop, "sock_connect", _guard, _loop_sock_connect_address),
        (curl_requests.Session, "request", None, _refuse),
        (curl_requests.AsyncSession, "request", None, _refuse_async),
    ]


def install():
    """Install the guard. A no-op when it is already installed, including by a
    second copy of this module (the sitecustomize bootstrap imports it as
    `offline_guard`, conftest as `tests.offline_guard`): only the copy that
    replaced the originals saves and later restores them."""
    if _allowed_by_owner() or _saved or getattr(socket.socket.connect, "_offline_guard", False):
        return
    for owner, name, make, how in _targets():
        own = name in owner.__dict__
        original = owner.__dict__[name] if own else getattr(owner, name)
        _saved[(owner, name)] = (original, own)
        setattr(owner, name, make(original, how) if make else how)


def uninstall():
    for (owner, name), (original, own) in _saved.items():
        if own:
            setattr(owner, name, original)
        else:
            delattr(owner, name)  # the original was inherited; drop the override
    _saved.clear()


def install_for_children():
    """Make every child Python process started from the test session (subprocess
    or a multiprocessing spawn worker) install the same guard at start-up: a
    temporary folder holding a sitecustomize.py is put first on PYTHONPATH,
    which children inherit through the environment. Returns the folder;
    uninstall_for_children() removes it and restores PYTHONPATH."""
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    folder = tempfile.mkdtemp(prefix="offline-guard-")
    with open(os.path.join(folder, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write("import sys\n"
                 f"sys.path.insert(0, {here!r})\n"
                 "import offline_guard\n"
                 "offline_guard.install()\n")
    previous = os.environ.get("PYTHONPATH")
    _child_bootstraps.append((folder, previous))
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (folder, here, previous) if p)
    return folder


def uninstall_for_children():
    """Undo the most recent install_for_children()."""
    if not _child_bootstraps:
        return
    folder, previous = _child_bootstraps.pop()
    if previous is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = previous
    shutil.rmtree(folder, ignore_errors=True)


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    uninstall()
