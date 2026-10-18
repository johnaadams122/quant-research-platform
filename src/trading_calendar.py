"""Vendored, SHA-pinned NYSE session calendar (fails closed past its coverage).

The calendar file is committed and its SHA-256 is pinned here; it is
deliberately not recomputed at run time, which would defeat the pin.
Paths derive from this file's location, never from the working directory.
"""
import hashlib
import json
import os
import datetime as dt

NYSE_CAL_REL = "data/live/nyse_calendar_2026_2028.json"
NYSE_CAL_SHA256 = "3127f0fd32be80fbc6c2919c85ce0c3d718ff7e9bbe9b0dd41a495062a97007e"


def _repo_root_of_module():
    # Derived from __file__, NEVER from os.getcwd(): callers may start with
    # an arbitrary working directory.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_calendar_path(repo_root=None):
    return os.path.join(repo_root or _repo_root_of_module(),
                        *NYSE_CAL_REL.split("/"))


def load_default_sessions(repo_root=None):
    return load_sessions(default_calendar_path(repo_root), NYSE_CAL_SHA256)


def _sha256(path):
    # Hash with CRLF normalised to LF so a Windows working copy and an LF
    # clone of the same committed file give the same digest.
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def load_sessions(path, expected_sha256):
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"calendar sha mismatch: {actual} != {expected_sha256}")
    with open(path, "r") as f:
        raw = json.load(f)["sessions"]
    return [dt.date.fromisoformat(s) for s in raw]


def is_session(d, sessions):
    return d in set(sessions)


def assert_covered(d, sessions):
    if d > sessions[-1]:
        raise ValueError(f"date {d} past calendar coverage (ends {sessions[-1]})")
