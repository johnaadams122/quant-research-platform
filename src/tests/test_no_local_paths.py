import hashlib
import os
import pathlib
import re
import sys

import pytest

_SRC = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_REPO = _SRC.parent
# Files kept only in the private source tree (never part of the public release)
# are excluded by the last 16 hex characters of the SHA-256 of their
# repository-relative path (or of a parent folder path ending in "/"), so the
# exclusion list does not publish their names. In the public tree none of them
# exist and nothing is excluded.
_EXCLUDED_PATH_DIGESTS = frozenset({
    "3d2d5d7cf0e2e3b5", "9733eb8b7f6dd78d", "e5379e826949eb54", "b2ce97fed760d960",
})
_DRIVE_PATH = re.compile(r"(?i)\b[a-z]:[\\/]+(users|documents and settings)[\\/]")
_HOME_PATH = re.compile(r"(?i)[\\/](home|Users)[\\/][a-z0-9._-]+[\\/]")
_ENV_NAMES = re.compile(r"(?i)\b((mini|ana)" + "conda" + r"|mini" + r"forge|mamba" + r"forge)\d*\b")  # built from parts so this file does not match itself
_TILDE_HOME = re.compile("~" + r"[\\/]")
_PROFILE_VARS = re.compile(r"(?i)%(user" + r"profile|homepath|local" + r"appdata|appdata)%"
                           r"|\$env:(user" + r"profile|homepath|local" + r"appdata|appdata)\b")
_SYNCED_FOLDERS = re.compile(r"(?i)\bone" + r"drive\b")
_PATTERNS = (_DRIVE_PATH, _HOME_PATH, _ENV_NAMES, _TILDE_HOME, _PROFILE_VARS, _SYNCED_FOLDERS)


def _path_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[-16:]


def _is_private_only(path):
    rel = path.relative_to(_REPO).as_posix()
    candidates = [rel] + [rel[:i + 1] for i, ch in enumerate(rel) if ch == "/"]
    return any(_path_digest(c) in _EXCLUDED_PATH_DIGESTS for c in candidates)


def _shipped_python_files():
    files = list((_REPO / "scripts").glob("*.py")) + list(_SRC.rglob("*.py"))
    for path in sorted(files):
        if not _is_private_only(path):
            yield path


def test_no_absolute_user_paths_or_named_interpreters():
    offenders = []
    for path in _shipped_python_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in _PATTERNS):
                offenders.append(f"{path.relative_to(_REPO).as_posix()}:{number}")
    assert offenders == []


@pytest.mark.parametrize("sample", [
    "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "repo",
    "/" + "home/someone/repo/",
    "~" + "/repo/data",
    "%" + "USERPROFILE" + "%" + "/repo",
    "$env:" + "LOCALAPPDATA",
    "One" + "Drive/Documents",
    "mini" + "conda3/python.exe",
    "Mini" + "forge3/python.exe",
    "$env:" + "HOMEPATH",
], ids=["drive-users", "home", "tilde", "percent-var", "ps-env", "synced-folder", "conda", "forge", "ps-homepath"])
def test_each_pattern_catches_its_form(sample):
    assert any(pattern.search(sample) for pattern in _PATTERNS)
