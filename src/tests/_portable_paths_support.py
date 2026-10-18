"""Shared assertion helper: a committed results JSON must not name the
workstation it was produced on.

A drive-letter path, a POSIX-absolute path, the current user's home directory
and the pytest tmp directory all count as a leak. Repo-relative paths (and the
`<external>/<name>` placeholder for an input that lives outside the repo) are
fine.
"""
import os
import re

DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]")


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from strings_in(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from strings_in(inner)


def workstation_leaks(payload, tmp_path) -> list:
    """Every string in ``payload`` that names a local absolute location."""
    home = os.path.expanduser("~").replace("\\", "/").lower()
    tmp = str(tmp_path).replace("\\", "/").lower()
    leaks = []
    for text in strings_in(payload):
        norm = text.replace("\\", "/").lower()
        if (DRIVE_PATH.search(text) or norm.startswith("/") or tmp in norm
                or (len(home) > 3 and home in norm)):
            leaks.append(text)
    return leaks
