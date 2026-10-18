"""SEC EDGAR fair-access contact, read from a setting with no default.

SEC asks automated clients to send a User-Agent naming an organisation and a
contact email. This project ships no contact value: set SEC_EDGAR_USER_AGENT,
for example an organisation name followed by a contact email address, before any
EDGAR fetch.
The value is read only immediately before a request, never at import time.
"""
import os

ENV_VAR = "SEC_EDGAR_USER_AGENT"


class SecContactMissing(RuntimeError):
    """SEC_EDGAR_USER_AGENT is unset or does not contain a contact address."""


def sec_contact_user_agent(environ=None) -> str:
    env = os.environ if environ is None else environ
    value = (env.get(ENV_VAR) or "").strip()
    if not value or "@" not in value:
        raise SecContactMissing(
            f"set {ENV_VAR} to an organisation name and contact email before fetching from SEC EDGAR")
    return value
