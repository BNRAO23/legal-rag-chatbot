"""Authentication, Role-Based Access Control (RBAC), and Daily Quota Management.

Keeps all credentials, passwords, and user records strictly OUT of code by loading
from Streamlit secrets (`.streamlit/secrets.toml` or `st.secrets`).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

# Default path for local secrets when not running under Streamlit
SECRETS_PATHS = [
    Path(".streamlit/secrets.toml"),
    Path("secrets.toml"),
]

USAGE_RECORDS_PATH = Path("usage_records.json")


def _read_toml_secrets(path: Path) -> dict[str, Any]:
    """Read a TOML secrets file using standard library tomllib."""
    if not path.exists():
        return {}
    try:
        import tomllib
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def load_auth_config() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load users and roles configuration from Streamlit secrets or local secrets.toml.

    Credentials and user definitions are NEVER hardcoded in source files.
    """
    raw_secrets: dict[str, Any] = {}

    # 1. Attempt to read from streamlit runtime secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            raw_secrets = dict(st.secrets)
    except Exception:
        pass

    # 2. Fall back to local TOML files if st.secrets is empty
    if not raw_secrets.get("users"):
        custom_path = os.getenv("AUTH_SECRETS_PATH")
        search_paths = [Path(custom_path)] if custom_path else SECRETS_PATHS
        for p in search_paths:
            if p.exists():
                raw_secrets = _read_toml_secrets(p)
                if raw_secrets.get("users"):
                    break

    users = raw_secrets.get("users") or {}
    roles = raw_secrets.get("roles") or {}

    return users, roles


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Verify username and password against configured secrets without hardcoded data.

    Supports either plain `password` or SHA-256 `password_hash` in secrets.
    """
    if not username or not password:
        return None

    users, _ = load_auth_config()
    uname_clean = username.strip().lower()

    # Find matching user case-insensitively
    target_user: dict[str, Any] | None = None
    target_key = ""
    for k, u in users.items():
        if k.lower() == uname_clean:
            target_user = dict(u)
            target_key = k
            break

    if not target_user:
        return None

    stored_plain = target_user.get("password")
    stored_hash = target_user.get("password_hash")

    is_valid = False
    if stored_hash:
        entered_hash = hashlib.sha256(password.strip().encode("utf-8")).hexdigest()
        is_valid = hmac.compare_digest(entered_hash, str(stored_hash).strip())
    elif stored_plain:
        is_valid = hmac.compare_digest(password.strip(), str(stored_plain).strip())

    if not is_valid:
        return None

    role = str(target_user.get("role", "corporate_lawyer"))
    daily_limit = int(target_user.get("daily_limit", 10))

    return {
        "username": target_key,
        "name": str(target_user.get("name", target_key)),
        "role": role,
        "daily_limit": daily_limit,
    }


def get_allowed_documents(role: str) -> list[str]:
    """Return the list of document names permitted for the specified user role."""
    _, roles = load_auth_config()
    role_info = roles.get(role) or {}
    docs = role_info.get("allowed_documents")
    if isinstance(docs, list):
        return [str(d) for d in docs]
    return []


def get_role_description(role: str) -> str:
    """Return human-readable role description."""
    _, roles = load_auth_config()
    role_info = roles.get(role) or {}
    return str(role_info.get("description") or role.replace("_", " ").title())


class UsageTracker:
    """Tracks questions asked per user per day and enforces daily quota limits."""

    def __init__(self, records_path: Path = USAGE_RECORDS_PATH) -> None:
        self.path = records_path

    def _today_str(self) -> str:
        return datetime.date.today().isoformat()

    def _load_data(self) -> dict[str, dict[str, int]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_data(self, data: dict[str, dict[str, int]]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_usage(self, username: str, date_str: str | None = None) -> int:
        """Get the count of questions asked by username on a given date (defaults to today)."""
        dt = date_str or self._today_str()
        data = self._load_data()
        return data.get(dt, {}).get(username.lower(), 0)

    def check_daily_cap(self, username: str, daily_limit: int) -> tuple[bool, int, int, str]:
        """Check if user has reached their daily question limit.

        Returns:
            (is_allowed, current_count, daily_limit, message)
        """
        current_count = self.get_usage(username)
        today = self._today_str()
        if current_count >= daily_limit:
            msg = (
                f"Daily question limit reached for user '{username}'. "
                f"You have used {current_count}/{daily_limit} questions for today ({today}). "
                "Please try again tomorrow or contact an administrator to increase your quota."
            )
            return False, current_count, daily_limit, msg
        return True, current_count, daily_limit, ""

    def increment_usage(self, username: str) -> int:
        """Increment and persist today's question count for user, returning the new count."""
        dt = self._today_str()
        data = self._load_data()
        if dt not in data:
            data[dt] = {}
        u_key = username.lower()
        new_val = data[dt].get(u_key, 0) + 1
        data[dt][u_key] = new_val
        self._save_data(data)
        return new_val

    def reset_usage(self, username: str, date_str: str | None = None) -> None:
        """Reset usage for a user on a given date (useful for automated testing)."""
        dt = date_str or self._today_str()
        data = self._load_data()
        if dt in data and username.lower() in data[dt]:
            del data[dt][username.lower()]
            self._save_data(data)


# Global tracker instance
usage_tracker = UsageTracker()

