"""Minimal Tushare Pro client initialization for MIRAGE data jobs."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from typing import Any


DEFAULT_TUSHARE_HTTP_URL = "https://ts.gyzcloud.top/api"


class TushareConfigurationError(RuntimeError):
    """Raised when required Tushare configuration is missing."""


def create_pro_client(
    environ: Mapping[str, str] | None = None,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Build a Tushare Pro client using environment-only credentials.

    The import is intentionally delayed until after the token check so a
    missing credential fails before SDK initialization or network activity.
    """

    env = os.environ if environ is None else environ
    token = env.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise TushareConfigurationError("TUSHARE_TOKEN is required")
    http_url = env.get("TUSHARE_HTTP_URL", DEFAULT_TUSHARE_HTTP_URL).strip()
    if not http_url:
        http_url = DEFAULT_TUSHARE_HTTP_URL

    ts = import_module("tushare")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = http_url
    return pro
