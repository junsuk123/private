from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, url2pathname, urlopen


class DataCollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    text: str


class HttpClient:
    def __init__(self, user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36") -> None:
        self.user_agent = user_agent
        self.timeout_seconds = _positive_float(os.getenv("HTTP_CLIENT_TIMEOUT_SECONDS"), default=8.0)
        self.robots_timeout_seconds = _positive_float(os.getenv("HTTP_CLIENT_ROBOTS_TIMEOUT_SECONDS"), default=3.0)
        self.max_attempts = _positive_int(os.getenv("HTTP_CLIENT_ATTEMPTS"), default=2)

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return True
        if not parsed.scheme or not parsed.netloc:
            return False
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            request = Request(robots_url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.robots_timeout_seconds) as response:
                text = response.read().decode("utf-8", errors="replace")
        except (OSError, URLError, HTTPError, TimeoutError):
            return True
        return _robots_can_fetch(text, self.user_agent, url)

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> HttpResponse:
        full_url = _with_query(url, params)
        parsed = urlparse(full_url)
        if parsed.scheme == "file":
            path = Path(url2pathname(unquote(parsed.path)))
            return HttpResponse(full_url, 200, path.read_text(encoding="utf-8"))

        if not self.can_fetch(full_url):
            raise DataCollectionError(f"robots.txt disallows fetching {full_url}")

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/json,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            request = Request(full_url, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    text = response.read().decode(charset, errors="replace")
                    return HttpResponse(full_url, response.status, text)
            except (HTTPError, URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                break
        raise DataCollectionError(f"failed to fetch {full_url}: {last_exc}") from last_exc

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self.get_text(url, params)
        return json.loads(response.text)

    def get_csv_rows(self, url: str, params: dict[str, Any] | None = None) -> list[dict[str, str]]:
        response = self.get_text(url, params)
        return list(csv.DictReader(StringIO(response.text)))


def _with_query(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def _robots_can_fetch(text: str, user_agent: str, url: str) -> bool:
    """Evaluate robots rules using RFC 9309's most-specific-match precedence.

    ``urllib.robotparser`` uses the first matching rule.  That incorrectly blocks
    sites such as the Bank of Korea whose file declares ``Disallow: /`` followed
    by the more specific ``Allow: /portal/``.
    """
    groups: list[tuple[tuple[str, ...], tuple[tuple[bool, str], ...]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []

    def flush() -> None:
        nonlocal agents, rules
        if agents:
            groups.append((tuple(agents), tuple(rules)))
        agents, rules = [], []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, value = (part.strip() for part in line.split(":", 1))
        field = field.lower()
        if field == "user-agent":
            if rules:
                flush()
            agents.append(value.lower())
        elif field in {"allow", "disallow"} and agents:
            if value or field == "allow":
                rules.append((field == "allow", value))
    flush()

    product = user_agent.split("/", 1)[0].strip().lower()
    exact = [rules for names, rules in groups if any(name != "*" and name in product for name in names)]
    applicable = exact or [rules for names, rules in groups if "*" in names]
    target = urlparse(url).path or "/"
    if urlparse(url).query:
        target = f"{target}?{urlparse(url).query}"
    matches: list[tuple[int, bool]] = []
    for group_rules in applicable:
        for allowed, pattern in group_rules:
            if not pattern and not allowed:
                continue
            anchored = pattern.endswith("$")
            raw_pattern = pattern[:-1] if anchored else pattern
            expression = re.escape(raw_pattern).replace(r"\*", ".*")
            expression = f"^{expression}{'$' if anchored else ''}"
            if re.search(expression, target):
                specificity = len(raw_pattern.replace("*", ""))
                matches.append((specificity, allowed))
    if not matches:
        return True
    best_specificity = max(item[0] for item in matches)
    return any(allowed for specificity, allowed in matches if specificity == best_specificity)


def _positive_float(value: str | None, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0 else default
