from __future__ import annotations

import csv
import json
import os
import time
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib import robotparser
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


def _robots_can_fetch(rules: str, user_agent: str, url: str) -> bool:
    """Evaluate robots rules using RFC 9309 longest-match precedence.

    ``urllib.robotparser`` uses first-match behaviour for some equal/overlapping
    rules.  Modern robots semantics select the most specific path and prefer
    Allow when specificity ties, which matters for feeds exposed below a site-wide
    Disallow rule.
    """
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    directives: list[tuple[bool, str]] = []

    def finish() -> None:
        nonlocal agents, directives
        if agents:
            groups.append((agents, directives))
        agents, directives = [], []

    for raw in str(rules or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        lowered = name.lower()
        if lowered == "user-agent":
            if directives:
                finish()
            agents.append(value.lower())
        elif lowered in {"allow", "disallow"} and agents:
            if value or lowered == "allow":
                directives.append((lowered == "allow", value))
    finish()

    requested_agent = str(user_agent or "").lower()
    selected: list[tuple[bool, str]] = []
    best_agent_specificity = -1
    for group_agents, group_rules in groups:
        matches = [
            0 if token == "*" else len(token)
            for token in group_agents
            if token == "*" or token in requested_agent
        ]
        if not matches:
            continue
        specificity = max(matches)
        if specificity > best_agent_specificity:
            selected = list(group_rules)
            best_agent_specificity = specificity
        elif specificity == best_agent_specificity:
            selected.extend(group_rules)

    target = urlparse(url)
    path = target.path or "/"
    if target.query:
        path += "?" + target.query
    matches: list[tuple[int, bool]] = []
    for allowed, pattern in selected:
        if not pattern and not allowed:
            continue
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        expression = "^" + re.escape(body).replace(r"\*", ".*")
        if anchored:
            expression += "$"
        if re.search(expression, path):
            matches.append((len(body.replace("*", "")), allowed))
    if not matches:
        return True
    longest = max(item[0] for item in matches)
    return any(allowed for length, allowed in matches if length == longest)


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
