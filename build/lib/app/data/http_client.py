from __future__ import annotations

import csv
import json
import time
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


class HttpClient:
    def __init__(self, user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36") -> None:
        self.user_agent = user_agent

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return True
        if not parsed.scheme or not parsed.netloc:
            return False
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = robotparser.RobotFileParser()
        try:
            parser.set_url(robots_url)
            parser.read()
        except (OSError, URLError, HTTPError):
            return True
        return parser.can_fetch(self.user_agent, url)

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
        for attempt in range(3):
            request = Request(full_url, headers=headers)
            try:
                with urlopen(request, timeout=18) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    text = response.read().decode(charset, errors="replace")
                    return HttpResponse(full_url, response.status, text)
            except (HTTPError, URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
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
