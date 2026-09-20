from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .manifest import record_artifact, utc_now


class DownloadError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, user_agent: str, timeout: int = 60) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET", "HEAD")),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def download(
        self,
        url: str,
        destination: Path,
        artifact_key: str,
        *,
        refresh: bool = False,
        params: dict[str, Any] | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not refresh:
            record_artifact(artifact_key, destination, source_url=url, cache_hit=True)
            return destination

        tmp = destination.with_suffix(destination.suffix + ".part")
        with self.session.get(url, params=params, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            tmp.replace(destination)
            record_artifact(
                artifact_key,
                destination,
                source_url=response.url,
                retrieved_at=utc_now(),
                http_etag=response.headers.get("ETag"),
                http_last_modified=response.headers.get("Last-Modified"),
                content_type=response.headers.get("Content-Type"),
                cache_hit=False,
            )
        return destination

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post_overpass(
        self,
        endpoints: list[str],
        query: str,
        destination: Path,
        artifact_key: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not refresh:
            with destination.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        errors: list[str] = []
        for endpoint in endpoints:
            for attempt in range(2):
                try:
                    response = self.session.post(
                        endpoint,
                        data={"data": query},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    tmp = destination.with_suffix(".json.tmp")
                    with tmp.open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle, ensure_ascii=False)
                        handle.write("\n")
                    tmp.replace(destination)
                    record_artifact(
                        artifact_key,
                        destination,
                        source_url=endpoint,
                        retrieved_at=utc_now(),
                        cache_hit=False,
                    )
                    return payload
                except (requests.RequestException, ValueError) as exc:
                    errors.append(f"{endpoint} attempt {attempt + 1}: {exc}")
                    time.sleep(2 ** attempt)
        raise DownloadError("Overpass request failed: " + " | ".join(errors))

