#!/usr/bin/env python3
"""
Clean legacy PubMed JSON records and request one-sentence clinical outcomes
from a local OpenAI-compatible Llama 3.3 70B server.

Usage:
    python pubmed_summarizer.py --input pubmed.json --output summaries.json

The input may be a JSON array or newline-delimited JSON (JSONL).
Requires: pip install requests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_ENDPOINT = "http://192.168.1.100:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pubmed_summarizer")


def load_records(path: Path) -> list[Any]:
    """Read a JSON array, or fall back to one-JSON-object-per-line JSONL."""
    raw = path.read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Support a common wrapper shape.
            for key in ("records", "articles", "results", "data"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            return [parsed]
        raise ValueError("Top-level JSON must be an array or object.")
    except json.JSONDecodeError as array_error:
        records = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d", line_number)
        if records:
            return records
        raise ValueError(f"Input is neither valid JSON nor usable JSONL: {array_error}") from array_error


def clean_text(value: Any) -> str:
    """Convert acceptable text to a trimmed string; null/invalid values become empty."""
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def first_nonempty(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def normalize_record(item: Any, index: int) -> dict[str, Any] | None:
    """Normalize aliases and reject entries that cannot support a summary."""
    if not isinstance(item, dict):
        logger.warning("Skipping record %d: expected an object, got %s", index, type(item).__name__)
        return None

    record_id = first_nonempty(item, ("id", "pmid", "PMID")) or f"record-{index}"
    title = first_nonempty(item, ("ArticleTitle", "title", "article_title"))
    abstract = first_nonempty(item, ("abstract_text", "Abstract", "abstract", "abstractText"))

    authors_value = item.get("authorList", item.get("authors", []))
    if isinstance(authors_value, list):
        authors = [clean_text(author) for author in authors_value]
        authors = [author for author in authors if author]
    elif isinstance(authors_value, str):
        authors = [authors_value.strip()] if authors_value.strip() else []
    else:
        authors = []

    if not abstract:
        logger.warning("Skipping %s: abstract is missing or null", record_id)
        return None

    return {
        "id": record_id,
        "title": title or "Untitled study",
        "abstract": abstract,
        "authors": authors,
    }


def make_session() -> requests.Session:
    """Retry transient HTTP failures; connection/read timeouts are handled explicitly."""
    session = requests.Session()
    retry = Retry(
        total=0,  # application-level retry below keeps behavior explicit
        connect=0,
        read=0,
        status=0,
        allowed_methods=frozenset({"POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def summarize(
    session: requests.Session,
    record: dict[str, Any],
    endpoint: str,
    model: str,
    timeout_seconds: int,
    api_key: str | None,
) -> str:
    prompt = (
        "Summarize the clinical outcome of this study in one sentence. "
        "Use only information supported by the abstract; do not infer missing results. "
        "If the abstract does not report a clinical outcome, say so.\n\n"
        f"Title: {record['title']}\n"
        f"Abstract: {record['abstract']}"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful biomedical summarization assistant. Be concise and factual.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 120,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=(5, timeout_seconds),  # connect timeout, read timeout
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("Server response has no choices.")
            message = choices[0].get("message", {})
            result = clean_text(message.get("content"))
            if not result:
                raise ValueError("Server returned an empty summary.")
            return result

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Local inference failed after {MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.25)
            logger.warning(
                "Request failed (%s), retrying in %.1fs (%d/%d)",
                type(exc).__name__, delay, attempt, MAX_ATTEMPTS,
            )
            time.sleep(delay)
        except requests.HTTPError as exc:
            # Retry only server-side errors; client errors usually require fixing the request.
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status >= 500 and attempt < MAX_ATTEMPTS:
                delay = min(2 ** (attempt - 1), 8)
                logger.warning("Server returned HTTP %s; retrying in %ds", status, delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP request failed: {exc}") from exc

    raise RuntimeError("Unexpected retry-loop exit.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output", type=Path, default=Path("summaries.json"))
    parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-attempt read timeout in seconds (default: 30)",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        raw_records = load_records(args.input)
    except (OSError, ValueError) as exc:
        logger.error("Could not load input: %s", exc)
        return 2

    session = make_session()
    api_key = os.getenv("LLM_API_KEY")
    output: list[dict[str, Any]] = []
    valid_count = 0

    for index, item in enumerate(raw_records, start=1):
        record = normalize_record(item, index)
        if record is None:
            continue
        valid_count += 1
        try:
            outcome = summarize(
                session, record, args.endpoint, args.model, args.timeout, api_key
            )
            output.append({**record, "clinical_outcome": outcome, "error": None})
            logger.info("Summarized %s", record["id"])
        except (RuntimeError, ValueError) as exc:
            logger.error("Could not summarize %s: %s", record["id"], exc)
            output.append({**record, "clinical_outcome": None, "error": str(exc)})

    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Finished: %d usable records, %d output rows -> %s",
                valid_count, len(output), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
