#!/usr/bin/env python3
"""Generate tags for blog posts using an LLM.

This script scans Quarto markdown files (.qmd) in selected directories,
optionally sends their text to a language model and inserts the returned
tags into the YAML front matter under the ``categories`` field.

Environment variables:
    OPENAI_API_KEY: API key for the OpenAI service. If not set, the script
        can run in ``--dry-run`` mode, which generates deterministic tags
        locally without contacting an external API.

Usage:
    python scripts/generate_tags_with_llm.py [--dry-run] [--model MODEL]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import yaml

MAX_TAGS = 8
TARGET_DIRS = [
    Path("ds/posts"),
    Path("bi/posts"),
    Path("dp/posts"),
    Path("challenges/posts"),  # may not exist but kept for consistency
]


def normalise_tag(tag: str) -> str:
    """Convert tag into kebab-case"""
    tag = tag.strip().lower()
    tag = re.sub(r"[^a-z0-9]+", "-", tag)
    return tag.strip("-")


def read_front_matter(text: str) -> Tuple[dict, str]:
    """Return YAML front matter and body from a qmd file."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end]
    body = text[end + 4 :]
    data = yaml.safe_load(fm_text) or {}
    return data, body


def write_front_matter(data: dict, body: str) -> str:
    fm_text = yaml.safe_dump(data, sort_keys=False).strip()
    return f"---\n{fm_text}\n---\n{body}"


def call_openai_api(api_key: str, text: str, model: str) -> List[str]:
    """Call OpenAI API to generate tags for the given text."""
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("openai package is required to call the API") from exc

    client = OpenAI(api_key=api_key)
    prompt = (
        "Generate up to {max_tags} short tags for the following article. "
        "Return only a comma-separated list of tags without numbering or "
        "additional text.\n\n""Article:\n{text}"
    ).format(max_tags=MAX_TAGS, text=text[:4000])
    response = client.responses.create(model=model, input=prompt)
    tags_text = response.output_text
    raw_tags = re.split(r"[\n,]", tags_text)
    return [normalise_tag(t) for t in raw_tags if t.strip()][:MAX_TAGS]


def generate_dummy_tags(text: str) -> List[str]:
    """Fallback tag generator used in dry-run mode."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= MAX_TAGS:
            break
    return [normalise_tag(w) for w in seen]


def process_file(path: Path, api_key: str | None, model: str, dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    front, body = read_front_matter(text)
    article_text = body.strip()

    if dry_run or not api_key:
        tags = generate_dummy_tags(article_text)
    else:
        tags = call_openai_api(api_key, article_text, model)

    if tags:
        front["categories"] = tags
        new_text = write_front_matter(front, body)
        if dry_run:
            print(f"{path}: {tags}")
        else:
            path.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tags for Quarto posts")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; generate deterministic tags locally",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Use --dry-run to run without calling the API."
        )

    for base in TARGET_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.qmd"):
            process_file(path, api_key, args.model, args.dry_run)


if __name__ == "__main__":
    main()
