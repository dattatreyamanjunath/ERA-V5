#!/usr/bin/env python3
"""
verify_dates.py — re-check every date in js/mechanisms.js against arXiv itself.

The assignment's warning was that an AI agent will happily invent a launch date.
So none of the dates on this page are trusted from memory: each arXiv-backed row
is checked against the arXiv API's `published` field, which is the v1 submission
date. Rows whose primary source is not a paper (a Reddit post, a vendor release
note, a model launch) cannot be machine-checked and are listed separately for a
human to eyeball against the URL in `sourceUrl`.

Usage:   python3 scripts/verify_dates.py            # from assignment-8/
Exit 0 = every arXiv date matches. Exit 1 = at least one mismatch.
No third-party dependencies.
"""

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "js" / "mechanisms.js"


def load_rows():
    """Pull (id, date, arxiv, name, sourceUrl) out of mechanisms.js without a JS engine."""
    text = SRC.read_text(encoding="utf-8")
    body = text[text.index("const MECHANISMS = ["):]
    rows = []
    for block in body.split("\n{\n")[1:]:
        def field(key):
            # keys are packed several to a line, so anchor on a delimiter, not ^
            m = re.search(rf"(?:^|[\s,{{]){key}:\s*'((?:[^'\\]|\\.)*)'", block, re.M)
            return m.group(1).replace("\\'", "'") if m else None
        arx = field("arxiv")
        if arx is None and not re.search(r"(?:^|[\s,{])arxiv:\s*null", block, re.M):
            print(f"  ! could not read an arxiv field for block starting: {block[:60]!r}")
        rows.append({
            "id": field("id"),
            "date": field("date"),
            "name": field("name"),
            "arxiv": arx,
            "sourceUrl": field("sourceUrl"),
        })
    return [r for r in rows if r["id"] and r["date"]]


def fetch(ids):
    """Ask arXiv for these ids. Returns {id_without_version: (published_date, title)}."""
    out = {}
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        url = f"{API}?id_list={','.join(chunk)}&max_results={len(chunk)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            feed = ET.fromstring(resp.read())
        for e in feed.findall("a:entry", NS):
            raw = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
            base = re.sub(r"v\d+$", "", raw)
            published = e.find("a:published", NS).text[:10]
            title = " ".join(e.find("a:title", NS).text.split())
            out[base] = (published, title)
        if i + 25 < len(ids):
            time.sleep(3)  # arXiv asks for a 3s gap between API calls
    return out


def main():
    rows = load_rows()
    papers = [r for r in rows if r["arxiv"]]
    others = [r for r in rows if not r["arxiv"]]

    print(f"mechanisms.js: {len(rows)} rows — {len(papers)} arXiv-backed, "
          f"{len(others)} needing manual check\n")
    print("Checking every arXiv date against the API `published` field (= v1 submission).\n")

    live = fetch(sorted({r["arxiv"] for r in papers}))
    failures = []

    print(f"{'STATUS':<8}{'STORED':<12}{'ARXIV v1':<12}{'ID':<12}MECHANISM")
    print("-" * 100)
    for r in sorted(papers, key=lambda r: r["date"]):
        got = live.get(r["arxiv"])
        if not got:
            failures.append((r, "not found on arXiv"))
            print(f"{'MISSING':<8}{r['date']:<12}{'-':<12}{r['arxiv']:<12}{r['name']}")
            continue
        published, title = got
        ok = published == r["date"]
        if not ok:
            failures.append((r, f"arXiv says {published}"))
        print(f"{'OK' if ok else 'FAIL':<8}{r['date']:<12}{published:<12}"
              f"{r['arxiv']:<12}{r['name'][:44]}")
        print(f"{'':<32}{'':<12}└─ {title[:70]}")

    print("\nRows with no paper — verify these by hand against the URL:\n")
    for r in sorted(others, key=lambda r: r["date"]):
        print(f"  {r['date']}  {r['name']}")
        print(f"              {r['sourceUrl']}")

    print()
    if failures:
        print(f"FAILED — {len(failures)} date(s) do not match arXiv:")
        for r, why in failures:
            print(f"  {r['id']}: stored {r['date']}, {why}")
        return 1
    print(f"PASS — all {len(papers)} arXiv-backed dates match the API exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
