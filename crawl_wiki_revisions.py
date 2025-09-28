#!/usr/bin/env python3
"""
Wikipedia revision crawler (plain-text dumps for current + offsets 3..147).

Usage:
  python crawl_wiki_revisions.py \
      --titles "New York City" "San Francisco" \
      --out-dir ./wikipedia_cities \
      --max-offset 147 --step 3

Or provide a file with one title per line:
  python crawl_wiki_revisions.py --titles-file titles.txt --out-dir ./out
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

WIKI_API = "https://en.wikipedia.org/w/api.php"

def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.25,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    s.headers.update({
        "User-Agent": "WikipediaRevisionCrawler/1.0 (academic; contact: you@example.com)"
    })
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def sanitize_title(title: str) -> str:
    # Create a safe folder name from a page title
    safe = title.replace("/", "-").replace("\\", "-").replace(":", " -").strip()
    return "_".join(safe.split())

def compute_offsets(step: int, max_offset: int) -> List[int]:
    offsets = [0]  # current revision
    k = step
    while k <= max_offset:
        offsets.append(k)
        k += step
    return offsets

def fetch_revision_list(session: requests.Session, title: str, needed_count: int) -> List[Dict]:
    """
    Get a list of revisions starting from the current (index 0) going older.
    Returns at most needed_count revisions, each as {'revid': int, 'timestamp': '...'}.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp",
        "rvslots": "main",
        "rvlimit": str(needed_count),
        "rvdir": "older",        # start at current, then go older
        "redirects": "1",
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        raise ValueError(f"Page not found or missing: {title}")

    revs = pages[0].get("revisions", [])
    # revs[0] should be the current (most recent) when rvdir=older
    return revs

# def fetch_plaintext_for_revid(session: requests.Session, revid: int) -> str:
#     """
#     Use action=parse to render HTML for a specific revision (oldid) and strip to plain text.
#     """
#     params = {
#         "action": "parse",
#         "format": "json",
#         "formatversion": "2",
#         "oldid": str(revid),
#         "prop": "text",  # get rendered HTML
#     }
#     r = session.get(WIKI_API, params=params, timeout=30)
#     r.raise_for_status()
#     data = r.json()
#     if "error" in data:
#         raise RuntimeError(f"parse error for oldid={revid}: {data['error']}")
#     html = data.get("parse", {}).get("text", "")
#     # Strip HTML to plain text
#     soup = BeautifulSoup(html, "html.parser")
#     # Remove tables, references, and navboxes if desired (optional cleanup)
#     for el in soup.select("table, sup.reference, div.navbox, div.hatnote"):
#         el.decompose()
#     text = soup.get_text(separator=" ", strip=True)
#     return text

# def fetch_plaintext_for_revid(session, revid: int) -> str:
#     """
#     Render a specific revision (oldid) and return ONLY:
#       - section headings (h2/h3/h4) that actually have paragraph content
#       - paragraph text (<p>)
#     Removes: tables/infobox, TOC, navboxes, hatnotes, thumbnails, reflists,
#              inline reference markers ([1]), coordinates, edit links, etc.
#     Produces clean, plain text lines separated by newlines.
#     """
#     params = {
#         "action": "parse",
#         "format": "json",
#         "formatversion": "2",
#         "oldid": str(revid),
#         "prop": "text|sections",  # sections not strictly required, but harmless
#         # (We use the rendered HTML; it’s consistent across old revisions.)
#     }
#     r = session.get(WIKI_API, params=params, timeout=30)
#     r.raise_for_status()
#     data = r.json()
#     if "error" in data:
#         raise RuntimeError(f"parse error for oldid={revid}: {data['error']}")
# 
#     html = data.get("parse", {}).get("text", "")
#     soup = BeautifulSoup(html, "html.parser")
# 
#     # Focus on the main content container
#     root = soup.select_one("div.mw-parser-output") or soup
# 
#     # Strip non-article chrome and clutter
#     drop_selectors = [
#         "table.infobox", "table.vertical-navbox", "table.navbox", "table.metadata",
#         "table.ambox", "table.toc", "#toc", "div.toc", "div.navbox",
#         "div.hatnote", "div.stub", "div.sidebar", "div.metadata",
#         "div.thumb", "figure", "div.reflist", "ol.references", "dl.reflist",
#         "span.mw-editsection", "span.mw-cite-backlink", "span.mw-empty-elt",
#         "span.geo", "span.coordinates", "span.mw-kartographer-maplink",
#     ]
#     for sel in drop_selectors:
#         for el in root.select(sel):
#             el.decompose()
# 
#     # Remove inline reference markers like [1], [a], etc.
#     for el in root.select("sup.reference"):
#         el.decompose()
# 
#     lines = []
#     pending_heading = None
#     have_para_after_heading = False
# 
#     # Iterate high-level flow of the article body only:
#     for node in root.children:
#         name = getattr(node, "name", None)
#         if name in ("h2", "h3", "h4"):
#             # flush a previous heading if it actually had paragraph content
#             if pending_heading and have_para_after_heading:
#                 lines.append(pending_heading)
#             # start a new pending heading (strip any leftover anchors/spans)
#             # Note: edit links were already removed above
#             heading_text = node.get_text(separator=" ", strip=True)
#             # Some headings include trailing "edit" artifacts; the selector above should remove them.
#             pending_heading = heading_text
#             have_para_after_heading = False
# 
#         elif name == "p":
#             # Inline cleanups inside paragraphs (coordinates, references already handled)
#             # Remove any leftover superscripts or footnote-return anchors
#             for el in node.select("sup, a.mw-jump-link"):
#                 el.decompose()
# 
#             text = node.get_text(separator=" ", strip=True)
#             if text:
#                 # If a real paragraph appears after a heading, emit the heading once
#                 if pending_heading and not have_para_after_heading:
#                     lines.append(pending_heading)
#                     pending_heading = None
#                 lines.append(text)
#                 have_para_after_heading = True
# 
#         # Ignore lists, tables, galleries, etc. by design to keep only narrative text
# 
#     # Do NOT append a trailing heading with no paragraphs beneath it
#     cleaned = "\n".join(lines)
#     return cleaned


def fetch_plaintext_for_revid(session, revid: int) -> str:
    """
    Render a specific revision (oldid) and return ONLY:
      • section headings (h2/h3/h4), ALWAYS kept
      • paragraph text (<p>)
    Removes: infobox/TOC/navboxes/tables/thumbs/hatnotes/reflists,
             inline reference markers (e.g., [1]), coordinates, edit links, scripts/styles.
    Produces clean, newline-separated plain text in article order.
    """
    params = {
        "action": "parse",
        "format": "json",
        "formatversion": "2",
        "oldid": str(revid),
        "prop": "text",
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"parse error for oldid={revid}: {data['error']}")

    html = data.get("parse", {}).get("text", "")
    soup = BeautifulSoup(html, "html.parser")

    # Focus on article body
    root = soup.select_one("div.mw-parser-output") or soup

    # Strip non-article chrome and clutter entirely
    drop_selectors = [
        "table.infobox", "table.vertical-navbox", "table.navbox", "table.metadata",
        "table.ambox", "table.toc", "#toc", "div.toc", "div.navbox",
        "div.hatnote", "div.stub", "div.sidebar", "div.metadata",
        "div.thumb", "figure", "div.reflist", "ol.references", "dl.reflist",
        "style", "script",
    ]
    for sel in drop_selectors:
        for el in root.select(sel):
            el.decompose()

    # Remove inline reference superscripts and other small clutter
    for el in root.select("sup.reference, sup.citation, span.mw-editsection, a.mw-jump-link, span.mw-empty-elt, span.geo, span.coordinates"):
        el.decompose()

    # Collect headings and paragraphs IN ORDER, keeping headings unconditionally
    lines = []
    # Use recursive search after pruning, so headings inside simple wrappers are still found
    for el in root.find_all(["h2", "h3", "h4", "p"], recursive=True):
        name = el.name
        if name in ("h2", "h3", "h4"):
            # Headings often wrap the visible text in <span class="mw-headline">…</span>
            heading_text = el.get_text(" ", strip=True)
            if heading_text:
                lines.append(heading_text)
        elif name == "p":
            # Extra cleanup inside paragraphs
            for s in el.select("sup, .mw-editsection, .mw-jump-link"):
                s.decompose()
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text)

    # Optional: drop empty trailing lines (rare)
    cleaned = "\n".join([ln for ln in lines if ln.strip()])
    return cleaned

def ensure_out_dir(base: Path, title: str) -> Path:
    folder = base / sanitize_title(title)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def filename_for_offset(offset: int, prefix: str) -> str:
    return f"{prefix}_C.txt" if offset == 0 else f"{prefix}_C-{offset}.txt"

def crawl_title(
    session: requests.Session,
    title: str,
    out_dir: Path,
    prefix: str,
    offsets: List[int],
    request_delay: float = 0.1,
) -> List[Tuple[int, int, str]]:
    """
    Returns list of (offset, revid, path_str) saved.
    """
    needed_count = max(offsets) + 1  # need at least this many revisions in the list
    revs = fetch_revision_list(session, title, needed_count=needed_count)

    # Map desired offsets to revision ids that actually exist
    available_max_index = len(revs) - 1
    chosen: List[Tuple[int, int]] = []
    for off in offsets:
        if off <= available_max_index:
            chosen.append((off, revs[off]["revid"]))
        else:
            print(f"[WARN] {title}: requested offset {off} but only {len(revs)} revisions available; skipping.")

    save_dir = ensure_out_dir(out_dir, title)
    results = []
    for off, revid in chosen:
        try:
            text = fetch_plaintext_for_revid(session, revid)
            fname = filename_for_offset(off, prefix)
            path = save_dir / fname
            path.write_text(text, encoding="utf-8")
            results.append((off, revid, str(path)))
            print(f"[OK] {title}: saved {fname} (revid {revid})")
        except Exception as e:
            print(f"[ERROR] {title}: failed offset {off} (revid {revid}) → {e}")
        time.sleep(request_delay)  # be polite to the API
    return results

def read_titles_from_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    ap = argparse.ArgumentParser(description="Crawl Wikipedia revisions to plain text files.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--titles", nargs="+", help="One or more Wikipedia page titles.")
    src.add_argument("--titles-file", type=str, help="Path to a file with one title per line.")
    ap.add_argument("--prefix", type=str)
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory (will create subfolders per title).")
    ap.add_argument("--step", type=int, default=3, help="Revision step size (default: 3).")
    ap.add_argument("--max-offset", type=int, default=147, help="Max revision offset (default: 147).")
    ap.add_argument("--delay", type=float, default=0.1, help="Delay between API calls in seconds (default: 0.1).")
    args = ap.parse_args()

    titles: List[str]
    if args.titles:
        titles = args.titles
    else:
        titles = read_titles_from_file(Path(args.titles_file))

    offsets = compute_offsets(step=args.step, max_offset=args.max_offset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    print(f"Will fetch offsets: {offsets}")
    for title in titles:
        print(f"\n=== {title} ===")
        crawl_title(session, title, out_dir, args.prefix, offsets, request_delay=args.delay)

if __name__ == "__main__":
    main()
