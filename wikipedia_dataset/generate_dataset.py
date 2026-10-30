#!/usr/bin/env python3
"""
Generate an i.i.d. dataset of random Wikipedia articles (English by default).

Each observation is one independent call to the MediaWiki list=random module
(namespace 0, redirects excluded), so rows are i.i.d. draws with replacement.
For each article we record structural covariates and two recent-activity
targets (pageviews over the last 30 days, edits over the last 12 months).

Usage:
    python3 generate_dataset.py                    # 1000 rows -> dataset.csv + dataset_key.csv
    python3 generate_dataset.py -n 800 -o other.csv
    python3 generate_dataset.py --lang it          # another language edition
    python3 generate_dataset.py --contact you@example.org   # 200 req/min instead of 10
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

API = "https://{lang}.wikipedia.org/w/api.php"
PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "{lang}.wikipedia/all-access/user/{title}/daily/{start}/{end}")
USER_AGENT = "IIDDatasetBuilder/1.0 (statistics research dataset)"
PAUSE = 0.35  # seconds between requests (Wikimedia allows 200/min with a contact in the User-Agent)
BATCH = 50   # page ids per metadata request
LANG = "en"  # Wikipedia language edition, set from --lang

FIELDS = [
    "page_id", "title",
    # covariates
    "n_categories", "n_internal_links", "n_languages", "has_image", "age_years", "length_kb",
    # targets
    "views_30d", "edits_12m", "log_views_30d", "log_edits_12m",
]

# column names in the distributed file: covariates x1..x6, bivariate response y1, y2
MASK = {
    "n_categories": "x1",
    "n_internal_links": "x2",
    "n_languages": "x3",
    "has_image": "x4",
    "age_years": "x5",
    "length_kb": "x6",
    "log_views_30d": "y1",
    "log_edits_12m": "y2",
}

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def get(url, params=None, ok404=False):
    for attempt in range(8):
        try:
            r = session.get(url, params=params, timeout=30)
            if ok404 and r.status_code == 404:
                time.sleep(PAUSE)
                return None
            if r.status_code == 200:
                data = r.json()
                if data.get("error", {}).get("code") == "maxlag":
                    raise RuntimeError("maxlag")
                time.sleep(PAUSE)
                return data
            wait = int(r.headers.get("Retry-After", 0)) or min(5 * 2 ** attempt, 120)
            print(f"  HTTP {r.status_code}, waiting {wait}s", file=sys.stderr)
        except (requests.RequestException, ValueError, RuntimeError) as e:
            wait = min(5 * 2 ** attempt, 120)
            print(f"  error ({e}), waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"request failed: {url} {params}")


def api(**params):
    params.update(action="query", format="json", formatversion=2, maxlag=5)
    return get(API.format(lang=LANG), params)


def sample(n):
    """n independent calls to list=random: i.i.d. draws."""
    pages = []
    while len(pages) < n:
        p = api(list="random", rnnamespace=0, rnfilterredir="nonredirects",
                rnlimit=1)["query"]["random"][0]
        pages.append((p["id"], p["title"]))
        print(f"[sample {len(pages)}/{n}] {p['title']}")
    return pages


def metadata(page_ids):
    """Structural covariates, BATCH pages per request, following continuation."""
    out = defaultdict(lambda: defaultdict(int))
    for i in range(0, len(page_ids), BATCH):
        params = dict(pageids="|".join(map(str, page_ids[i:i + BATCH])),
                      prop="info|pageimages|categories|links|langlinks",
                      piprop="name", cllimit="max", clshow="!hidden",
                      pllimit="max", plnamespace=0, lllimit="max")
        while True:
            data = api(**params)
            for p in data["query"]["pages"]:
                m = out[p["pageid"]]
                if p.get("missing"):  # deleted after it was drawn
                    m["missing"] = 1
                    continue
                if "length" in p:  # prop=info may appear in only one continuation round
                    m["length"] = p["length"]
                m["has_image"] |= int("pageimage" in p)
                m["n_categories"] += len(p.get("categories", []))
                m["n_internal_links"] += len(p.get("links", []))
                m["n_languages"] += len(p.get("langlinks", []))
            if "continue" not in data:
                break
            params.update(data["continue"])
        print(f"[metadata {min(i + BATCH, len(page_ids))}/{len(page_ids)}]")
    return out


def age_and_edits(page_id, today):
    first = api(pageids=page_id, prop="revisions", rvdir="newer", rvlimit=1,
                rvprop="timestamp")["query"]["pages"][0]["revisions"][0]["timestamp"]
    created = datetime.fromisoformat(first.replace("Z", "+00:00"))

    params = dict(pageids=page_id, prop="revisions", rvprop="timestamp", rvlimit="max",
                  rvdir="older", rvend=(today - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    edits = 0
    while True:
        data = api(**params)
        edits += len(data["query"]["pages"][0].get("revisions", []))
        if "continue" not in data:
            break
        params.update(data["continue"])
    return (today - created).days / 365.25, edits


def views(title, today):
    end = today.date() - timedelta(days=1)
    start = end - timedelta(days=29)
    pv = get(PAGEVIEWS.format(lang=LANG, title=quote(title.replace(" ", "_"), safe=""),
                              start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")),
             ok404=True)  # 404 = no views recorded in the period
    return sum(x["views"] for x in pv["items"]) if pv else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000, help="number of observations")
    ap.add_argument("-o", default="dataset.csv", help="output CSV file (a *_key.csv is written alongside)")
    ap.add_argument("--lang", default="en", help="Wikipedia language edition (default: en)")
    ap.add_argument("--contact", default="", help="email or URL for the User-Agent; without it Wikimedia allows only 10 requests/min")
    args = ap.parse_args()

    global LANG
    LANG = args.lang
    if args.contact:
        session.headers["User-Agent"] = USER_AGENT.replace(")", f"; {args.contact})")

    stem = args.o.removesuffix(".csv")
    sample_path, targets_path = stem + "_sample.json", stem + "_targets.jsonl"

    # the draws and the reference date are saved first, so an interrupted run resumes
    # on the same sample; delete these two files to draw a new one
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            saved = json.load(f)
        LANG = saved["lang"]
        today = datetime.fromisoformat(saved["reference_date"]).replace(tzinfo=timezone.utc)
        pages = [tuple(x) for x in saved["pages"]]
        print(f"Resuming sample of {len(pages)} draws from {sample_path}")
    else:
        # fixed reference date for the whole snapshot (UTC midnight)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        pages = sample(args.n)
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump({"lang": LANG, "reference_date": today.date().isoformat(),
                       "pages": pages}, f, ensure_ascii=False)

    unique = dict(pages)  # duplicate draws are queried only once
    meta = metadata(list(unique))
    unique = {i: t for i, t in unique.items() if not meta[i].get("missing")}

    targets = {}
    if os.path.exists(targets_path):
        with open(targets_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                targets[r["page_id"]] = (r["age"], r["edits"], r["views"])
    with open(targets_path, "a", encoding="utf-8") as f:
        for k, (page_id, title) in enumerate(unique.items(), 1):
            if page_id not in targets:
                age, edits = age_and_edits(page_id, today)
                targets[page_id] = (age, edits, views(title, today))
                f.write(json.dumps({"page_id": page_id, "age": age, "edits": edits,
                                    "views": targets[page_id][2]}) + "\n")
                f.flush()
            print(f"[targets {k}/{len(unique)}] {title}")

    rows = []
    for page_id, title in pages:
        if page_id not in unique:
            continue  # page deleted between draw and measurement
        m = meta[page_id]
        age, edits, v = targets[page_id]
        rows.append({
            "page_id": page_id,
            "title": title,
            "n_categories": m["n_categories"],
            "n_internal_links": m["n_internal_links"],
            "n_languages": m["n_languages"],
            "has_image": m["has_image"],
            "age_years": round(age, 3),
            "length_kb": round(m["length"] / 1024, 3),
            "views_30d": v,
            "edits_12m": edits,
            "log_views_30d": round(math.log1p(v), 4),
            "log_edits_12m": round(math.log1p(edits), 4),
        })

    # full file with real names and identifiers (keep private)
    key_path = args.o.removesuffix(".csv") + "_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # masked file to distribute: no identifiers, neutral column names
    with open(args.o, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MASK.values())
        w.writerows([r[col] for col in MASK] for r in rows)

    print(f"Done. Snapshot of {today.date()}: {len(rows)} rows "
          f"({len(unique)} distinct articles) saved to {args.o} (key: {key_path})")

if __name__ == "__main__":
    main()
