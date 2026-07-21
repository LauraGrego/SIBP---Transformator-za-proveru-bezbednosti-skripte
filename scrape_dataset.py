#!/usr/bin/env python3
"""
Harvest REAL install/deploy/admin shell scripts from GitHub and label them
heuristically as `safe` or `risky`, matching the classifier dataset schema:

    {"id", "label", "script", "category", "description"}

USAGE
-----
    export GITHUB_TOKEN=ghp_xxx          # a fine-grained or classic PAT
    python3 scrape_dataset.py --per-query 50 --out data/scraped.jsonl

NOTES / CAVEATS
---------------
* Requires a GitHub token. The code-search API (/search/code) needs auth and
  is rate-limited to ~10 requests/min, 1000 results per query. The script
  sleeps between pages to stay under the limit.
* Heuristic labels are a STARTING POINT, not ground truth. `risky` is assigned
  when a script shows privileged/destructive signals; everything else that
  looks like a real script is `safe`. REVIEW a sample by hand before training,
  and consider a `--review` pass. Malicious detection is intentionally left out
  here (you already have that set); scripts hitting malicious signals are
  DROPPED rather than mislabeled.
* Respect licenses/ToS. Scraped code carries its origin repo's license; the
  `description` field records the source URL for provenance.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

API = "https://api.github.com"

# Filenames that are typically install/deploy/admin shell scripts.
SEARCH_QUERIES = [

    "filename:install.sh",
    "filename:setup.sh",
    "filename:deploy.sh",
    "filename:entrypoint.sh",
    "filename:bootstrap.sh",
    "filename:postinstall.sh",
    "filename:provision.sh",
    "filename:customization.sh",
    "path:debian filename:postinst",
    "filename:configure.sh",
    "filename:init.sh",
]

# ---- Heuristic label signals -------------------------------------------------
# If ANY malicious signal matches, the script is DROPPED (not labeled), because
# you already own the malicious set and we don't want to pollute safe/risky.
MALICIOUS_SIGNALS = [
    r"/dev/tcp/\d+\.\d+\.\d+\.\d+",          # reverse shell to raw IP
    r"\bnc\b.*-e\b", r"ncat\b.*-e\b",          # netcat exec
    r"base64\s+-d.*\|\s*(ba)?sh",              # decode-then-exec
    r"curl[^\n]*\|\s*(ba)?sh.*\b(\d{1,3}\.){3}\d{1,3}",  # pipe from raw IP
    r"\bchattr\s+\+i\b.*\.ssh",                 # lock backdoored keys
    r"\bhistory\s+-c\b", r"unset\s+HISTFILE",   # anti-forensics
    r"\bcrontab\b.*(curl|wget).*\|\s*(ba)?sh", # cron persistence + fetch-exec
]

# Any of these -> `risky` (legitimate but privileged/destructive/irreversible).
RISKY_SIGNALS = [
    r"\bsudo\b", r"\bdoas\b",
    r"\brm\s+-[a-z]*r[a-z]*f", r"\brm\s+-[a-z]*f[a-z]*r",
    r"\bdd\s+if=", r"\bmkfs\b", r"\bfdisk\b", r"\bparted\b",
    r"\bchmod\s+-?R?\s*777\b", r"\bchown\s+-R\b",
    r"curl[^\n]*\|\s*(sudo\s+)?(ba)?sh", r"wget[^\n]*\|\s*(sudo\s+)?(ba)?sh",
    r"\bapt(-get)?\s+install\b", r"\byum\s+install\b", r"\bdnf\s+install\b",
    r"\bsystemctl\s+(enable|start|daemon-reload)\b",
    r"\biptables\s+-F\b", r"\bufw\s+disable\b", r"\bsetenforce\s+0\b",
    r">\s*/etc/", r"\btee\s+/etc/", r"\bsed\s+-i\b.*/etc/",
    r"\buseradd\b", r"\busermod\b", r"NOPASSWD", r"/etc/sudoers",
    r"--privileged\b", r"\bswapon\b", r"\bmodprobe\b",
]

MAL_RE = re.compile("|".join(MALICIOUS_SIGNALS), re.IGNORECASE)
RISKY_RE = re.compile("|".join(RISKY_SIGNALS), re.IGNORECASE)


def strip_comments(script):
    """Same policy as the synthetic generator: keep shebang, drop comments."""
    out = []
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("#") and not s.startswith("#!"):
            continue
        out.append(line)
    return "\n".join(out).strip("\n") + "\n"


def classify(script):
    """Return 'safe' | 'risky' | None (None => malicious-looking, drop it)."""
    if MAL_RE.search(script):
        return None
    if RISKY_RE.search(script):
        return "risky"
    return "safe"


def gh_request(url, token):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "shell-classifier-scraper",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r), r.headers


def search_code(query, token, per_query):
    """Yield (repo_full_name, path, download url pieces) for a search query."""
    items = []
    page = 1
    while len(items) < per_query and page <= 10:
        url = (f"{API}/search/code?q={urllib.parse.quote(query)}"
               f"&per_page=100&page={page}")
        try:
            data, _ = gh_request(url, token)
        except urllib.error.HTTPError as e:
            if e.code in (403, 422):   # rate limit or search bounds
                print(f"  [{query}] stopped: HTTP {e.code}", file=sys.stderr)
                break
            raise
        batch = data.get("items", [])
        if not batch:
            break
        items.extend(batch)
        page += 1
        time.sleep(6)   # code search: ~10 req/min
    return items[:per_query]


def fetch_content(item, token):
    """Fetch and base64-decode a file's contents from a search result."""
    url = item["url"]   # contents API url for the file
    try:
        data, _ = gh_request(url, token)
    except urllib.error.HTTPError:
        return None
    if data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", "replace")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=50,
                    help="max files to pull per search query")
    ap.add_argument("--out", default="data/scraped.jsonl")
    ap.add_argument("--max-bytes", type=int, default=20000,
                    help="skip files larger than this (bytes)")
    ap.add_argument("--max-total", type=int, default=1000,
                    help="stop the whole run after collecting this many scripts")
    ap.add_argument("--max-per-label", type=int, default=None,
                    help="stop adding to a label once it reaches this many "
                         "(keeps safe/risky balanced; default: no per-label cap)")
    args = ap.parse_args()

    def label_full(lbl):
        return (args.max_per_label is not None
                and counts[lbl] >= args.max_per_label)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN (a GitHub personal access token) first.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    seen, counts, dropped = set(), {"safe": 0, "risky": 0}, 0
    n = 0
    stop = False
    with open(args.out, "w") as fout:
        for q in SEARCH_QUERIES:
            if stop:
                break
            print(f"Searching: {q}")
            for item in search_code(q, token, args.per_query):
                if n >= args.max_total:
                    stop = True
                    break
                content = fetch_content(item, token)
                time.sleep(1)
                if not content or len(content.encode()) > args.max_bytes:
                    continue
                script = strip_comments(content)
                if len(script.strip()) < 20:
                    continue
                h = hash(script)
                if h in seen:
                    continue
                seen.add(h)
                label = classify(script)
                if label is None:
                    dropped += 1
                    continue
                if label_full(label):
                    continue
                n += 1
                counts[label] += 1
                rec = {
                    "id": f"scraped-{n:05d}",
                    "label": label,
                    "script": script,
                    "category": "",
                    # provenance kept here; blank it out before training if you
                    # want category/description empty for non-malicious rows.
                    "description": item.get("html_url", ""),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {n} records -> {args.out}")
    print(f"  safe={counts['safe']}  risky={counts['risky']}  "
          f"dropped(malicious-looking)={dropped}")
    print("REMEMBER: heuristic labels — hand-review a sample before training.")


if __name__ == "__main__":
    main()
