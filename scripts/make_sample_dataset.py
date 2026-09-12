#!/usr/bin/env python3
"""Generate a SYNTHETIC sample dataset for pipeline verification.

This is NOT a benchmark and NOT a substitute for real PhishTank / OpenPhish /
Tranco / Majestic data. It exists so `python -m detector.train --sample` can
exercise the whole pipeline (clean -> features -> time-split -> train -> eval ->
save) end to end, offline, deterministically.

It is deliberately built to AVOID the domain-age / popularity trap the README
describes:
  * benign rows include obscure domains, brand-new domains, deep paths, query
    strings, odd-but-legit TLDs, tiny orgs;
  * phishing rows include old-looking domains, short URLs, plain .com, as well
    as the usual new/suspicious-TLD/keyword-heavy cases;
  * both classes are spread across the same 2025-2026 date range.

Output: data/raw/sample/sample_dataset.csv  (columns: url,label,source,date)
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
N_BENIGN = 340
N_PHISH = 260
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample" / "sample_dataset.csv"

START = datetime(2025, 1, 1)
END = datetime(2026, 9, 1)

rng = random.Random(SEED)


def rand_date() -> str:
    span = (END - START).days
    return (START + timedelta(days=rng.randint(0, span),
                              seconds=rng.randint(0, 86399))).isoformat()


# A few well-known domains ...
BENIGN_FAMOUS = [
    "google.com", "github.com", "wikipedia.org", "bbc.co.uk", "nytimes.com",
    "cloudflare.com", "python.org", "mozilla.org", "apache.org", "debian.org",
    "sqlite.org", "ffmpeg.org", "kernel.org", "postgresql.org", "gnu.org",
    "openstreetmap.org", "archlinux.org", "freebsd.org", "curl.se", "nlnet.nl",
]
# ... plus a large procedurally-generated pool of obscure small-org domains
# across many TLDs, so the benign class is NOT "just famous old domains"
# (README: dataset-bias protection). Rebuilt deterministically below.
_BW1 = ["river", "green", "north", "quiet", "bright", "cedar", "harbor", "maple",
        "stone", "willow", "amber", "coral", "delta", "ember", "frost", "grove",
        "hazel", "iron", "juno", "larch", "moss", "opal", "pine", "quill",
        "rowan", "sage", "teal", "umbra", "vale", "wren"]
_BW2 = ["bakery", "dental", "studio", "consulting", "library", "clinic", "farm",
        "works", "retreat", "collective", "supply", "brew", "press", "labs",
        "garden", "forge", "atelier", "kitchen", "school", "guild", "cycles",
        "records", "roasters", "botanicals", "tailors", "pottery"]
_BTLD = ["com", "org", "net", "io", "co", "dev", "shop", "co.uk", "de", "fr",
         "nl", "se", "es", "ca", "com.au", "app", "studio", "family", "coffee"]


def _benign_pool(n: int) -> list[str]:
    pool, seen = [], set()
    tries = 0
    while len(pool) < n and tries < n * 50:
        tries += 1
        host = f"{rng.choice(_BW1)}-{rng.choice(_BW2)}.{rng.choice(_BTLD)}"
        if host not in seen:
            seen.add(host)
            pool.append(host)
    return pool


BENIGN_HOSTS = BENIGN_FAMOUS + _benign_pool(220)
BENIGN_PATHS = [
    "/", "/about", "/contact", "/blog/2026/03/release-notes",
    "/docs/getting-started/installation", "/products/category/12/item/9981",
    "/search?q=annual+report&page=3", "/user/profile/settings?tab=security",
    "/wiki/Special:RecentChanges", "/api/v2/status", "/careers/engineering",
    "/2025/annual-review.pdf", "/support/tickets?status=open&sort=desc",
    "/shop/checkout?cart=88213&coupon=SPRING", "/events/meetup-2026-05-14",
]

PHISH_TEMPLATES = [
    # brand in subdomain, different registered domain
    "http://{brand}.account-verify-{n}.{tld}/login",
    "https://secure-{brand}.{rand}-support.{tld}/signin?session={n}",
    "http://{brand}-{kw}.{rand}.{tld}/{kw}/confirm",
    # typosquats on plain .com (old-looking, short)
    "http://{typo}.com/",
    "https://{typo}.com/login",
    "http://www.{typo}.net/account/update",
    # IP-based
    "http://{ip}/{brand}/webscr?cmd=_login-run",
    "http://{ip}:8081/secure/{kw}",
    # @ trick
    "http://{brand}.com@{rand}-{kw}.{tld}/verify",
    # punycode / homograph
    "http://xn--{puny}-{n}.{tld}/{kw}",
    # long deep credential-phish path on a random domain
    "https://{rand}{n}.{tld}/{kw}/{kw2}/session/{n}{n}/reauth?next=/wallet",
    # suspicious TLD, keyword heavy
    "http://{kw}-{brand}-{kw2}.{tld}/",
    # plain-looking, minimal
    "https://{rand}-{brand}.{tld}/",
]

BRANDS = ["paypal", "microsoft", "apple", "amazon", "netflix", "binance",
          "facebook", "google", "instagram", "linkedin"]
TYPOS = ["paypa1", "paypaI", "micros0ft", "arnazon", "app1e", "netfl1x",
         "binnance", "faceb00k", "g00gle", "linkedln", "1nstagram", "amaz0n-support"]
KWS = ["login", "verify", "secure", "account", "update", "confirm", "wallet",
       "kyc", "signin", "password", "payment", "webscr", "auth", "recover"]
PHISH_TLDS = ["xyz", "top", "tk", "click", "gq", "ml", "cf", "zip", "com", "net",
              "info", "online", "live", "support"]
PUNY = ["pypal", "cloudflare", "glgle", "arnazon", "micrsft"]


def rand_ip() -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def make_phish() -> str:
    t = rng.choice(PHISH_TEMPLATES)
    return t.format(
        brand=rng.choice(BRANDS), typo=rng.choice(TYPOS), kw=rng.choice(KWS),
        kw2=rng.choice(KWS), tld=rng.choice(PHISH_TLDS), n=rng.randint(1, 999),
        rand="".join(rng.choice("abcdefghijkmnpqrstuvwxyz") for _ in range(rng.randint(5, 12))),
        ip=rand_ip(), puny=rng.choice(PUNY),
    )


def make_benign() -> str:
    host = rng.choice(BENIGN_HOSTS)
    scheme = "https" if rng.random() < 0.85 else "http"
    path = rng.choice(BENIGN_PATHS)
    return f"{scheme}://{host}{path}"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: set[str] = set()

    def add(url: str, label: int, source: str):
        if url in seen:
            return
        seen.add(url)
        rows.append({"url": url, "label": label, "source": source, "date": rand_date()})

    tries = 0
    while sum(1 for r in rows if r["label"] == 0) < N_BENIGN and tries < N_BENIGN * 20:
        add(make_benign(), 0, "sample_benign")
        tries += 1
    tries = 0
    while sum(1 for r in rows if r["label"] == 1) < N_PHISH and tries < N_PHISH * 40:
        add(make_phish(), 1, "sample_phish")
        tries += 1

    rng.shuffle(rows)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "label", "source", "date"])
        w.writeheader()
        w.writerows(rows)

    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    print(f"wrote {OUT}  ({len(rows)} rows: benign={n0}, phishing={n1})")
    print("NOTE: synthetic sample for pipeline verification only.")


if __name__ == "__main__":
    main()
