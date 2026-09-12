"""Layer 3, part 2: dataset assembly, cleaning, and leakage-safe splitting.

Sources this pipeline normalises (files are *not* shipped - download them):

  phishing : PhishTank (CSV), OpenPhish (text feed)
  benign   : Tranco (CSV: rank,domain), Majestic Million (CSV: GlobalRank,...,Domain,...)

Everything is normalised to columns:  url, label, source, date
  label: 0 = benign, 1 = phishing
  date : ISO-8601 string if the source carries a per-row timestamp, else NaT
         (OpenPhish / Tranco / Majestic have no per-row dates - see README).

See the README "Dataset bias" section for why the *composition* of these lists
matters more than their size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ML_CONFIG, MLConfig
from .lexical_features import extract_lexical_features, registered_domain_of

NORM_COLUMNS = ["url", "label", "source", "date"]


# ==========================================================================
# per-source loaders  ->  normalised DataFrame(url, label, source, date)
# ==========================================================================
def _df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=NORM_COLUMNS)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def load_phishtank(path: str | Path) -> pd.DataFrame:
    """PhishTank CSV: phish_id,url,phish_detail_url,submission_time,verified,..."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    url_col = _first_col(raw, ["url", "URL"])
    date_col = _first_col(raw, ["submission_time", "verification_time"], required=False)
    recs = [
        {"url": u, "label": 1, "source": "phishtank",
         "date": (raw[date_col][i] if date_col else None)}
        for i, u in enumerate(raw[url_col]) if u.strip()
    ]
    return _df(recs)


def load_openphish(path: str | Path) -> pd.DataFrame:
    """OpenPhish feed: one URL per line, no timestamp."""
    text = Path(path).read_text(errors="replace")
    recs = [
        {"url": line.strip(), "label": 1, "source": "openphish", "date": None}
        for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    return _df(recs)


def load_tranco(path: str | Path, *, list_date: str | None = None,
                limit: int | None = None) -> pd.DataFrame:
    """Tranco CSV: rank,domain  (no header). Whole-list date is optional."""
    raw = pd.read_csv(path, dtype=str, header=None, keep_default_na=False,
                      names=["rank", "domain"])
    domains = raw["domain"] if "domain" in raw else raw.iloc[:, -1]
    recs = []
    for i, d in enumerate(domains):
        if limit is not None and i >= limit:
            break
        d = d.strip().lower()
        if d:
            recs.append({"url": f"http://{d}/", "label": 0, "source": "tranco",
                         "date": list_date})
    return _df(recs)


def load_majestic(path: str | Path, *, list_date: str | None = None,
                  limit: int | None = None) -> pd.DataFrame:
    """Majestic Million CSV: GlobalRank,TldRank,Domain,TLD,RefSubNets,RefIPs,..."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    dom_col = _first_col(raw, ["Domain", "domain"])
    recs = []
    for i, d in enumerate(raw[dom_col]):
        if limit is not None and i >= limit:
            break
        d = d.strip().lower()
        if d:
            recs.append({"url": f"http://{d}/", "label": 0, "source": "majestic",
                         "date": list_date})
    return _df(recs)


def load_normalized_csv(path: str | Path) -> pd.DataFrame:
    """Already-normalised url,label,source,date CSV (the common format)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    url_col = _first_col(raw, ["url", "URL"])
    label_col = _first_col(raw, ["label", "Label", "class", "target"])
    src_col = _first_col(raw, ["source", "src"], required=False)
    date_col = _first_col(raw, ["date", "timestamp", "submission_time"], required=False)
    recs = []
    for i in range(len(raw)):
        recs.append({
            "url": raw[url_col][i],
            "label": raw[label_col][i],
            "source": raw[src_col][i] if src_col else "unknown",
            "date": raw[date_col][i] if date_col else None,
        })
    df = _df(recs)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


_LOADERS = {
    "phishtank": load_phishtank,
    "openphish": load_openphish,
    "tranco": load_tranco,
    "majestic": load_majestic,
    "normalized": load_normalized_csv,
    "csv": load_normalized_csv,
}


def load_source(kind: str, path: str | Path, **kw) -> pd.DataFrame:
    kind = kind.lower()
    if kind not in _LOADERS:
        raise ValueError(f"unknown source kind {kind!r}; known: {sorted(_LOADERS)}")
    fn = _LOADERS[kind]
    # only pass kwargs the loader accepts
    import inspect
    allowed = set(inspect.signature(fn).parameters)
    return fn(path, **{k: v for k, v in kw.items() if k in allowed})


def load_many(specs: list[dict]) -> pd.DataFrame:
    """specs: [{"kind": "phishtank", "path": "...", "list_date": "..."}, ...]"""
    frames = [load_source(**s) for s in specs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _df([])
    return pd.concat(frames, ignore_index=True)


def _first_col(df: pd.DataFrame, candidates: list[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"none of {candidates} in columns {list(df.columns)}")
    return None


# ==========================================================================
# cleaning
# ==========================================================================
@dataclass
class CleanStats:
    total_in: int = 0
    total_out: int = 0
    phishing: int = 0
    benign: int = 0
    removed_missing_url: int = 0
    removed_missing_label: int = 0
    removed_bad_label: int = 0
    removed_duplicate_urls: int = 0
    removed_domain_cap: int = 0
    invalid_urls_kept: int = 0        # unparseable but retained (flagged feature)
    empty_host_dropped: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def render(self) -> str:
        d = self.as_dict()
        ratio = (d["phishing"] / d["benign"]) if d["benign"] else float("inf")
        return (
            "Dataset cleaning\n"
            f"  Total URLs (in) ............ {d['total_in']}\n"
            f"  Total URLs (out) ........... {d['total_out']}\n"
            f"  Phishing .................. {d['phishing']}\n"
            f"  Benign ................... {d['benign']}\n"
            f"  Class ratio (phish:benign) {ratio:.3f} : 1\n"
            f"  Removed duplicates ........ {d['removed_duplicate_urls']}\n"
            f"  Removed missing URL ....... {d['removed_missing_url']}\n"
            f"  Removed missing label ..... {d['removed_missing_label']}\n"
            f"  Removed bad label ......... {d['removed_bad_label']}\n"
            f"  Removed by per-domain cap . {d['removed_domain_cap']}\n"
            f"  Dropped (empty host) ...... {d['empty_host_dropped']}\n"
            f"  Invalid URLs kept (flagged) {d['invalid_urls_kept']}\n"
        )


def _registered_domain(url: str) -> str:
    return registered_domain_of(url)


def clean_dataset(
    df: pd.DataFrame,
    *,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    dedupe_urls: bool = True,
    drop_malformed: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, CleanStats]:
    """Clean a normalised dataframe. Keeps as much as possible (spec #4)."""
    st = CleanStats(total_in=len(df))
    df = df.copy()

    # --- URLs -------------------------------------------------------------
    df["url"] = df["url"].astype(str).str.strip()
    mask_missing_url = df["url"].isin(["", "nan", "None"])
    st.removed_missing_url = int(mask_missing_url.sum())
    df = df[~mask_missing_url]

    # --- labels --------------------------------------------------------
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    mask_missing_label = df["label"].isna()
    st.removed_missing_label = int(mask_missing_label.sum())
    df = df[~mask_missing_label]

    mask_bad_label = ~df["label"].isin([0, 1])
    st.removed_bad_label = int(mask_bad_label.sum())
    df = df[~mask_bad_label]
    df["label"] = df["label"].astype(int)

    # --- parse-ability -----------------------------------------------
    lex_cache: dict[str, tuple[str, str, bool]] = {}

    def _parse(u: str):
        if u not in lex_cache:
            try:
                lx = extract_lexical_features(u)
                lex_cache[u] = (lx.hostname or "",
                                (lx.registered_domain or lx.hostname or "").lower(),
                                bool(lx.is_malformed))
            except Exception:  # noqa: BLE001
                lex_cache[u] = ("", "", True)
        return lex_cache[u]

    parsed = df["url"].map(_parse)
    df["_host"] = parsed.map(lambda t: t[0])
    df["registered_domain"] = parsed.map(lambda t: t[1])
    df["_malformed"] = parsed.map(lambda t: t[2])

    empty_host = df["_host"] == ""
    st.empty_host_dropped = int(empty_host.sum())
    df = df[~empty_host]

    if drop_malformed:
        st.invalid_urls_kept = 0
        df = df[~df["_malformed"]]
    else:
        st.invalid_urls_kept = int(df["_malformed"].sum())

    # --- exact-duplicate URLs (keep earliest by date, then first seen) --
    if dedupe_urls:
        before = len(df)
        df = df.sort_values("date", na_position="last", kind="stable")
        df = df.drop_duplicates(subset="url", keep="first")
        st.removed_duplicate_urls = before - len(df)

    # --- per-registered-domain cap (campaign flooding) ----------------
    # Grouped by domain ALONE, not (domain, label): capping per pair would let
    # one domain contribute `cap` benign *plus* `cap` phishing rows, i.e. twice
    # the documented limit, and a single big campaign could still dominate.
    cap = ml_config.max_urls_per_registered_domain
    if cap is not None and cap > 0:
        before = len(df)
        df = (
            df.sort_values("date", na_position="last", kind="stable")
              .groupby("registered_domain", group_keys=False, sort=False)
              .head(cap)
        )
        st.removed_domain_cap = before - len(df)

    df = df.drop(columns=["_host", "_malformed"]).reset_index(drop=True)

    st.total_out = len(df)
    st.phishing = int((df["label"] == 1).sum())
    st.benign = int((df["label"] == 0).sum())

    if verbose:
        print(st.render())
    return df, st


# ==========================================================================
# leakage-safe splitting  (spec #5)
# ==========================================================================
@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: str
    date_ranges: dict = field(default_factory=dict)
    dropped_cross_split_domains: int = 0

    def sizes(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _date_bounds(s: pd.Series) -> tuple[str | None, str | None]:
    s = s.dropna()
    if s.empty:
        return None, None
    return str(s.min()), str(s.max())


def time_based_split(
    df: pd.DataFrame,
    *,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
) -> SplitResult:
    """Older rows -> train, then validation, newest -> test (spec #5).

    Rows without a usable date go to `ml_config.undated_bucket`
    ("train" | "exclude" | "grouped"). "grouped" routes undated rows through a
    domain-grouped random split so no domain spans train/test.
    """
    df = df.copy()
    if "registered_domain" not in df.columns:
        df["registered_domain"] = df["url"].map(_registered_domain)
    # Be robust to a hand-built frame whose `date` is strings / naive / missing.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    train_end = pd.Timestamp(ml_config.train_end, tz="UTC")
    val_end = pd.Timestamp(ml_config.val_end, tz="UTC")

    dated = df[df["date"].notna()]
    undated = df[df["date"].isna()]

    tr = dated[dated["date"] < train_end]
    va = dated[(dated["date"] >= train_end) & (dated["date"] < val_end)]
    te = dated[dated["date"] >= val_end]

    method = "time_based"
    if undated.empty:
        pass
    elif ml_config.undated_bucket == "train":
        tr = pd.concat([tr, undated], ignore_index=True)
        method = "time_based+undated_to_train"
    elif ml_config.undated_bucket == "exclude":
        method = "time_based+undated_excluded"
    elif ml_config.undated_bucket == "grouped":
        g_tr, g_va, g_te = _grouped_split(undated, ml_config)
        tr = pd.concat([tr, g_tr], ignore_index=True)
        va = pd.concat([va, g_va], ignore_index=True)
        te = pd.concat([te, g_te], ignore_index=True)
        method = "time_based+undated_grouped"

    # If time slicing produced an empty train or test (e.g. all rows undated and
    # excluded, or all dates on one side), fall back to a domain-grouped split.
    if len(tr) == 0 or len(te) == 0:
        g_tr, g_va, g_te = _grouped_split(df, ml_config)
        return SplitResult(g_tr, g_va, g_te, "grouped_fallback",
                           {"note": "no usable time span; used grouped split"})

    res = SplitResult(
        tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True),
        method,
        {
            "train": _date_bounds(tr["date"]),
            "val": _date_bounds(va["date"]),
            "test": _date_bounds(te["date"]),
        },
    )
    if ml_config.drop_cross_split_domains:
        res = _strip_cross_split_domains(res)
    return res


def _grouped_split(
    df: pd.DataFrame, ml_config: MLConfig, ratios=(0.7, 0.15, 0.15)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic split by registered domain - a domain is wholly in one
    bucket. Used as a fallback / for undated rows."""
    import numpy as np

    if df.empty:
        return df.copy(), df.copy(), df.copy()
    domains = df["registered_domain"].fillna("").astype(str)
    uniq = sorted(domains.unique())
    rng = np.random.default_rng(ml_config.random_seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(n * ratios[0])
    n_va = int(n * (ratios[0] + ratios[1]))
    buckets = {d: ("train" if i < n_tr else "val" if i < n_va else "test")
               for i, d in enumerate(uniq)}
    b = domains.map(buckets)
    return (df[b == "train"].reset_index(drop=True),
            df[b == "val"].reset_index(drop=True),
            df[b == "test"].reset_index(drop=True))


def _strip_cross_split_domains(res: SplitResult) -> SplitResult:
    """Ensure no registered domain appears in both train and (val|test)."""
    train_domains = set(res.train["registered_domain"].dropna())
    n_before = len(res.val) + len(res.test)
    val = res.val[~res.val["registered_domain"].isin(train_domains)]
    test = res.test[~res.test["registered_domain"].isin(train_domains)]
    # also keep val/test disjoint
    val_domains = set(val["registered_domain"].dropna())
    test = test[~test["registered_domain"].isin(val_domains)]
    res.val = val.reset_index(drop=True)
    res.test = test.reset_index(drop=True)
    res.dropped_cross_split_domains = n_before - (len(res.val) + len(res.test))
    return res


def class_distribution(df: pd.DataFrame) -> dict:
    n1 = int((df["label"] == 1).sum())
    n0 = int((df["label"] == 0).sum())
    return {
        "phishing": n1,
        "benign": n0,
        "total": n0 + n1,
        "phishing_fraction": (n1 / (n0 + n1)) if (n0 + n1) else 0.0,
        "ratio_phishing_to_benign": (n1 / n0) if n0 else float("inf"),
    }
