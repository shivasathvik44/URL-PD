# Phishing URL Detector

A phishing-URL detector built on a **four-layer architecture** with an explicit
decision engine on top. Each layer contributes evidence; **no single layer makes
the final call**, and every verdict comes with the reasons behind it.

| Layer | What it does | Cost | Network |
|------:|--------------|------|---------|
| **1** | **Lexical analysis** — URL string/structure only | microseconds | none |
| **2** | **DNS / domain intelligence** — DNS, RDAP/WHOIS, TLS | ~0.1–5 s | DNS + HTTPS |
| **3** | **Machine learning** — Random Forest over the L1+L2 feature vector | ~1 ms | none |
| **4** | **Page content + LLM** — fetch the page, analyse text/forms/brand | ~1–90 s | HTTPS + LLM (local by default) |
| — | **Decision engine** — blocklist → ML thresholds → Layer 4 → rules | — | none |

```
                              URL
                               │
                               ▼
                   ┌───────────────────────┐
                   │ LAYER 1  lexical      │  offline, deterministic
                   │ shape · brands · TLD  │  41 features
                   └───────────┬───────────┘
                               ▼
                   ┌───────────────────────┐
                   │ LAYER 2  domain intel │  cached (24 h), best-effort
                   │ DNS · RDAP · TLS · IP │  24 features
                   └───────────┬───────────┘
                               ▼
                   ┌───────────────────────┐
                   │ LAYER 3  Random Forest│  65 ordered features
                   │ P(phishing) ∈ [0,1]   │  uncalibrated
                   └───────────┬───────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │                         │                         │
 blocklist?              P ≥ HIGH (0.70)           P ≤ LOW (0.30)
     │                         │                         │
     ▼                         ▼                         ▼
  PHISHING                 PHISHING                    SAFE
                               │
                    LOW < P < HIGH  ("uncertain band")
                               │
                               ▼
                   ┌───────────────────────┐
                   │ LAYER 4  page + LLM   │  SSRF-guarded GET, no JS
                   │ text · forms · brand  │  static facts computed first,
                   │ hidden · scripts      │  then shown to the LLM
                   └───────────┬───────────┘
                               ▼
                   ┌───────────────────────┐
                   │   DECISION ENGINE     │  ordered rules, no averaging
                   └───────────┬───────────┘
                               ▼
        ┌──────────────────────────────────────────┐
        │ classification   phishing│safe│uncertain  │
        │ risk_score       ordinal severity         │
        │ ml_probability   Layer 3 raw output       │
        │ llm_likelihood   Layer 4 enum → number    │
        │ reasons[]        code + message + source  │
        │ timings · metadata · errors               │
        └──────────────────────────────────────────┘
```

---

## Quick start

```bash
git clone <this-repo> phishing-url-detector
cd phishing-url-detector

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # optional: the `detector` console script

python -m detector train --sample --no-layer2      # ~5 s, offline
python -m detector analyze "http://paypal.com-verify.tk/login/secure"
```

Layers 1–3 need nothing else. **Layer 4's LLM stage runs on a local model by
default** — install [Ollama](https://ollama.com/download), then:

```bash
ollama pull gemma3:4b
ollama list                      # expect: gemma3:4b
```

No API key, no per-call cost, no page content leaving the machine. Without it
Layer 4 simply reports `llm-unavailable` and the verdict falls back to Layers
1–3. Anthropic remains available as an alternative — see
[LLM configuration](#llm-configuration).

```
LAYER 1  lexical (offline)
  registered domain .. com-verify.tk
  brand in wrong place yes (paypal)
  suspicious TLD ..... yes (.tk)
  keywords ........... login, verify, secure
...
------------------------------------------------------------------------
RESULT       PHISHING
risk score   0.94   (ordinal severity, NOT a calibrated probability)
ml / llm     0.9383 / -
path         ml_high

reasons:
  !! [high  ] URL puts the brand 'paypal' in the subdomain or path, but the
              registered domain is 'com-verify.tk'
     [low   ] Top-level domain '.tk' is one commonly abused for disposable
              phishing domains
     [info  ] URL-feature model scored 0.94 probability of phishing (at or
              above the 0.70 high-confidence threshold)
```

Python 3.10+.

---

## The one top-level API

```python
from detector import analyze_url

result = analyze_url("https://paypal.security-check.xyz/login")

result.classification     # "phishing" | "safe" | "uncertain"
result.risk_score         # 0..1 ordinal severity (NOT a calibrated probability)
result.ml_probability     # Layer 3 raw output, or None if no model
result.llm_likelihood     # Layer 4 enum → number, or None if Layer 4 did not run
result.layer4_ran         # was the page actually fetched and analysed?
result.reason_messages()  # human-readable evidence, most serious first

result.layer1             # LexicalFeatures
result.layer2             # DomainIntelligence | None
result.page               # PageContent | None
result.llm                # LLMAnalysis | None

result.timings_ms         # {'layer1_ms': 0.4, 'layer3_ms': 1.2, 'total_ms': 1.6}
result.metadata           # schema version, model version, thresholds, ...
result.errors             # anything that degraded, never raised
result.as_dict()          # the whole thing, JSON-safe and credential-redacted
```

`analyze_url` **never raises**. A failure in any layer is recorded in `errors`
and degrades the verdict (usually to `"uncertain"`).

Lower-level building blocks, for when you want one layer on its own:

```python
from detector import (
    extract_lexical_features,   # Layer 1  -> LexicalFeatures
    analyze_domain,             # Layer 2  -> DomainIntelligence
    analyze,                    # Layers 1+2 -> DetectionResult (raw features)
    predict_url,                # Layers 1-3 -> probability dict
    fetch_page, analyze_html,   # Layer 4a/4b -> PageContent / PageStructure
    analyze_page_content,       # Layer 4d -> LLMAnalysis
    decide, triage,             # decision engine (pure functions)
)
```

---

## Command line

```bash
python -m detector analyze URL [URL ...]      # all four layers + decision
python -m detector analyze --json URL         # machine-readable
python -m detector features URL               # raw Layer 1+2 features, no verdict
python -m detector train --sample             # train the Layer 3 model
python -m detector evaluate --csv data.csv    # score a model against labelled data
```

`analyze` flags:

| Flag | Effect |
|---|---|
| `--json` | one JSON document (an array for several URLs) |
| `--layer1-only` / `--no-layer2` | no DNS/WHOIS/TLS traffic at all |
| `--no-llm` | never run Layer 4, even in the uncertain band |
| `--no-tls` | skip the TLS certificate check |
| `--no-cache` | bypass both caches |
| `--low` / `--high` | override the decision thresholds |
| `--models-dir` | where `random_forest.joblib` lives |
| `--log-level`, `--log-json` | structured logging (global flags, before the subcommand) |

**Exit codes** carry the verdict for scripting: `0` safe, `1` phishing,
`2` uncertain.

`python main.py <same args>` works identically from a checkout without
installing. After `pip install -e .` the console script is `detector`.

---

## Local Service & Firefox Extension

In addition to the Python library and CLI, the detector includes a local **FastAPI service** and a companion **Firefox WebExtension** (Manifest V3) for automated browser security.

### 1. Run the Local Service

Install the service dependencies and start the hardened daemon:

```bash
pip install -e ".[service]"
python -m detector_service
```

- Binds exclusively to `127.0.0.1:8000` (loopback only).
- Generates a local authentication token saved with `0600` permissions at `data/processed/service_token`.
- Enforces Host header validation to mitigate DNS rebinding attacks.
- Limits concurrent scans using an asynchronous semaphore.

### 2. Load the Firefox WebExtension

1. Open Firefox and go to `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on...** and select `extension/manifest.json`.
3. Open the extension Options page (`about:addons` or popup gear icon).
4. Paste your local bearer token (`cat data/processed/service_token`) and click **Save Settings**.
5. As you browse, visited pages are scanned locally. Suspicious or phishing sites trigger a visual toolbar badge and redirect to an interstitial warning page (`warning/warning.html`) with risk scores and detected reasons.

### 3. Build Deterministic Extension Release

To package the Firefox WebExtension into a reproducible zip archive with an authoritative SHA-256 checksum:

```bash
python3 scripts/package_extension.py
```

This creates:
- `dist/phishing-url-detector-extension-0.5.0.zip`
- `dist/phishing-url-detector-extension-0.5.0.zip.sha256`

Verify the archive checksum:

```bash
cd dist && sha256sum -c phishing-url-detector-extension-0.5.0.zip.sha256 && cd ..
```

See [`extension/README.md`](extension/README.md) for full architecture details, threat models, testing instructions, and packaging options.
See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) and [`SECURITY.md`](SECURITY.md) for release engineering and vulnerability policies.

---

# Layer 1 — Lexical analysis

Parses a URL and extracts **41 deterministic, network-free features**. Layer 1
makes no decision; it returns features and later layers weigh them.

```python
from detector import extract_lexical_features
f = extract_lexical_features("https://paypal.security-check.xyz/login")
f.brand_in_wrong_place        # True  — 'paypal' is in the subdomain
f.registered_domain           # 'security-check.xyz'
```

The only dependency is [`tldextract`](https://github.com/john-kurkowski/tldextract)
for **Public Suffix List–aware** registered-domain extraction, run from its
bundled PSL snapshot (`suffix_list_urls=()`) — offline, deterministic, and never
splitting domains on `.` by hand.

### Feature groups

| Group | Features |
|---|---|
| **Shape** | `url_length`, `domain_length`, `registered_domain_length`, `subdomain`, `subdomain_depth`, `dot_count`, `hyphen_count`, `digit_count`, `special_char_count`, `path_length`, `query_length`, `fragment_length` |
| **Red flags** | `has_ip`, `ip_version`, `has_at_symbol`, `has_double_slash`, `has_port`, `port`, `is_suspicious_port` |
| **Keywords** | `keyword_count`, `keywords_found`, plus one `keyword_<name>` flag per configured word (`login`, `verify`, `secure`, `account`, `update`, `confirm`, `wallet`, `kyc`, `signin`, `password`, `payment`) |
| **Brand** | `brand_in_wrong_place`, `brand_in_wrong_place_name` — a brand appears in the subdomain/path but the registered-domain label is something else |
| **Typosquatting** | `levenshtein_distance`, `closest_brand`, `is_typosquat_suspicious`, `is_exact_brand_match` (hand-rolled edit distance against the brand list) |
| **Homograph** | `is_punycode`, `contains_non_ascii` |
| **Entropy** | `entropy` — Shannon bits/char of the registered domain (`google.com` ≈ 2.6, `a8f3k2j9x7q1z5b4.top` ≈ 4.0) |
| **TLD risk** | `tld`, `is_suspicious_tld` (`zip top tk xyz click gq ml cf`) |

### Two semantics worth knowing

**`has_at_symbol` means "`@` in the authority", not "`@` anywhere."**
`http://paypal.com@evil.com/` really loads `evil.com`, and that is the signal.
Scanning the whole URL string made every `?email=user@gmail.com` link raise the
flag — a large false-positive source. (This changed in feature schema v2.)

**`registered_domain` falls back to the whole hostname** when there is no
recognised public suffix (`build.ci.internal`, a typo'd TLD). Collapsing to the
last label would make `a.internal` and `b.internal` share a Layer 2 cache entry
and look same-origin to the cross-domain form check.

Malformed input never raises: it yields `is_malformed=True` with the other
fields best-effort.

---

# Layer 2 — DNS / domain intelligence

Takes the hostname and registered domain Layer 1 already extracted (it does not
re-parse) and gathers domain-level intelligence over the network. **Any DNS /
RDAP / WHOIS / TLS failure becomes a feature, never an exception, and never a
phishing decision.**

```python
from detector import analyze_domain
intel = analyze_domain("https://example.com/login", do_tls=True)
```

| Probe | Features |
|---|---|
| **DNS** (A/AAAA/NS/MX) | `dns_resolves`, `ipv4_count`, `ipv6_count`, `has_ns`, `ns_count`, `nameservers`, `has_mx`, `mx_count`, `mx_hosts`, `ttl`, `dns_errors` |
| **Domain age** (RDAP → WHOIS) | `domain_age_available`, `domain_age_source`, `creation_date`, `expiration_date`, `registrar`, `domain_age_days`, `is_new_domain` |
| **IP intelligence** (local only) | `public_ip_count`, `private_ip_count`, `reserved_ip_count`, `loopback_ip_count`, `multicast_ip_count`, `ip_details` |
| **Reverse DNS** (public IPs only) | `reverse_dns_available`, `reverse_dns_names` |
| **TLS** (opt-in, `do_tls=True`) | `tls_available`, `tls_cert_valid`, `tls_issuer`, `tls_not_after`, `tls_days_until_expiry`, `tls_cert_age_days`, `tls_hostname_matches`, `tls_is_self_signed`, `tls_version`, `tls_error` |

**Bounded fan-out.** Exactly four record queries per domain (A, AAAA, NS, MX)
plus at most `reverse_dns_max_lookups` (default 3) PTR queries, each with an
explicit timeout (`dns_timeout=3 s` per query, `dns_lifetime=6 s` total). No
retry storms, no recursion into discovered NS/MX names, no port scans.

**Rate limiting.** `DetectorConfig.domain_lookup_min_interval` (default 0)
enforces a minimum spacing between outbound lookups for the *same* domain, so a
bulk run over one campaign's thousand URLs cannot hammer a registry's RDAP
endpoint. The cache removes most of that traffic; this bounds what is left.

**Bounded total latency.** The probes run in sequence, so per-probe timeouts
multiply rather than bound the total: four DNS queries at 6 s, then RDAP, then
WHOIS, then TLS could add up to well over a minute on an unresponsive network.
Four things keep that in check, none of which removes a check:

- **NS/MX are skipped** when the same name already came back `NXDOMAIN` (the
  name does not exist, so those records cannot either), or when A *and* AAAA
  both timed out (a resolver that ignored two queries will ignore two more).
  A subdomain's `NXDOMAIN` still leaves the registered domain queried.
- **TLS is skipped when DNS produced no address** - a handshake needs one, and
  failing costs a resolution plus up to three connection attempts. Measured:
  this alone turned a 25 s analysis into 106 s.
- **`DetectorConfig.layer2_budget`** (default 20 s) is a wall-clock deadline
  for all of Layer 2; remaining probes are skipped and say so in `notes`.
- **Name resolution is hard-capped** at `security.RESOLVE_TIMEOUT` (5 s).
  `socket.getaddrinfo` is a blocking C call with no timeout parameter, so it
  runs on a daemon thread that we stop waiting for.

**TLS goes through the same SSRF guard as Layer 4**, so it cannot be used to
probe internal TLS services. The handshake is attempted in three tiers — full
verification, chain-only (to read a certificate presented for the wrong
hostname), then unverified (to record that TLS answered at all).

**RDAP bootstrap is cached to disk** (`data/processed/rdap_bootstrap.json`,
refreshed weekly). Without it, every process re-downloaded the IANA bootstrap
registry before its first RDAP query.

### What Layer 2 signals do and do not mean

- **A snapshot, not history.** DNS answers reflect this moment; fast-flux and
  round-robin change between calls. Passive-DNS history is out of scope.
- **RDAP/WHOIS is unreliable by nature.** Many ccTLDs expose no creation date,
  redact everything, or rate-limit. `domain_age_available=false` is normal, and
  unknown age is **never** treated as "new" or as "phishing".
- **CDNs muddy every IP/TTL signal.** A domain behind Cloudflare shares IPs and
  low TTLs with millions of benign sites.
- **Free certificates are the norm on both sides.** A Let's Encrypt certificate
  issued yesterday is as common on a new small business as on a phishing kit.

---

# Layer 3 — Machine learning

Takes the **65 numeric/boolean features** from Layers 1 + 2 (41 lexical, 24
DNS/domain) and predicts `P(phishing) ∈ [0, 1]` with a **Random Forest**. The
model never sees a URL string — only the feature vector.

### Why Random Forest

The input is a fixed-width row of hand-engineered numbers — a classic tabular
classification problem, not text modelling (Layer 1 already did the string
work). Random Forest is robust to feature scale and irrelevant features, hard
to catastrophically overfit, needs no scaling/encoding pipeline, trains in
seconds, and exposes `feature_importances_`. `model.py::_build_estimator` is
the single place the algorithm is chosen, so XGBoost/LightGBM can be slotted in
later without touching anything else.

### The feature schema is authoritative and versioned

`feature_schema.py` owns the feature names, their order, their sentinels and
the schema hash. Training, inference and both caches import from it.

Two guards, both **fatal**:

- **`SCHEMA_VERSION`** is bumped by hand when a feature's *meaning* changes.
  A model or cache row carrying a different version is refused.
- **`schema_hash()`** fingerprints the exact ordered list, so an accidental
  reordering is caught even when the version was not bumped.

`PhishingModel.load()` validates the saved schema against **the current code's**
schema, not merely against its own copy:

```
$ python -m detector analyze https://example.com/
SchemaMismatch: current feature pipeline vs saved model: schema version
mismatch - expected v2, got v1. The feature definitions changed; retrain the
model (python -m detector train --sample) or clear the cache.
```

This is the difference between a loud failure and confident nonsense: a model
scored against a reordered or differently-meaning vector does not error, it
just returns wrong answers.

### Dataset preparation

Phishing and benign URLs come from public lists. **The files are not shipped —
you download them.** Each is normalised to `url,label,source,date`:

| Class | Source | `kind=` | Per-row timestamp? |
|---|---|---|---|
| phishing | **PhishTank** (verified CSV) | `phishtank` | yes (`submission_time`) |
| phishing | **OpenPhish** (feed, one URL/line) | `openphish` | **no** |
| benign | **Tranco** (`rank,domain`) | `tranco` | no (whole-list date only) |
| benign | **Majestic Million** | `majestic` | no (whole-list date only) |

```json
[
  {"kind": "phishtank", "path": "data/raw/phishtank/verified_online.csv"},
  {"kind": "openphish", "path": "data/raw/openphish/feed.txt"},
  {"kind": "tranco",    "path": "data/raw/tranco/top-1m.csv",    "list_date": "2026-06-01", "limit": 40000},
  {"kind": "majestic",  "path": "data/raw/majestic/majestic_million.csv", "list_date": "2026-06-01", "limit": 40000}
]
```

```bash
python -m detector train --spec data/raw/sources.json
```

> **Timestamp limitation.** Only PhishTank carries a real per-row date. The
> pipeline **does not invent timestamps** — undated rows go to the training
> bucket by default (`MLConfig.undated_bucket`, also `"exclude"` /`"grouped"`).
> With mostly-undated data the trainer falls back to a domain-grouped split.

### Dataset bias — the domain-age trap

**A naive phishing corpus teaches the model the wrong thing.** If phishing URLs
are scraped fresh (domains registered days ago) and benign URLs are the Tranco
top 10k (domains registered a decade ago), `domain_age_days` alone separates the
classes and the model learns *"old = safe"*. It will then wave through a
phishing page on a **compromised 10-year-old domain** and flag a **legitimate
brand-new small business**.

Mitigations built in:

- **Compose the benign set deliberately** — newer domains, obscure sites, small
  orgs, deep paths, query strings, varied TLDs. Include phishing on **old**
  domains, plain `.com`, and short URLs.
- **Domain-age features are isolated** (`l2_domain_age_available`,
  `l2_domain_age_days`, `l2_is_new_domain`) and can be dropped wholesale with
  `--no-domain-age-features`. Train both ways: if metrics collapse without age,
  your dataset is age-biased.
- **Every run reports domain-age reliance** — the summed importance of the age
  features, printed and saved to `training_metadata.json`, with a **WARNING**
  above `MLConfig.domain_age_reliance_warn` (0.35).
- Unknown age is `-1` with an availability flag, so "we couldn't check" (common
  for obscure *benign* domains) is distinct from "young".

### Leakage prevention

1. **Exact-duplicate URLs removed** before splitting.
2. **Per-registered-domain cap** (default 50), grouped by domain **alone** —
   capping per `(domain, label)` pair would let one domain contribute 50 benign
   *plus* 50 phishing rows, twice the documented limit.
3. **Time-based split**: older data trains, newer data tests — the realistic
   setting, since you predict *future* URLs.
4. **No registered domain spans train and test** — after the time cut, any
   domain already in train is removed from val/test, and val/test are kept
   disjoint too.
5. **Grouped fallback**: if timestamps cannot produce a non-empty train *and*
   test, a deterministic seeded split by registered domain is used instead.
6. **Per-source metrics** are reported (see below) so *source-artifact* leakage
   is visible, not just documented.

### Evaluation

Accuracy is **not** the headline and is labelled "NOT the primary metric" in
the output — a "predict benign" stub scores >90% accuracy on an imbalanced
corpus while catching zero phishing. Reported: **precision, recall, F1,
confusion matrix, ROC-AUC, PR-AUC**, a threshold sweep, and a per-source
breakdown.

```
                     Predicted
                 Benign    Phishing
Actual Benign      TN         FP
Actual Phishing    FN         TP
```

* **Precision** = TP/(TP+FP) — of what we flagged, how much was really phishing.
  Low precision ⇒ false positives ⇒ legitimate sites blocked, users learn to
  click through warnings.
* **Recall** = TP/(TP+FN) — of the real phishing, how much we caught.
  Low recall ⇒ false negatives ⇒ users handed to the attacker.
* **The trade-off is a policy choice, not a modelling one.** A browser warning
  wants high precision; a SOC triage queue wants high recall. Layer 3 exposes
  the knob; `DecisionConfig` sets the policy.

Threshold tuning is swept over `{0.3 … 0.9}` **on validation, never on test**.

```bash
python -m detector evaluate --csv data/raw/sample/sample_dataset.csv --no-layer2
python -m detector evaluate --csv data.csv --json          # for automation
```

### Why "99% accuracy" is not a good phishing detector

A model can hit 99% accuracy by learning **dataset artifacts**:

* **domain age / popularity** (the trap above);
* **source signatures** — PhishTank URLs are long and messy, Tranco entries are
  bare `http://domain/`; a model can separate them on `url_length` alone and
  never look at anything phish-specific;
* **TLD skew** — if every `.tk` in your corpus happens to be phishing;
* **collection-time skew** — phishing scraped this week, benign from last year.

The per-source table makes this concrete. On the bundled synthetic sample:

```
  Per-source breakdown (a large gap here suggests the model is separating FEEDS, not phishing):
  source         |      n |    pos |  mean P |    acc | ROC-AUC
  -------------------------------------------------------------
  sample_benign  |    340 |      0 |  0.0428 | 1.0000 |     n/a
  sample_phish   |    260 |    260 |  0.9564 | 0.9962 |     n/a
```

Each source is **single-class** (`ROC-AUC: n/a`), so the reported ~1.0 precision
and recall measure the *generator*, not phishing. Treat the `--sample` numbers
as a **pipeline smoke test only**. On real data expect meaningfully lower, more
honest numbers — and read the PR-AUC and the confusion matrix at your chosen
threshold, not the accuracy.

### Explainability caveat

`top_features` is the model's **dataset-global, impurity-based**
`feature_importances_` intersected with this URL's active features. It is a
*hint*, not a faithful per-prediction attribution. For real per-prediction
explanations: `pip install ".[explain]"` and use SHAP on `model.estimator`.

---

# Layer 4 — Page content + LLM

The expensive, highest-signal layer. It runs **only** for URLs the decision
engine could not resolve from Layers 1–3, and it has four stages:

| Stage | Module | Job |
|---|---|---|
| 4a | `web_fetcher.py` | SSRF-guarded, bounded GET of the page |
| 4b | `html_analyzer.py` | static parse: text, title, iframes, hidden content, scripts |
| 4c | `form_analyzer.py` | what each form collects, and which domain it posts to |
| 4d | `llm_analyzer.py` | LLM judgement over the fenced page content |

**Static facts are computed before the LLM sees anything**, and the decision
engine checks them *first*. "This login form posts to a different domain" is
read from an `action` attribute by code — a page full of text telling an LLM
otherwise cannot change it.

### 4a — Fetching a hostile page

This is the component an attacker gets to point wherever they like, so every
default is a restriction:

- **GET only.** No other method is ever issued, **no form is ever submitted**.
- **No cookies, no credentials, no ambient proxy config.** A fresh session with
  a cookie policy that refuses to store or send anything, and
  `trust_env = False`, so no `.netrc`/proxy credential is carried to a phishing
  page and a page cannot set state that leaks into the next fetch.
- **TLS verification always on.** A verification failure is recorded as a
  feature, never worked around.
- **Bounded**: connect (5 s) and read (8 s) timeouts, a 20 s wall-clock ceiling
  across all hops, 3 redirects, a 1 MB **decoded**-body cap and a 20,000-char
  text cap, plus a `Content-Type` allowlist (`text/html`, XHTML, `text/plain`).
- **Nothing is executed.** No JavaScript engine, no subprocess, no file written.

The body is read incrementally through `iter_content`, which decodes
`Content-Encoding` as it goes, so the cap applies to *decompressed* bytes.
Reading the raw stream instead would cap the *compressed* size — letting a small
gzip expand without limit, and producing binary garbage for any compressed page.

### 4b/4c — What static analysis extracts

| Signal | Detail |
|---|---|
| visible text, `<title>` | `<script>`/`<style>`/hidden content excluded |
| **hidden content** | `display:none`, `visibility:hidden`, `opacity:0`, `font-size:0`, zero-size, off-screen (`left:-9999px`), `hidden`, `aria-hidden` — counted, and the hidden *text* kept separately |
| **forms** | action (resolved against `<base>`), method, and every field |
| **field purpose** | password, one-time code, payment card, bank details, identity document, wallet seed phrase, username — from `type`, `name`, `id`, `autocomplete`, `placeholder`, `inputmode`, `pattern` and ARIA label **together** |
| **cross-domain post** | compared at the *registered domain* level, so `login.x.com → api.x.com` is fine but `paypal-login.com → collector.ru` is not |
| iframes | split into same-site and cross-site |
| scripts | count, inline size, external sources, cross-site sources, obfuscation markers (`eval(`, `atob(`, `fromCharCode`, long hex/percent/base64 blobs) |
| redirects | `<meta http-equiv=refresh>` targets, and the HTTP redirect chain |

Field classification is deliberately multi-signal because phishing kits use
`type="text"` with `name="pwd"` to dodge naive `type="password"` checks, and
un-named 6-digit numeric boxes for OTPs. **No field value is ever read, stored
or logged** — only the shape of the form.

### 4d — LLM analysis

The LLM is shown two kinds of material, kept strictly apart:

1. **Verified facts** computed by code (the URL, registered domain, DNS results,
   what each form collects and where it posts). A page cannot argue with these.
2. **Untrusted page data** — title, visible text, hidden text — fenced inside a
   per-request random nonce.

It must return exactly one JSON object with exactly these keys:

```json
{
  "brand_claimed": "PayPal",
  "brand_impersonation": true,
  "social_engineering": true,
  "social_engineering_tactics": ["urgency: 24h account closure"],
  "credential_request": true,
  "page_purpose": "login",
  "suspicious_observations": ["form posts to an unrelated domain"],
  "prompt_injection_attempt": false,
  "risk_level": "high",
  "confidence": "high"
}
```

**There is no free-text `reasoning` field**, by design — the output carries
concise labels, not chain-of-thought. Anything longer is truncated, and an
unexpected field is rejected.

**`llm_likelihood` is derived in code** from the `(risk_level, confidence)` pair
via a fixed table. The model picks a bucket; code assigns the number. A
manipulated response cannot inject an out-of-range score.

### 4e — Providers: local Ollama (default) or Anthropic

`llm_analyzer.py` never names a provider. It calls one method —
`complete(system, prompt, max_tokens)` — on whatever `LLMClient`
`llm_client.create_llm_client()` hands it, chosen from `LLMConfig.provider`:

| `provider` | Client | Model default | Needs | Cost |
|---|---|---|---|---|
| `ollama` *(default)* | `OllamaLLMClient` | `gemma3:4b` | a local Ollama server | none |
| `anthropic` | `AnthropicLLMClient` | `claude-haiku-4-5-…` | `pip install ".[llm]"` + `ANTHROPIC_API_KEY` | per call |

**Local inference is the default because it costs nothing per call and no page
content leaves the machine.** The trade-off is quality and latency: a 4B model
is a weaker analyst than a hosted one, and a call takes tens of seconds on CPU
rather than a second or two — hence the 120 s default timeout for `ollama`
against 20 s for `anthropic`.

Both providers produce **the same JSON schema**, validated by the same code.
There is no second, looser contract for the local model: a local model is no
more trustworthy about a hostile page than a hosted one. `OllamaLLMClient` asks
Ollama for grammar-constrained JSON (`"format": "json"`), which guarantees the
reply *parses* — it guarantees nothing about the schema, so every field still
goes through `validate_response()`.

`OllamaLLMClient` talks to `POST {base_url}/api/chat` with `requests`, which is
already a core dependency for the page fetch — **the local provider adds no new
dependency**, and the `ollama` Python package is not required.

#### Running Ollama

```bash
# 1. install (Linux/macOS; see ollama.com/download for other platforms)
curl -fsSL https://ollama.com/install.sh | sh

# 2. start the server (a systemd/launchd service usually does this already)
ollama serve

# 3. pull and verify the model
ollama pull gemma3:4b
ollama list              # expect a gemma3:4b row
ollama run gemma3:4b     # optional: an interactive smoke test

# 4. analyse
python -m detector analyze "https://example.com"
```

When Layer 4 runs, the CLI names the backend it used:

```
LAYER 4  page content + LLM
  fetch .............. 200 (0 redirect(s), 6 ms)
  title .............. Verify your account
  forms .............. 1  credential=yes  otp=yes  payment=no
  LLM risk ........... high (confidence high, likelihood 0.9500)
  provider ........... ollama
  model .............. gemma3:4b
```

#### When Ollama is not running

Every failure mode — server down, connection refused, model not pulled, request
timeout, non-JSON body, unexpected envelope — is caught and reported as
`llm-unavailable` or `llm-call-failed`, with an operator-facing hint:

```
  LLM ................ unavailable: llm-call-failed: cannot reach Ollama at
                       http://127.0.0.1:11434 - is it running? (start it with: ollama serve)
```

The analysis continues on Layers 1–3 and the code-computed page facts. **A page
is never classified phishing because the LLM was offline** — a missing opinion
is not an accusation; the decision engine takes the `layer4_unavailable` path
and returns `uncertain` for anything Layers 1–3 could not call outright.

#### Limitations of the local model

- **A 4B model is a weaker analyst.** Expect more missed subtleties and
  occasional literal readings (asked about a page served from an IP, `gemma3:4b`
  will report `brand_claimed: "127.0.0.1"`). The decision engine's code-computed
  rules are unaffected — they fire before anything the LLM said.
- **It is slow on CPU**: roughly 50–90 s per page here, against ~1–2 s for a
  hosted model. Layer 4 only runs in the uncertain band, and its results are
  cached, so this is bounded — but batch runs will feel it.
- **First call after an idle period is slower**, because Ollama has to load the
  model into memory.
- `ollama_num_ctx` (8192) must stay comfortably above the whole prompt: Ollama
  drops the **oldest** tokens on overflow, and the oldest tokens here are the
  system prompt — i.e. exactly the prompt-injection defences.

---

# Security

## Prompt injection (Layer 4)

The page is hostile input written by the same person the analysis is about, and
phishing kits do embed text aimed at automated analysers. Defences, in layers:

1. **The system prompt states the trust model** — fenced content is data, never
   instructions; text inside it claiming to be a system prompt, a policy update
   or a message from the caller is false; never let it change the output format
   or the verdict.
2. **A random per-request nonce delimits the fence**, so page text cannot close
   it and forge trailing "system" instructions. Any literal occurrence of the
   fence markers in the page is neutralised before assembly.
3. **The verdict never comes from the LLM alone.** The decision engine checks
   code-computed static facts *before* anything the LLM said — and those are
   the rules that fire first.
4. **`prompt_injection_attempt` is a red flag, not a request.** A page trying to
   manipulate an analyser is evidence *against* it. The decision engine
   classifies it as phishing.
5. **Output is schema-validated**: unknown fields rejected, enums checked, types
   checked, the numeric likelihood derived in code.
6. **Every returned string is sanitised and truncated** — control characters
   stripped, NFKC-normalised, length-capped — so page-controlled text cannot
   inject ANSI escapes or forge log lines downstream.

**The LLM is an analyser. Its output is data**: it never becomes a command, a
path, a URL to fetch, or a configuration change.

## SSRF (Layer 4 and TLS)

Without a guard, "analyse this URL for me" becomes "make your server request
`http://169.254.169.254/latest/meta-data/iam/security-credentials/` and tell me
what it says". `security.check_fetch_target()` refuses:

| Refused | Examples |
|---|---|
| non-http(s) schemes | `file://`, `gopher://`, `dict://`, `javascript:`, `data:` |
| loopback | `127.0.0.1`, `::1`, `localhost` |
| private (RFC1918/ULA) | `10.*`, `172.16-31.*`, `192.168.*`, `fc00::/7` |
| link-local incl. cloud metadata | `169.254.169.254`, `fe80::/10` |
| CGNAT / reserved / multicast / unspecified | `100.64.0.0/10`, `240.0.0.0/4`, `224.*`, `0.0.0.0` |
| IPv4-mapped IPv6 wrapping any of the above | `::ffff:127.0.0.1` |
| single-label hostnames | `http://jenkins/`, `http://intranet/` |
| named metadata hosts | `metadata.google.internal` |

**Every resolved address must be public** — a host resolving to one public and
one private address is refused, since we cannot control which a later
connection picks.

**The check runs on every redirect hop.** Redirects are followed manually
precisely so that hop 1 being public says nothing about hop 2:

```python
>>> fetch_page("http://169.254.169.254/latest/meta-data/")
PageContent(fetched=False,
            blocked_reason="host 169.254.169.254 resolves to non-public address(es)")
# ...and zero HTTP requests were issued.
```

A refusal is a **policy outcome**, not evidence: a blocked fetch never counts
toward a phishing verdict.

**DNS rebinding mitigation — connect-time IP pinning.** The fetcher defeats
DNS rebinding (TOCTOU race conditions between validation and socket connection)
by using `PinnedHTTPAdapter`. When `check_fetch_target` validates an address,
the TCP socket in `PinnedHTTPConnection` / `PinnedHTTPSConnection` is bound
directly to the validated IP, bypassing connect-time DNS resolution while
preserving the requested hostname for the HTTP `Host` header, TLS Server Name
Indication (SNI), and TLS certificate hostname verification. Ambient proxies
via `trust_env` are disabled and proxy configurations are refused to ensure
pinning cannot be bypassed.

For hostile-input production deployments, running the fetcher in a network-isolated
sandbox (egress firewall restricted to public ranges) remains recommended as
defense-in-depth.

`allow_private_targets=True` disables the address checks for controlled test
environments. The scheme allowlist still applies. **Never enable it in
production.**

## Secrets

- **The default provider needs no secret at all.** Local Ollama is
  unauthenticated loopback HTTP; nothing is read from the environment for it,
  and the endpoint is operator configuration that no fetched page can influence.
- When the Anthropic provider is selected, its API key is read from the
  environment named by `LLMConfig.api_key_env` (default `ANTHROPIC_API_KEY`) and
  **never stored in a printable field** — `AnthropicLLMClient` defines
  `__repr__` explicitly so a traceback or log line cannot leak it.
- **All logging passes through a redaction filter**: API keys, credential-ish
  query parameters (`password`, `token`, `otp`, `session`, `cvv`, …), and
  forbidden fields (`cookie`, `authorization`, `form_values`) are scrubbed
  regardless of what a call site passes.
- **URLs are redacted in output too.** Phishing URLs routinely carry the
  victim's email or a session token in the query string; `result.as_dict()`
  redacts them, including the Layer 1 echo.
- The Layer 4 cache is keyed by a **SHA-256 of the URL**, so raw URLs are not
  written to the cache file.
- `.gitignore` covers `.env`, `*.key`, `*.pem`, both cache databases and the
  RDAP bootstrap file.

## Other properties

- **No shell execution, no subprocess, no `eval`** anywhere in the codebase.
- **No file is written from fetched content.**
- Bounded everything: DNS fan-out, response size, redirect count, LLM tokens,
  reverse-DNS lookups, log field sizes.

---

# The decision engine

```
1. blocklist      -> phishing       (policy, no further work)
2. allowlist      -> safe           (policy, no further work)
3. ML >= HIGH     -> phishing       (Layer 4 skipped: already confident)
4. ML <= LOW      -> safe           (Layer 4 skipped: already confident)
5. otherwise      -> run Layer 4, then combine by explicit ordered rules
```

### Scores are not averaged

Averaging a model probability with an LLM's self-rating produces a number that
means nothing: the two are on different scales, measure different things, and
one of them is influenced by attacker-controlled text. Step 5 applies **ordered
rules over discrete findings**, and the final `risk_score` is derived from
*which rule fired*, clamped monotonically against the model's probability.

| # | Rule (first match wins) | Verdict | `risk_score` |
|---|---|---|---|
| R1 | static: credential form posts **cross-domain** | phishing | `max(ml, 0.90)` |
| R2 | static: page requests a **wallet seed phrase** | phishing | `max(ml, 0.90)` |
| R3 | static: credential form posts over **plain HTTP** | phishing | `max(ml, 0.80)` |
| — | Layer 4 unavailable | uncertain | `ml` |
| R4 | LLM found a **prompt-injection attempt** | phishing | `max(ml, 0.80)` |
| R5 | **brand impersonation + credential request** | phishing | `max(ml, 0.80)` |
| R6 | LLM rated **high risk** | phishing | `max(ml, 0.70)` |
| R7 | brand impersonation, no credential ask | uncertain | `max(ml, 0.50)` |
| R8 | LLM rated **low risk**, no corroborating static flags | safe | `min(ml, 0.20)` |
| R9 | still ambiguous | uncertain | `ml` |

Rules R1–R3 are **code-computed** and are checked before anything the LLM said,
because static facts cannot be talked out of by the page's text while an LLM's
judgement can be attempted.

### Three scores, kept separate

| Field | What it is |
|---|---|
| `ml_probability` | Random Forest raw output. **Uncalibrated.** |
| `llm_likelihood` | Fixed mapping from the LLM's `(risk_level, confidence)` enums. A self-report, not a measurement. |
| `risk_score` | **Ordinal severity**, derived from which rule fired. |

**`risk_score` is not a calibrated probability.** No calibration (Platt scaling,
isotonic regression, reliability-curve fitting) has been performed on this
model, so 0.8 does *not* mean "80% of URLs scoring 0.8 are phishing". It orders
severity and nothing more, and both the JSON output and the CLI say so.

### Explanations

Every reason is generated from a feature that was actually measured, and its
message quotes the measurement:

```
!! [high  ] A form collecting username/email, password submits to
            'collector-site.ru', a different domain from the page's own
            'paypal-secure-login.com'
!! [high  ] Domain was registered 4 day(s) ago
 ! [medium] Page hides 312 characters of text from the visible rendering
   [info  ] URL-feature model scored 0.52 probability of phishing
```

Each carries a stable machine `code` (`l4_cross_domain_credential_post`,
`l2_domain_new`, …), a `severity`, the `layer` that observed it, and its
`source` (`rule` / `model` / `llm` / `policy`) — so LLM-derived claims are
always distinguishable from code-computed facts, and downstream tooling never
has to string-match prose. **Nothing here invents evidence.**

---

# Caching

Two caches, separate on purpose:

| | Layer 2 cache | Layer 4 cache |
|---|---|---|
| Keyed by | **registered domain** | **SHA-256 of the URL** |
| Why | DNS/RDAP facts belong to a domain and are shared by every URL on it | page content belongs to one URL |
| Default TTL | 24 h positive / **5 min negative** (0 = never expire, used for reproducible training) | 6 h |
| Stores | full `DomainIntelligence` (lossless round-trip) | completed `LLMAnalysis` only |
| File | `data/processed/layer2_cache.sqlite` | `data/processed/page_cache.sqlite` |

Both carry a **version column**; a row written by a different version is ignored
rather than reinterpreted against a changed schema. Both **never cache a
failure** — one transient DNS blip or LLM outage would otherwise be pinned for
the whole TTL. Layer 4's TTL is deliberately short: phishing pages are taken
down, re-pointed and cloaked constantly, so a stale "safe" verdict is worse than
a cache miss.

Disable with `--no-cache`, `DETECTOR_CACHE_ENABLED=0`, or
`CacheConfig(layer2_enabled=False, page_enabled=False)`.

---

# Configuration

Every tunable lives in `config.py`. Nothing outside it hard-codes a keyword,
brand, timeout, threshold, size limit or model name.

```python
from detector import (
    DetectorConfig, MLConfig, DecisionConfig,
    NetworkConfig, LLMConfig, CacheConfig, analyze_url,
)

cfg = DetectorConfig(                      # Layers 1-2
    brands=("paypal", "mybank", "acme"),
    suspicious_tlds=("xyz", "top", "lol"),
    dns_timeout=2.0, new_domain_max_age_days=45,
    dns_nameservers=("1.1.1.1", "8.8.8.8"),      # pin the resolver
    domain_lookup_min_interval=0.5,               # rate-limit per domain
    layer2_budget=20.0,                           # wall-clock cap for ALL of Layer 2
)

ml = MLConfig(                             # Layer 3
    n_estimators=400, max_depth=24, class_weight="balanced",
    train_end="2026-02-01", val_end="2026-06-01",
    max_urls_per_registered_domain=25,
    include_domain_age_features=False,            # the leakage experiment
    random_seed=42,
)

decision = DecisionConfig(                 # thresholds + Layer 4 policy
    low_threshold=0.25, high_threshold=0.75,
    blocklist_domains=("known-bad.com",),
    allowlist_domains=("our-intranet.com",),
    network=NetworkConfig(max_redirects=2, max_response_bytes=500_000),
    llm=LLMConfig(provider="ollama", model="gemma3:4b", max_tokens=1024),
    cache=CacheConfig(page_ttl_seconds=1800,
                      layer2_negative_ttl_seconds=300),
)

result = analyze_url(url, config=cfg, ml_config=ml, decision_config=decision)
```

A `DecisionConfig` with inverted or out-of-range thresholds raises `ConfigError`
at construction.

### Environment overrides

```bash
export ANTHROPIC_API_KEY=sk-...          # the ONLY secret; never in config
                                         # (not needed for the default provider)
export DETECTOR_LOW_THRESHOLD=0.25
export DETECTOR_HIGH_THRESHOLD=0.75
export DETECTOR_ENABLE_LAYER4=0
export DETECTOR_BLOCKLIST=bad1.com,bad2.com
export DETECTOR_ALLOWLIST=intranet.example.com
export LLM_PROVIDER=ollama               # or: anthropic
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export DETECTOR_LLM_MODEL=gemma3:4b      # a model name for THAT provider
export DETECTOR_LLM_TIMEOUT=120
export DETECTOR_FETCH_MAX_BYTES=2000000
export DETECTOR_FETCH_MAX_REDIRECTS=2
export DETECTOR_ALLOW_PRIVATE_TARGETS=0   # lab only, never in production
export DETECTOR_CACHE_ENABLED=1
```

Applied via `decision_config_from_env()`, which the CLI calls for `analyze`.

### LLM configuration

**Default — local Ollama.** Nothing to configure beyond a running server and a
pulled model:

```bash
ollama list                          # expect: gemma3:4b
python -m detector analyze "https://example.com"
```

`LLM_PROVIDER`, `OLLAMA_BASE_URL` and `DETECTOR_LLM_MODEL` override the
endpoint and model; `DETECTOR_LLM_PROVIDER` and `DETECTOR_OLLAMA_BASE_URL` are
accepted as project-prefixed synonyms and win if both are set.

**Alternative — Anthropic.** Still fully supported; it needs **both**:

```bash
pip install ".[llm]"                 # the anthropic SDK - an optional extra
export ANTHROPIC_API_KEY=sk-...
export LLM_PROVIDER=anthropic        # the model defaults to claude-haiku-4-5
```

Switching provider switches the default model with it, because a model name is
meaningless across providers — asking Anthropic for `gemma3:4b` is an error, not
a fallback. Set `DETECTOR_LLM_MODEL` to override.

If the selected provider cannot run — no API key, SDK not installed, Ollama not
started, model not pulled, an unknown provider name — Layer 4 reports
`llm-unavailable` / `llm-call-failed` and the decision engine returns
`"uncertain"` for anything Layers 1–3 could not call outright. **That is the
correct degradation** — never a guess, never a silent "safe".

A further provider can be added by implementing the two-method `LLMClient`
protocol (`available`, `complete`) and registering it in
`llm_client._PROVIDERS`, or by passing an instance as `llm_client=`.

---

# Logging

The detector is a library, so it never configures the root logger on import. It
emits structured records on the `detector.*` hierarchy:

```bash
python -m detector --log-level INFO --log-json analyze "https://example.com/"
```

```json
{"ts":"2026-09-12T14:02:11","level":"INFO","logger":"detector.pipeline",
 "message":"analysis complete","analysis_id":"3f9a21c4de77",
 "classification":"safe","risk_score":0.28,"decision_path":"ml_low",
 "layer4_ran":false,"total_ms":94.2}
```

Every analysis carries an `analysis_id` so Layer 1/2/3/4 records for one URL can
be correlated, and every record passes through the redaction filter. Logged:
analysis id, layer, timing, failures, final result. **Never logged:** passwords,
OTPs, cookies, authorization headers, API keys, or any submitted form value.

Per-layer timings are on every result (`result.timings_ms`).

---

# Running the tests

```bash
pytest                       # 563 Python tests, ~10 s
ruff check src/ tests/ scripts/ # static analysis
node extension/tests/run_tests.js # 26 Firefox extension tests
python3 scripts/security_audit.py # offline repository security audit
```

**Nothing in the suite touches the network, a real LLM, or an API key.** Every
layer that does I/O has an injection point, and `tests/conftest.py` supplies the
doubles: `FakeResolver` (DNS), `FakeHttpClient` (fetch), `FakeLLM`,
`FakeModel`/`FakePredictor`.

| File | Covers |
|---|---|
| `test_lexical_features.py` | normal/malformed/IP/`@`/punycode/unicode URLs, typosquatting, `.co.uk`, entropy, brand mismatch, determinism |
| `test_dns_features.py` | DNS success/NXDOMAIN/timeout, IP classification, reverse-DNS capping |
| `test_domain_intelligence.py` | domain age, RDAP→WHOIS fallback, L1→L2 pipeline |
| `test_tls_features.py` | SSRF guard on TLS, hostname/wildcard matching, expired/self-signed certs |
| `test_dataset.py` | loaders, cleaning, dedup, per-domain cap, time/grouped splits |
| `test_feature_pipeline.py` | vector construction, ordering, missing-L2 sentinels |
| `test_feature_schema.py` | schema hash, order sensitivity, version mismatch, sentinels |
| `test_model_and_prediction.py` | training, probabilities, thresholds, save/load |
| `test_model_schema_guard.py` | stale/reordered/short schemas refused; cache invalidation on retrain |
| `test_evaluation.py` | metrics, threshold sweep, confusion-matrix layout |
| `test_security.py` | **SSRF policy** (52 cases), sanitisation, redaction |
| `test_web_fetcher.py` | GET-only, no cookies, redirect revalidation, size/type limits |
| `test_html_analyzer.py` | text/forms/iframes/hidden content/scripts, malformed HTML |
| `test_form_analyzer.py` | password/OTP/payment/seed-phrase detection, cross-domain posts |
| `test_llm_analyzer.py` | **prompt-injection defences**, schema validation, failure paths |
| `test_ollama_client.py` | the local provider: request shape, provider selection, every Ollama failure mode, the injection fence, the Layer 4 gate — all mocked, **no server required** |
| `test_caches.py` | versioning, TTL, lossless round-trip, failures never cached |
| `test_decision_engine.py` | triage, every combination rule, no-averaging |
| `test_explanations.py` | reasons grounded in observed values, attribution |
| `test_pipeline.py` | `analyze_url` end to end, Layer 4 gate, every degradation |
| `test_cli.py` | subcommands, JSON output, exit codes |
| `test_layer4_integration.py` | the **real** `requests` client against a loopback HTTP server: cookie policy, GET-only, redirects, size/type limits, and the full fetch→HTML→form→LLM chain |
| `test_layer2_timeouts.py` | DNS short-circuits, TLS skip, the Layer 2 budget, bounded name resolution |
| `test_train_script.py` | end-to-end `detector train` on a fixture CSV |
| `test_service.py` | local FastAPI service: health check, analyze endpoint, Host header validation, bearer auth, concurrency limits |
| `test_service_ssrf.py` | service SSRF regression: IPv4/IPv6 loopback, private ranges, AWS metadata, DNS rebinding mitigations |
| `test_connection_pinning_ssrf.py` | connection pinning & SSRF: DNS rebinding mitigation, IPv4/IPv6 special ranges, alternative IP encodings, redirect hop re-validation, TLS SNI & certificate verification under pinning, proxy rejection |
| `test_package_extension.py` | Firefox extension packaging: deterministic bit-for-bit archives, manifest validation, sha256 checksums |

---

# Project structure

```
phishing-url-detector/
├── .github/
│   ├── dependabot.yml         # weekly automated dependency & GitHub Actions updates
│   └── workflows/ci.yml       # GitHub Actions CI matrix (Python 3.10-3.12, Node 20)
├── extension/                 # Firefox WebExtension (MV3)
│   ├── manifest.json          # Firefox add-on manifest
│   ├── background/            # background service, cache, and API client
│   ├── popup/                 # toolbar status popup
│   ├── warning/               # interstitial phishing warning page
│   ├── options/               # local service and token settings
│   └── tests/                 # Node test suite (26 passing tests)
├── src/
│   ├── detector/              # core 4-layer detection engine
│   │   ├── __init__.py        # public API
│   │   ├── cli.py             # analyze / features / train / evaluate
│   │   ├── config.py          # EVERY tunable: Detector/ML/Decision/Network/LLM/Cache
│   │   ├── security.py        # SSRF guard, sanitisation, redaction
│   │   ├── observability.py   # structured logging, redaction filter, timings
│   │   │
│   │   ├── url_parser.py      # L1: safe scheme-tolerant parsing
│   │   ├── brands.py          # L1: Levenshtein, typosquat, brand-in-place
│   │   ├── lexical_features.py# L1: extract_lexical_features()
│   │   │
│   │   ├── dns_features.py    # L2: A/AAAA/NS/MX/TTL/PTR + IP classification
│   │   ├── domain_age.py      # L2: RDAP -> WHOIS (bootstrap cached)
│   │   ├── tls_features.py    # L2: certificate inspection
│   │   ├── domain_intelligence.py # L2: orchestration + per-domain rate limiting
│   │   ├── layer2_cache.py    # L2: versioned SQLite cache, keyed by domain
│   │   │
│   │   ├── feature_schema.py  # L3: AUTHORITATIVE ordered schema + guards
│   │   ├── feature_pipeline.py# L3: layer dicts -> feature vector
│   │   ├── dataset.py         # L3: loaders, cleaning, leakage-safe splits
│   │   ├── model.py           # L3: PhishingModel + schema validation on load
│   │   ├── evaluation.py      # L3: metrics, sweep, per-source breakdown
│   │   ├── prediction.py      # L3: Predictor (mtime-keyed model cache)
│   │   ├── train.py           # L3: training entry point
│   │   │
│   │   ├── web_fetcher.py     # L4a: SSRF-guarded bounded GET
│   │   ├── html_analyzer.py   # L4b: static HTML analysis
│   │   ├── form_analyzer.py   # L4c: field purpose + cross-domain destination
│   │   ├── llm_client.py      # L4d: LLMClient protocol + Ollama/Anthropic clients
│   │   ├── llm_analyzer.py    # L4d: fenced prompt + strict schema validation
│   │   ├── page_cache.py      # L4: versioned SQLite cache, keyed by URL hash
│   │   │
│   │   ├── explanations.py    # reason codes -> human-readable messages
│   │   ├── decision_engine.py # triage + ordered combination rules
│   │   └── pipeline.py        # analyze_url(): the one top-level API
│   └── detector_service/      # local hardened FastAPI service
│       ├── app.py             # FastAPI app, endpoints, Host header & auth validation
│       ├── config.py          # loopback configuration & token generation
│       ├── schemas.py         # Pydantic schemas for requests, responses & errors
│       └── server.py          # Uvicorn runner bound to 127.0.0.1
├── data/                      # raw datasets (gitignored) + processed caches
├── models/                    # random_forest.joblib + feature_schema.json
├── reports/                   # feature importance, training metadata
├── scripts/
│   ├── make_sample_dataset.py # synthetic dataset generator
│   ├── package_extension.py   # deterministic Firefox extension packaging
│   ├── security_audit.py      # offline security and credential scanner
│   └── train.py               # training helper script
├── tests/                     # 563 offline tests (detector core + service + packaging)
├── SECURITY.md                # vulnerability disclosure policy & threat model
├── RELEASE_CHECKLIST.md       # step-by-step pre-release verification checklist
├── main.py                    # run from a checkout without installing
└── pyproject.toml             # packaging, pytest and ruff config
```

---

# Deployment notes

**Sandbox requirements.** Layer 4 fetches attacker-chosen URLs. The SSRF guard
is the first line of defence, not the only one you should have:

- Run the detector with **egress restricted to public address ranges** at the
  network level. The fetcher validates every hop and pins TCP connections to
  validated IPs, while network-level egress filtering provides defense-in-depth
  against system-level escapes.
- Run it as an **unprivileged user** with no access to internal services,
  cloud metadata endpoints, or credentials it does not need.
- The detector needs outbound **DNS**, **HTTPS to arbitrary public hosts**
  (Layer 4), **HTTPS to `rdap.org` and registry RDAP endpoints** (Layer 2), and
  — only with `LLM_PROVIDER=anthropic` — **HTTPS to the Anthropic API**. Nothing
  else. The default `ollama` provider needs **no egress at all**: it talks to
  `127.0.0.1:11434`, which the sandbox may otherwise keep closed.
- Use `--layer1-only` when you must not disclose interest in a URL: running
  Layers 2–4 tells your resolver, the registry, **and the site itself** that you
  are looking at it.

**Privacy.** Layer 2 emits DNS/RDAP traffic naming the domain; Layer 4 fetches
the page, so the operator sees your request. **With the default `ollama`
provider, page content never leaves the machine** — inference is local. With
`LLM_PROVIDER=anthropic`, Layer 4 sends page content to a third-party LLM API;
do not select it for URLs whose contents must not leave your environment.

---

# Limitations

**Detection**

- **No JavaScript rendering.** A page whose credential form is injected
  client-side looks emptier to Layer 4 than to a real browser.
- **Cloaking defeats Layer 4.** A page can serve bland content to an automated
  fetcher and the real kit to a victim; `fetched=True` with innocuous content is
  a normal, non-malicious-looking outcome.
- **A snapshot, not history.** No passive DNS, no reputation feeds, no
  certificate-transparency monitoring, no URL-history tracking.
- **`llm_likelihood` is a self-report**, not a measurement.

**Modelling**

- **`risk_score` is not calibrated** — no calibration has been performed.
- The Random Forest is only as good as the corpus; see "Dataset bias" and
  "Why 99% accuracy is not a good phishing detector" above.
- **Explainability is global, not per-prediction** without the SHAP extra.

**Architecture**

- **No learned rules→ML→LLM combiner.** The combination is a fixed, documented
  rule table, not a model trained on labelled combined-signal data.
- **Policy is thresholds and lists only** — no per-tenant policy, no
  human-review queue, no feedback loop from analyst decisions.
- **DNS rebinding** is mitigated via connect-time IP pinning and hop-wise
  validation in `PinnedHTTPAdapter`; see the SSRF section.
- **Single-process, synchronous.** No batch concurrency, no rate-limited worker
  pool for bulk scanning.
