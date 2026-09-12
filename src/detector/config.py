"""Every tunable in the project, in one module.

Nothing outside this file should hard-code a keyword list, a brand, a timeout,
a threshold, a size limit or a model name. The config objects are frozen
dataclasses, so a caller customises by constructing a new one (or with
`dataclasses.replace`), never by mutating shared state.

    DetectorConfig   Layers 1-2: word/brand/TLD lists, DNS + RDAP/WHOIS limits
    NetworkConfig    Layer 4 outbound HTTP: timeouts, size caps, redirects, SSRF
    LLMConfig        Layer 4 LLM: provider, model, limits, where a key comes from
    CacheConfig      Layer 2 / Layer 4 caches: paths, TTLs, on/off
    MLConfig         Layer 3: model hyper-parameters, split dates, artifacts
    DecisionConfig   the decision engine: thresholds + the three configs above

Secrets are **never** stored here as literals - `LLMConfig.api_key_env` names an
environment variable and nothing else. The default Layer 4 provider is local
Ollama, which needs no credential at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Project root = two levels up from this file (src/detector/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# Layer 1 - lexical analysis word lists
# ===========================================================================
# Words that show up disproportionately in credential-phishing URLs. `KEYWORDS`
# drives `keyword_count`; `IMPORTANT_KEYWORDS` additionally get an individual
# binary feature each (keyword_<name>).
KEYWORDS: tuple[str, ...] = (
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "confirm",
    "wallet",
    "kyc",
    "signin",
    "password",
    "payment",
)
IMPORTANT_KEYWORDS: tuple[str, ...] = KEYWORDS

# --- brands most often impersonated -----------------------------------------
BRANDS: tuple[str, ...] = (
    "paypal",
    "microsoft",
    "google",
    "apple",
    "amazon",
    "facebook",
    "instagram",
    "netflix",
    "linkedin",
    "github",
    "binance",
)

# --- TLDs cheap/abused enough to be a weak signal on their own -------------
SUSPICIOUS_TLDS: tuple[str, ...] = (
    "zip",
    "top",
    "tk",
    "xyz",
    "click",
    "gq",
    "ml",
    "cf",
)

# --- ports we consider normal for web traffic -----------------------------
SAFE_PORTS: frozenset[int] = frozenset({80, 443, 8080, 8443})

# --- characters counted as "special" in the shape features ----------------
# Dots, hyphens and digits are counted separately, so they are excluded here.
SPECIAL_CHARS: frozenset[str] = frozenset("@%?=&#~+!$*(),;:'\"[]{}|^<>\\ `")

# --- typosquatting -------------------------------------------------------
TYPOSQUAT_MAX_DISTANCE: int = 2
TYPOSQUAT_MIN_LABEL_LEN: int = 4

# --- Layer 2: DNS / domain intelligence --------------------------------------
DNS_TIMEOUT: float = 3.0
DNS_LIFETIME: float = 6.0
RDAP_TIMEOUT: float = 6.0
WHOIS_TIMEOUT: float = 6.0
TLS_TIMEOUT: float = 5.0
NEW_DOMAIN_MAX_AGE_DAYS: int = 30
REVERSE_DNS_MAX_LOOKUPS: int = 3


@dataclass(frozen=True)
class DetectorConfig:
    """Immutable bundle of every Layer 1 / Layer 2 tunable."""

    keywords: tuple[str, ...] = KEYWORDS
    important_keywords: tuple[str, ...] = IMPORTANT_KEYWORDS
    brands: tuple[str, ...] = BRANDS
    suspicious_tlds: tuple[str, ...] = SUSPICIOUS_TLDS
    safe_ports: frozenset[int] = SAFE_PORTS
    special_chars: frozenset[str] = SPECIAL_CHARS
    typosquat_max_distance: int = TYPOSQUAT_MAX_DISTANCE
    typosquat_min_label_len: int = TYPOSQUAT_MIN_LABEL_LEN

    # Layer 2
    dns_timeout: float = DNS_TIMEOUT
    dns_lifetime: float = DNS_LIFETIME
    rdap_timeout: float = RDAP_TIMEOUT
    whois_timeout: float = WHOIS_TIMEOUT
    tls_timeout: float = TLS_TIMEOUT
    new_domain_max_age_days: int = NEW_DOMAIN_MAX_AGE_DAYS
    reverse_dns_max_lookups: int = REVERSE_DNS_MAX_LOOKUPS
    # Optional explicit resolver addresses; None = use the system resolver.
    dns_nameservers: tuple[str, ...] | None = None
    # Minimum seconds between outbound DNS/RDAP/WHOIS bursts for one domain.
    # Guards against hammering a registry when many URLs share a domain.
    domain_lookup_min_interval: float = 0.0

    #: Wall-clock ceiling for ALL of Layer 2 for one domain.
    #:
    #: Each probe already has its own timeout, but they run in sequence: four
    #: DNS queries, then RDAP, then WHOIS, then TLS. On an unresponsive
    #: resolver every one of them can burn its full budget, and the individual
    #: limits multiply instead of bounding the total. When this deadline
    #: passes, the remaining probes are skipped and say so in `notes`, rather
    #: than a single analysis running for minutes.
    layer2_budget: float = 20.0

    # normalised, deduplicated views -------------------------------------
    def normalised_keywords(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(k.lower() for k in self.keywords))

    def normalised_brands(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(b.lower() for b in self.brands))

    def normalised_suspicious_tlds(self) -> frozenset[str]:
        return frozenset(t.lower().lstrip(".") for t in self.suspicious_tlds)


DEFAULT_CONFIG = DetectorConfig()


# ===========================================================================
# Layer 4 - outbound HTTP and the SSRF policy
# ===========================================================================
@dataclass(frozen=True)
class NetworkConfig:
    """Everything about fetching a (hostile, untrusted) web page.

    The defaults are deliberately restrictive: a phishing page is adversarial
    input, and the fetcher is the component an attacker gets to point at
    whatever they like. See `security.py` for the SSRF checks these drive.
    """

    connect_timeout: float = 5.0
    read_timeout: float = 8.0
    #: Hard wall-clock ceiling for one fetch including redirects.
    total_timeout: float = 20.0

    #: Bytes of *decoded* response body to keep. Also bounds decompression.
    max_response_bytes: int = 1_000_000
    #: Characters of extracted visible text kept for analysis.
    max_text_chars: int = 20_000
    #: Redirect hops followed; each hop is SSRF-validated before it is taken.
    max_redirects: int = 3

    #: TLS certificate verification. Never turn this off outside a lab.
    verify_tls: bool = True
    user_agent: str = (
        "phishing-url-detector/0.5 (passive security analysis; GET only)"
    )
    allowed_schemes: tuple[str, ...] = ("http", "https")
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
    )
    #: Allow fetching loopback/private/link-local targets. Controlled test
    #: environments only - this disables the SSRF guard's core protection.
    allow_private_targets: bool = False
    #: Extra hostnames that are always refused, regardless of what they resolve to.
    blocked_hosts: tuple[str, ...] = (
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    )


DEFAULT_NETWORK_CONFIG = NetworkConfig()


# ===========================================================================
# Layer 4 - LLM
# ===========================================================================
#: Providers Layer 4 can talk to, and the defaults each one is happiest with.
#: `ollama` is the project default: it runs the model on this machine, so it
#: needs no API key and costs nothing per call, but a small local model on CPU
#: is far slower than a hosted one - hence the much longer default timeout.
PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "ollama": {"model": "gemma3:4b", "timeout": 120.0},
    "anthropic": {"model": "claude-haiku-4-5-20251001", "timeout": 20.0},
}

DEFAULT_LLM_PROVIDER = "ollama"


@dataclass(frozen=True)
class LLMConfig:
    """LLM analyzer settings, for whichever provider Layer 4 is pointed at.

    `provider` selects the client (see `llm_client.create_llm_client`); `model`
    is always the model name *for that provider*. No API key is ever stored
    here - `api_key_env` names an environment variable, and only the Anthropic
    client reads it. The local Ollama provider needs no credential at all.
    """

    enabled: bool = True
    #: "ollama" (local, default) or "anthropic" (hosted).
    provider: str = DEFAULT_LLM_PROVIDER
    model: str = "gemma3:4b"
    timeout: float = 120.0
    max_tokens: int = 1024
    #: Name of the environment variable holding the key. Never the key itself.
    #: Only consulted by the Anthropic provider.
    api_key_env: str = "ANTHROPIC_API_KEY"

    # --- Ollama (local inference) -----------------------------------------
    #: Base URL of the local Ollama server. Operator-controlled, never derived
    #: from a fetched page.
    ollama_base_url: str = "http://127.0.0.1:11434"
    #: Context window requested from Ollama. This must comfortably exceed the
    #: whole prompt: Ollama drops the OLDEST tokens when a prompt overflows the
    #: context, and the oldest tokens here are the system prompt - i.e. exactly
    #: the prompt-injection defences. Sized for the analyzer prompt (~3k tokens)
    #: plus the reply, well above Ollama's 4096-token default.
    ollama_num_ctx: int = 8192
    #: Greedy decoding: the same page should yield the same analysis.
    ollama_temperature: float = 0.0

    # Output limits. LLM output is influenced by attacker-controlled page text,
    # so everything it returns is truncated and sanitised before it is surfaced.
    max_summary_chars: int = 300
    max_tactics: int = 8
    max_tactic_chars: int = 80
    max_brand_chars: int = 40

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def for_provider(
        self,
        provider: str | None,
        *,
        model: str | None = None,
        timeout: float | None = None,
        ollama_base_url: str | None = None,
    ) -> LLMConfig:
        """Return this config switched to `provider`.

        Switching providers also switches to that provider's default model and
        timeout, because a model name is meaningless across providers (asking
        Anthropic for "gemma3:4b" is an error, not a fallback). Staying on the
        same provider keeps whatever model and timeout are already configured.
        Explicit arguments always win.
        """
        provider = (provider or self.provider).strip().lower()
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        same = provider == self.provider
        return replace(
            self,
            provider=provider,
            model=model or (self.model if same else str(defaults.get("model", self.model))),
            timeout=(
                timeout
                if timeout is not None
                else (self.timeout if same else float(defaults.get("timeout", self.timeout)))
            ),
            ollama_base_url=ollama_base_url or self.ollama_base_url,
        )


DEFAULT_LLM_CONFIG = LLMConfig()


# ===========================================================================
# caches
# ===========================================================================
@dataclass(frozen=True)
class CacheConfig:
    """Layer 2 (domain-level) and Layer 4 (URL-level) caches.

    They are separate on purpose: DNS/RDAP facts belong to a *domain* and are
    shared by every URL on it, while page analysis belongs to one *URL*.
    """

    layer2_enabled: bool = True
    layer2_path: str = str(PROJECT_ROOT / "data" / "processed" / "layer2_cache.sqlite")
    #: 0 or negative = never expire (the training-time behaviour).
    layer2_ttl_seconds: float = 24 * 3600.0
    #: Much shorter TTL for "this domain does not resolve" rows.
    #:
    #: A negative DNS answer is far less durable than a positive one: it is
    #: produced equally by an unregistered domain, a domain that has just been
    #: taken down, and our own resolver having a bad minute. Caching that for a
    #: full day would pin a transient outage as fact - and, worse, freeze a
    #: brand-new phishing domain's record from before it was delegated.
    layer2_negative_ttl_seconds: float = 300.0

    page_enabled: bool = True
    page_path: str = str(PROJECT_ROOT / "data" / "processed" / "page_cache.sqlite")
    page_ttl_seconds: float = 6 * 3600.0


DEFAULT_CACHE_CONFIG = CacheConfig()


# ===========================================================================
# Layer 3: machine-learning classification
# ===========================================================================
@dataclass(frozen=True)
class MLConfig:
    """Every Layer 3 knob in one place."""

    # --- reproducibility ---------------------------------------------------
    random_seed: int = 42

    # --- Random Forest hyper-parameters (kept deliberately small) --------
    model_kind: str = "random_forest"      # future: "xgboost" / "lightgbm"
    n_estimators: int = 300
    max_depth: int | None = 20
    min_samples_leaf: int = 2
    min_samples_split: int = 4
    max_features: str = "sqrt"
    # Class imbalance: phishing/benign are rarely 50/50. "balanced" reweights
    # by inverse class frequency instead of duplicating rows.
    class_weight: str | None = "balanced"
    n_jobs: int = -1

    # --- reporting threshold -----------------------------------------
    # The single cut used when *reporting* training/evaluation metrics. The
    # live decision path does not use it - see DecisionConfig.low/high_threshold.
    default_threshold: float = 0.5
    eval_thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # --- time-based split boundaries ------------------------
    train_end: str = "2026-01-01"
    val_end: str = "2026-05-01"
    # Where rows land when a source has no per-row timestamp.
    undated_bucket: str = "train"          # "train" | "exclude" | "grouped"

    # --- leakage controls ---------------------------------
    drop_cross_split_domains: bool = True  # a domain may not span train & test
    max_urls_per_registered_domain: int | None = 50   # None = keep all

    # --- domain-age leakage guard ------------------
    include_domain_age_features: bool = True
    #: Training prints a warning above this share of importance on age features.
    domain_age_reliance_warn: float = 0.35

    # --- artifact locations -----------------------------------------
    data_dir: str = str(PROJECT_ROOT / "data")
    models_dir: str = str(PROJECT_ROOT / "models")
    reports_dir: str = str(PROJECT_ROOT / "reports")

    model_filename: str = "random_forest.joblib"
    schema_filename: str = "feature_schema.json"
    metadata_filename: str = "training_metadata.json"


DEFAULT_ML_CONFIG = MLConfig()


# ===========================================================================
# the decision engine
# ===========================================================================
class ConfigError(ValueError):
    """Raised for a self-contradictory configuration."""


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds and policy for the final decision, plus the Layer 4 configs.

    Layer 3's probability decides whether Layer 4 (a page fetch + an LLM call)
    runs at all: at or above `high_threshold` -> phishing, at or below
    `low_threshold` -> safe, and only the band in between pays for Layer 4.
    """

    low_threshold: float = 0.3
    high_threshold: float = 0.7

    #: Master switch for Layer 4 (page fetch + LLM).
    enable_layer4: bool = True

    #: Registered domains always classified phishing without any further work.
    blocklist_domains: tuple[str, ...] = ()
    #: Registered domains always classified safe (checked after the blocklist).
    allowlist_domains: tuple[str, ...] = ()

    network: NetworkConfig = field(default_factory=NetworkConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_threshold <= 1.0:
            raise ConfigError(f"low_threshold out of range: {self.low_threshold}")
        if not 0.0 <= self.high_threshold <= 1.0:
            raise ConfigError(f"high_threshold out of range: {self.high_threshold}")
        if self.low_threshold > self.high_threshold:
            raise ConfigError(
                f"low_threshold ({self.low_threshold}) must not exceed "
                f"high_threshold ({self.high_threshold})"
            )

    def normalised_blocklist(self) -> frozenset[str]:
        return frozenset(d.strip().lower().lstrip(".") for d in self.blocklist_domains if d.strip())

    def normalised_allowlist(self) -> frozenset[str]:
        return frozenset(d.strip().lower().lstrip(".") for d in self.allowlist_domains if d.strip())


DEFAULT_DECISION_CONFIG = DecisionConfig()


# ===========================================================================
# environment overrides
# ===========================================================================
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float_or_none(name: str) -> float | None:
    """Like `_env_float`, but reports "unset" so a default can be chosen later."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_first(*names: str) -> str | None:
    """First non-empty value among `names`, or None.

    Lets a project-scoped `DETECTOR_*` name and a conventional unprefixed one
    (e.g. `OLLAMA_BASE_URL`) both work, with the explicit one winning.
    """
    for name in names:
        raw = os.environ.get(name)
        if raw and raw.strip():
            return raw.strip()
    return None


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def decision_config_from_env(base: DecisionConfig | None = None) -> DecisionConfig:
    """Apply `DETECTOR_*` environment overrides to a DecisionConfig.

    Recognised: DETECTOR_LOW_THRESHOLD, DETECTOR_HIGH_THRESHOLD,
    DETECTOR_ENABLE_LAYER4, DETECTOR_BLOCKLIST, DETECTOR_ALLOWLIST,
    DETECTOR_LLM_PROVIDER (or LLM_PROVIDER), DETECTOR_LLM_MODEL,
    DETECTOR_LLM_TIMEOUT, DETECTOR_OLLAMA_BASE_URL (or OLLAMA_BASE_URL),
    DETECTOR_FETCH_MAX_BYTES, DETECTOR_FETCH_MAX_REDIRECTS,
    DETECTOR_ALLOW_PRIVATE_TARGETS, DETECTOR_CACHE_ENABLED.

    Secrets are never read here - only `LLMConfig.api_key_env` reads a key, and
    only at call time. Naming a provider is not a secret, so provider, model and
    endpoint all come from plain environment variables.
    """
    base = base or DecisionConfig()
    cache_enabled = _env_bool("DETECTOR_CACHE_ENABLED", True)
    return DecisionConfig(
        low_threshold=_env_float("DETECTOR_LOW_THRESHOLD", base.low_threshold),
        high_threshold=_env_float("DETECTOR_HIGH_THRESHOLD", base.high_threshold),
        enable_layer4=_env_bool("DETECTOR_ENABLE_LAYER4", base.enable_layer4),
        blocklist_domains=_env_tuple("DETECTOR_BLOCKLIST", base.blocklist_domains),
        allowlist_domains=_env_tuple("DETECTOR_ALLOWLIST", base.allowlist_domains),
        network=NetworkConfig(
            connect_timeout=base.network.connect_timeout,
            read_timeout=base.network.read_timeout,
            total_timeout=base.network.total_timeout,
            max_response_bytes=_env_int(
                "DETECTOR_FETCH_MAX_BYTES", base.network.max_response_bytes
            ),
            max_text_chars=base.network.max_text_chars,
            max_redirects=_env_int(
                "DETECTOR_FETCH_MAX_REDIRECTS", base.network.max_redirects
            ),
            verify_tls=base.network.verify_tls,
            user_agent=base.network.user_agent,
            allowed_schemes=base.network.allowed_schemes,
            allowed_content_types=base.network.allowed_content_types,
            allow_private_targets=_env_bool(
                "DETECTOR_ALLOW_PRIVATE_TARGETS", base.network.allow_private_targets
            ),
            blocked_hosts=base.network.blocked_hosts,
        ),
        llm=base.llm.for_provider(
            _env_first("DETECTOR_LLM_PROVIDER", "LLM_PROVIDER"),
            model=_env_first("DETECTOR_LLM_MODEL"),
            timeout=_env_float_or_none("DETECTOR_LLM_TIMEOUT"),
            ollama_base_url=_env_first("DETECTOR_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
        ),
        cache=CacheConfig(
            layer2_enabled=cache_enabled and base.cache.layer2_enabled,
            layer2_path=base.cache.layer2_path,
            layer2_ttl_seconds=base.cache.layer2_ttl_seconds,
            page_enabled=cache_enabled and base.cache.page_enabled,
            page_path=base.cache.page_path,
            page_ttl_seconds=base.cache.page_ttl_seconds,
        ),
    )
