"""Phishing URL detector - a four-layer detection pipeline.

    from detector import analyze_url
    result = analyze_url("https://paypal.security-check.xyz/login")
    result.classification      # "phishing" | "safe" | "uncertain"
    result.risk_score          # ordinal severity, NOT a calibrated probability
    result.reason_messages()   # human-readable evidence

Layers:
    1  lexical URL analysis      offline, deterministic    extract_lexical_features
    2  DNS / domain intelligence network, cached           analyze_domain
    3  Random Forest             tabular classifier        Predictor
    4  page fetch + LLM          uncertain band only       fetch_page / analyze_page_content
       decision engine           explicit rules            decide

`analyze_url` is the one top-level API; everything else here is a building
block it composes, exported for callers who need a single layer on its own.
"""

from __future__ import annotations

# --- Layer 1 ---------------------------------------------------------------
from .brands import brand_proximity, levenshtein

# --- configuration (every tunable lives in config.py) ----------------------
from .config import (
    DEFAULT_CACHE_CONFIG,
    DEFAULT_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_LLM_CONFIG,
    DEFAULT_ML_CONFIG,
    DEFAULT_NETWORK_CONFIG,
    CacheConfig,
    ConfigError,
    DecisionConfig,
    DetectorConfig,
    LLMConfig,
    MLConfig,
    NetworkConfig,
    decision_config_from_env,
)

# --- decision + orchestration ---------------------------------------------
from .decision_engine import Decision, TriageResult, combine, decide, triage

# --- Layer 2 ---------------------------------------------------------------
from .dns_features import DNSResult, classify_ip, resolve_domain
from .domain_age import DomainAgeResult, lookup_domain_age
from .domain_intelligence import DomainIntelligence, analyze_dns, analyze_domain
from .explanations import Reason

# --- Layer 3 ---------------------------------------------------------------
from .feature_pipeline import (
    active_feature_names,
    build_dataset,
    build_feature_vector,
    features_from_dicts,
)
from .feature_schema import (
    FEATURE_NAMES,
    MISSING,
    SCHEMA_VERSION,
    SchemaMismatch,
    schema_hash,
)

# --- Layer 4 ---------------------------------------------------------------
from .form_analyzer import FormInfo, InputField
from .html_analyzer import PageStructure, analyze_html
from .layer2_cache import Layer2Cache, open_layer2_cache
from .lexical_features import (
    LexicalFeatures,
    extract_lexical_features,
    registered_domain_of,
    shannon_entropy,
)
from .llm_analyzer import LLMAnalysis, analyze_page_content
from .llm_client import (
    AnthropicLLMClient,
    LLMClient,
    LLMClientError,
    OllamaLLMClient,
    create_llm_client,
)
from .model import PhishingModel
from .observability import configure_logging, get_logger
from .page_cache import PageCache, open_page_cache
from .pipeline import AnalysisResult, DetectionResult, analyze, analyze_url
from .prediction import ModelNotFound, Predictor, load_model, predict_url
from .security import UnsafeTargetError, check_fetch_target, is_public_address
from .tls_features import TLSResult, inspect_tls
from .url_parser import ParsedURL, parse_url
from .web_fetcher import PageContent, fetch_page

__all__ = [
    # top-level API
    "analyze_url",
    "AnalysisResult",
    # layers 1-2 building block
    "analyze",
    "DetectionResult",
    # layer 1
    "extract_lexical_features",
    "LexicalFeatures",
    "registered_domain_of",
    "shannon_entropy",
    "levenshtein",
    "brand_proximity",
    "parse_url",
    "ParsedURL",
    # layer 2
    "analyze_domain",
    "analyze_dns",
    "DomainIntelligence",
    "DNSResult",
    "resolve_domain",
    "classify_ip",
    "DomainAgeResult",
    "lookup_domain_age",
    "TLSResult",
    "inspect_tls",
    "Layer2Cache",
    "open_layer2_cache",
    # layer 3
    "build_feature_vector",
    "build_dataset",
    "features_from_dicts",
    "active_feature_names",
    "FEATURE_NAMES",
    "SCHEMA_VERSION",
    "SchemaMismatch",
    "schema_hash",
    "MISSING",
    "PhishingModel",
    "Predictor",
    "predict_url",
    "load_model",
    "ModelNotFound",
    # layer 4
    "fetch_page",
    "PageContent",
    "analyze_html",
    "PageStructure",
    "FormInfo",
    "InputField",
    "analyze_page_content",
    "LLMAnalysis",
    "LLMClient",
    "LLMClientError",
    "AnthropicLLMClient",
    "OllamaLLMClient",
    "create_llm_client",
    "PageCache",
    "open_page_cache",
    # decision engine
    "decide",
    "triage",
    "combine",
    "Decision",
    "TriageResult",
    "Reason",
    # security
    "check_fetch_target",
    "is_public_address",
    "UnsafeTargetError",
    # observability
    "configure_logging",
    "get_logger",
    # config
    "DetectorConfig",
    "DEFAULT_CONFIG",
    "MLConfig",
    "DEFAULT_ML_CONFIG",
    "DecisionConfig",
    "DEFAULT_DECISION_CONFIG",
    "NetworkConfig",
    "DEFAULT_NETWORK_CONFIG",
    "LLMConfig",
    "DEFAULT_LLM_CONFIG",
    "CacheConfig",
    "DEFAULT_CACHE_CONFIG",
    "ConfigError",
    "decision_config_from_env",
]

__version__ = "0.5.0"
