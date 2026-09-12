"""Layer 1 unit tests.

Covers the 10 required scenarios plus focused unit tests for the helpers
(Levenshtein, Shannon entropy, the URL parser).
"""

from __future__ import annotations

import math

import pytest

from detector import extract_lexical_features, parse_url
from detector.brands import (
    brand_in_wrong_place,
    brand_proximity,
    closest_brand,
    levenshtein,
)
from detector.config import DEFAULT_CONFIG
from detector.lexical_features import shannon_entropy


# --------------------------------------------------------------------------
# 1. a normal legitimate URL
# --------------------------------------------------------------------------
def test_legitimate_url_is_clean():
    f = extract_lexical_features("https://www.google.com/search?q=test")

    assert f.is_malformed is False
    assert f.hostname == "www.google.com"
    assert f.registered_domain == "google.com"
    assert f.subdomain == "www"
    assert f.subdomain_depth == 1
    assert f.tld == "com"
    assert f.has_ip is False
    assert f.has_at_symbol is False
    assert f.has_double_slash is False
    assert f.is_suspicious_tld is False
    assert f.is_suspicious_port is False
    assert f.keyword_count == 0
    assert f.brand_in_wrong_place is False
    # 'google' IS the registered domain -> exact match, NOT a typosquat
    assert f.is_exact_brand_match is True
    assert f.is_typosquat_suspicious is False
    assert f.levenshtein_distance == 0
    assert f.is_punycode is False
    assert f.contains_non_ascii is False


# --------------------------------------------------------------------------
# 2. paypal.security-check.xyz/login  -> brand in the wrong place + bad TLD
# --------------------------------------------------------------------------
def test_brand_in_subdomain_wrong_registered_domain():
    f = extract_lexical_features("https://paypal.security-check.xyz/login")

    assert f.registered_domain == "security-check.xyz"
    assert f.subdomain == "paypal"
    assert f.tld == "xyz"
    assert f.is_suspicious_tld is True
    assert f.brand_in_wrong_place is True
    assert f.brand_in_wrong_place_name == "paypal"
    assert f.keyword_count == 1
    assert "login" in f.keywords_found
    assert f.as_dict()["keyword_login"] is True
    assert f.hyphen_count >= 1
    # the registered-domain label itself is not a typo of a brand
    assert f.is_typosquat_suspicious is False


# --------------------------------------------------------------------------
# 3. paypa1.com  -> typosquat (edit distance 1 from 'paypal')
# --------------------------------------------------------------------------
def test_typosquat_close_to_brand():
    f = extract_lexical_features("paypa1.com")

    assert f.is_malformed is False        # scheme-less URL still parses
    assert f.registered_domain == "paypa1.com"
    assert f.closest_brand == "paypal"
    assert f.levenshtein_distance == 1
    assert f.is_typosquat_suspicious is True
    assert f.is_exact_brand_match is False
    assert f.digit_count == 1


# --------------------------------------------------------------------------
# 4. a URL containing '@'
# --------------------------------------------------------------------------
def test_at_symbol_uses_real_host_after_at():
    f = extract_lexical_features("http://paypal.com@evil.example.com/login")

    assert f.has_at_symbol is True
    assert f.hostname == "evil.example.com"      # host is AFTER the '@'
    assert f.registered_domain == "example.com"
    # 'paypal' appears in the userinfo, i.e. not in the real host/subdomain/path
    assert f.brand_in_wrong_place is False


# --------------------------------------------------------------------------
# 5. an IP-based URL
# --------------------------------------------------------------------------
def test_ipv4_url():
    f = extract_lexical_features("http://192.168.0.1:8081/account/verify")

    assert f.has_ip is True
    assert f.ip_version == 4
    assert f.hostname == "192.168.0.1"
    assert f.registered_domain == ""            # no PSL domain for an IP
    assert f.tld == ""
    assert f.port == 8081
    assert f.is_suspicious_port is True         # 8081 not in the safe set
    assert f.keyword_count == 2
    assert set(f.keywords_found) == {"verify", "account"}


def test_ipv6_url():
    f = extract_lexical_features("http://[2001:db8::1]:22/login")

    assert f.has_ip is True
    assert f.ip_version == 6
    assert f.hostname == "2001:db8::1"
    assert f.port == 22
    assert f.is_suspicious_port is True


# --------------------------------------------------------------------------
# 6. a punycode / xn-- domain
# --------------------------------------------------------------------------
def test_punycode_domain():
    f = extract_lexical_features("http://xn--pypal-4ve.com/signin")

    assert f.is_punycode is True
    assert f.contains_non_ascii is False        # the ASCII-compatible encoding
    assert f.tld == "com"


def test_literal_unicode_hostname():
    # Cyrillic 'а' (U+0430) instead of Latin 'a'
    f = extract_lexical_features("http://pаypal.com/login")

    assert f.contains_non_ascii is True
    assert f.is_punycode is False


# --------------------------------------------------------------------------
# 7. a high-entropy random-looking domain
# --------------------------------------------------------------------------
def test_high_entropy_domain():
    low = extract_lexical_features("https://google.com/")
    high = extract_lexical_features("https://a8f3k2j9x7q1z5b4.top/")

    assert high.entropy > low.entropy
    assert high.entropy > 3.5
    assert high.is_suspicious_tld is True       # .top


# --------------------------------------------------------------------------
# 8. a suspicious TLD
# --------------------------------------------------------------------------
@pytest.mark.parametrize("url,tld", [
    ("http://example.tk/", "tk"),
    ("http://example.zip/", "zip"),
    ("http://foo.bar.gq/path", "gq"),
    ("https://thing.click/", "click"),
])
def test_suspicious_tld(url, tld):
    f = extract_lexical_features(url)
    assert f.tld == tld
    assert f.is_suspicious_tld is True


def test_non_suspicious_tld():
    assert extract_lexical_features("https://example.org/").is_suspicious_tld is False


# --------------------------------------------------------------------------
# 9. a .co.uk domain -> correct PSL-aware registered-domain extraction
# --------------------------------------------------------------------------
def test_multi_part_suffix_co_uk():
    f = extract_lexical_features("https://login.portal.example.co.uk/dashboard")

    assert f.tld == "co.uk"
    assert f.registered_domain == "example.co.uk"
    assert f.subdomain == "login.portal"
    assert f.subdomain_depth == 2
    assert f.is_suspicious_tld is False


# --------------------------------------------------------------------------
# 10. malformed URLs must not crash
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "not a url",
    "http://",
    "https://",
    "ht!tp://%%%",
    "http:// spaces .com",
    "://missing-scheme",
    "http://[gggg::zzzz]/",
    None,
    12345,
])
def test_malformed_urls_do_not_raise(bad):
    f = extract_lexical_features(bad)                 # must not raise
    d = f.as_dict()
    assert isinstance(d, dict)
    assert "url_length" in d
    # nothing trustworthy should have been invented
    assert f.has_ip is False


def test_malformed_flag_set_for_junk():
    assert extract_lexical_features("not a url").is_malformed is True
    assert extract_lexical_features("").is_malformed is True
    assert extract_lexical_features("http://").is_malformed is True


# --------------------------------------------------------------------------
# helper: Levenshtein
# --------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,d", [
    ("", "", 0),
    ("abc", "abc", 0),
    ("", "abc", 3),
    ("abc", "", 3),
    ("paypa1", "paypal", 1),
    ("kitten", "sitting", 3),
    ("microsft", "microsoft", 1),
    ("flaw", "lawn", 2),
])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d
    assert levenshtein(b, a) == d        # symmetric


def test_closest_brand():
    brand, dist = closest_brand("faceb00k", DEFAULT_CONFIG.normalised_brands())
    assert brand == "facebook"
    assert dist == 2


def test_brand_proximity_guards_short_labels():
    # 'git' is within distance 3 of 'github' but too short / length-mismatched
    prox = brand_proximity("git", DEFAULT_CONFIG.normalised_brands(),
                           max_distance=2, min_label_len=4)
    assert prox.is_typosquat_suspicious is False


def test_brand_proximity_exact_is_not_typosquat():
    prox = brand_proximity("github", DEFAULT_CONFIG.normalised_brands(),
                           max_distance=2, min_label_len=4)
    assert prox.is_exact_brand is True
    assert prox.is_typosquat_suspicious is False
    assert prox.min_distance == 0


def test_brand_in_wrong_place_path():
    hit, name = brand_in_wrong_place(
        subdomain="", path="/amazon/signin", domain_label="totally-legit",
        brands=DEFAULT_CONFIG.normalised_brands(),
    )
    assert hit is True and name == "amazon"


# --------------------------------------------------------------------------
# helper: Shannon entropy
# --------------------------------------------------------------------------
def test_shannon_entropy_values():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("ab") == pytest.approx(1.0)
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    # 4 distinct symbols, uniform -> log2(4)
    assert shannon_entropy("abcd") == pytest.approx(math.log2(4))


# --------------------------------------------------------------------------
# helper: parser edge cases
# --------------------------------------------------------------------------
def test_parser_scheme_detection():
    assert parse_url("https://a.com").scheme_present is True
    assert parse_url("a.com/login").scheme_present is False
    assert parse_url("a.com/login").hostname == "a.com"


def test_parser_double_slash_in_path():
    assert parse_url("http://good.com//evil.com").has_double_slash_in_path is True
    assert parse_url("http://good.com/a/b").has_double_slash_in_path is False


def test_parser_credentials_and_port():
    p = parse_url("http://user:pass@host.example.com:8443/x")
    assert p.username == "user"
    assert p.hostname == "host.example.com"
    assert p.port == 8443
    assert p.has_at_symbol is True


def test_parser_query_and_fragment_lengths():
    f = extract_lexical_features("https://x.com/p?a=1&b=2#frag")
    assert f.query_length == len("a=1&b=2")
    assert f.fragment_length == len("frag")


def test_as_dict_has_example_contract_keys():
    d = extract_lexical_features("https://paypal.security-check.xyz/login").as_dict()
    for key in [
        "url_length", "domain_length", "dot_count", "hyphen_count", "digit_count",
        "subdomain_depth", "has_ip", "has_at_symbol", "has_double_slash",
        "keyword_count", "brand_in_wrong_place", "closest_brand",
        "levenshtein_distance", "is_punycode", "contains_non_ascii", "entropy",
        "tld", "is_suspicious_tld",
    ]:
        assert key in d, f"missing contract key: {key}"


def test_extraction_is_deterministic():
    u = "https://paypal.security-check.xyz/login?x=1"
    assert extract_lexical_features(u).as_dict() == extract_lexical_features(u).as_dict()


# ---------------------------------------------------------------------------
# has_at_symbol: authority only (schema v2 semantics)
# ---------------------------------------------------------------------------
def test_at_symbol_flag_fires_on_userinfo_disguise():
    """`http://paypal.com@evil.com/` really loads evil.com."""
    f = extract_lexical_features("http://paypal.com@evil-site.ru/login")
    assert f.has_at_symbol is True
    assert f.hostname == "evil-site.ru"
    assert f.registered_domain == "evil-site.ru"


def test_at_symbol_flag_ignores_an_email_in_the_query():
    """An '@' in a query string is ordinary and must not raise the flag.

    Scanning the whole URL string made every `?email=a@b.com` link look like a
    userinfo disguise - a large false-positive source before schema v2.
    """
    f = extract_lexical_features("https://example.com/signup?email=user@gmail.com")
    assert f.has_at_symbol is False


def test_at_symbol_flag_ignores_an_at_in_the_path():
    f = extract_lexical_features("https://example.com/users/@alice/profile")
    assert f.has_at_symbol is False


def test_at_symbol_flag_ignores_an_at_in_the_fragment():
    f = extract_lexical_features("https://example.com/page#contact@support")
    assert f.has_at_symbol is False


# ---------------------------------------------------------------------------
# registered-domain fallback
# ---------------------------------------------------------------------------
def test_registered_domain_falls_back_to_the_whole_host():
    """No public suffix -> keep the full host, not just the last label."""
    f = extract_lexical_features("http://build.ci.internal/job/1")
    assert f.registered_domain == "build.ci.internal"


def test_registered_domain_of_agrees_with_layer1_for_named_hosts():
    from detector.lexical_features import registered_domain_of

    for url in ("https://a.example.co.uk/x", "http://build.ci.internal/",
                "https://example.com/", "https://paypal.security-check.xyz/login"):
        assert registered_domain_of(url) == extract_lexical_features(url).registered_domain


def test_grouping_key_keeps_ip_hosts_distinct():
    """The one deliberate divergence from Layer 1's feature.

    Layer 1 reports "" for an IP host (it has no registered domain). The
    grouping key must not, or every IP-addressed URL would share one Layer 2
    cache entry.
    """
    from detector.lexical_features import registered_domain_of

    assert extract_lexical_features("http://192.168.1.1/x").registered_domain == ""
    assert registered_domain_of("http://192.168.1.1/x") == "192.168.1.1"
    assert registered_domain_of("http://203.0.113.9/x") != registered_domain_of(
        "http://192.168.1.1/x")
