"""Layer 2c: TLS certificate facts.

No test opens a real TLS connection. What is exercised is the parts that must
be right regardless of the network: the SSRF guard in front of the handshake,
hostname matching, and the pure result-building from a certificate dict.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from detector.tls_features import TLSResult, _build, hostname_matches_cert, inspect_tls


def _cert_time(dt: datetime) -> str:
    return dt.strftime("%b %d %H:%M:%S %Y GMT")


def _cert(*, host="example.com", issuer="Let's Encrypt", age_days=100,
          valid_days=90, sans=None):
    now = datetime.now(timezone.utc)
    return {
        "subject": ((("commonName", host),),),
        "issuer": ((("organizationName", issuer),), (("commonName", "R3"),)),
        "notBefore": _cert_time(now - timedelta(days=age_days)),
        "notAfter": _cert_time(now + timedelta(days=valid_days)),
        "subjectAltName": tuple(("DNS", n) for n in (sans or [host])),
    }


# ---------------------------------------------------------------------------
# SSRF guard runs before any socket is opened
# ---------------------------------------------------------------------------
def test_internal_host_is_refused_without_connecting():
    result = inspect_tls("127.0.0.1")
    assert result.checked is False
    assert "blocked" in result.tls_error


def test_metadata_host_is_refused():
    result = inspect_tls("metadata.google.internal")
    assert result.checked is False
    assert "blocked" in result.tls_error


def test_empty_host_is_not_an_error_path():
    result = inspect_tls("")
    assert result.checked is False
    assert result.tls_error == "empty-host"


# ---------------------------------------------------------------------------
# hostname matching
# ---------------------------------------------------------------------------
def test_exact_hostname_match():
    assert hostname_matches_cert("example.com", "example.com", ()) is True


def test_san_match():
    assert hostname_matches_cert("www.example.com", "other.com",
                                 ("example.com", "www.example.com")) is True


def test_wildcard_matches_one_label_only():
    assert hostname_matches_cert("a.example.com", None, ("*.example.com",)) is True
    assert hostname_matches_cert("a.b.example.com", None, ("*.example.com",)) is False


def test_mismatched_hostname():
    assert hostname_matches_cert("evil.com", "example.com", ("example.com",)) is False


def test_empty_host_never_matches():
    assert hostname_matches_cert("", "example.com", ("example.com",)) is False


# ---------------------------------------------------------------------------
# result building
# ---------------------------------------------------------------------------
def test_valid_certificate_fields_extracted():
    result = _build("example.com", _cert(), "TLSv1.3", verified=True, tls_error=None)

    assert result.tls_available is True
    assert result.cert_valid is True
    assert "Let's Encrypt" in result.issuer
    assert result.hostname_matches is True
    assert result.is_self_signed is False
    assert result.tls_version == "TLSv1.3"
    assert 99 <= result.cert_age_days <= 101
    assert 89 <= result.days_until_expiry <= 91


def test_freshly_issued_certificate_age_reported():
    result = _build("example.com", _cert(age_days=1), "TLSv1.3",
                    verified=True, tls_error=None)
    assert result.cert_age_days <= 1


def test_expired_certificate_has_negative_days_remaining():
    result = _build("example.com", _cert(valid_days=-5), "TLSv1.2",
                    verified=False, tls_error="cert expired")
    assert result.days_until_expiry < 0
    assert result.cert_valid is False
    assert result.tls_error == "cert expired"


def test_self_signed_detected():
    now = datetime.now(timezone.utc)
    same = ((("commonName", "self.example"),),)
    cert = {
        "subject": same, "issuer": same,
        "notBefore": _cert_time(now - timedelta(days=1)),
        "notAfter": _cert_time(now + timedelta(days=364)),
        "subjectAltName": (("DNS", "self.example"),),
    }
    result = _build("self.example", cert, "TLSv1.3", verified=False, tls_error="self signed")
    assert result.is_self_signed is True


def test_hostname_mismatch_recorded():
    result = _build("evil-lookalike.com", _cert(host="realbank.com"), "TLSv1.3",
                    verified=False, tls_error="hostname mismatch")
    assert result.hostname_matches is False
    assert result.cert_valid is False


def test_unparseable_dates_do_not_raise():
    cert = {"subject": ((("commonName", "x.com"),),), "issuer": ((("commonName", "y"),),),
            "notBefore": "garbage", "notAfter": None, "subjectAltName": ()}
    result = _build("x.com", cert, "TLSv1.3", verified=True, tls_error=None)
    assert result.cert_age_days is None
    assert result.days_until_expiry is None


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def test_cache_roundtrip():
    original = _build("example.com", _cert(), "TLSv1.3", verified=True, tls_error=None)
    restored = TLSResult.from_cache(original.to_cache())

    assert restored.issuer == original.issuer
    assert restored.cert_valid == original.cert_valid
    assert restored.subject_alt_names == original.subject_alt_names
    assert restored.cert_age_days == original.cert_age_days


def test_feature_dict_keys_are_prefixed():
    d = _build("example.com", _cert(), "TLSv1.3", verified=True,
               tls_error=None).feature_dict()
    assert all(k.startswith("tls_") for k in d)
