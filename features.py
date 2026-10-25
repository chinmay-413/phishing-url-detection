import re, math, difflib
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import tldextract

tld = tldextract.TLDExtract(suffix_list_urls=())   # bundled suffix list: works offline

SUSPICIOUS_WORDS = ["login", "signin", "verify", "update", "secure", "account", "bank", "free",
                    "bonus", "confirm", "password", "wallet", "support", "billing", "suspend",
                    "unlock", "recover", "invoice", "payment", "alert"]
BRANDS = ["google", "paypal", "amazon", "facebook", "microsoft", "apple", "netflix", "instagram",
          "whatsapp", "linkedin", "twitter", "youtube", "gmail", "outlook", "dropbox", "icloud",
          "ebay", "sbi", "hdfcbank", "icicibank", "axisbank", "paytm", "phonepe", "flipkart",
          "irctc", "airtel", "coinbase", "binance", "metamask", "spotify", "adobe", "docusign",
          "dhl", "fedex", "chase", "wellsfargo", "bankofamerica", "citibank", "steam", "discord",
          "telegram"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
              "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy"}
RISKY_TLDS = {"zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "country",
              "kim", "loan", "review", "stream", "download", "men", "party", "racing", "date",
              "icu", "buzz", "rest", "cyou", "monster", "support", "bid", "win"}
POPULAR = {"google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com", "apple.com",
           "microsoft.com", "twitter.com", "x.com", "instagram.com", "linkedin.com", "netflix.com",
           "github.com", "stackoverflow.com", "reddit.com", "yahoo.com", "bing.com", "whatsapp.com",
           "paypal.com", "ebay.com", "bbc.com", "bbc.co.uk", "nytimes.com", "cnn.com", "zoom.us",
           "adobe.com", "dropbox.com", "spotify.com", "openai.com", "anthropic.com", "kaggle.com",
           "python.org", "mozilla.org", "cloudflare.com", "flipkart.com", "irctc.co.in",
           "sbi.co.in", "hdfcbank.com", "icicibank.com", "paytm.com", "gov.in", "nic.in"}

LEGACY_COLS = ["url_length", "at_symbol", "sensitive_words_count", "path_length", "nb_dots",
               "nb_hyphens", "nb_and", "nb_www", "nb_com", "nb_underscore"]
_LEET = str.maketrans("013457@$", "oleatsas")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _parts(url):
    url = str(url).strip()
    u = url if "://" in url else "http://" + url
    try:
        p = urlparse(u)
    except ValueError:      # malformed URL, e.g. stray brackets
        try:
            p = urlparse(u.replace("[", "").replace("]", ""))
        except ValueError:
            p = urlparse("http://invalid")
    return url, u, p, tld(u)


def _entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _lookalike(domain):
    d = domain.lower()
    if d in BRANDS:
        return None
    dn = d.translate(_LEET)
    if dn in BRANDS:
        return dn
    near = difflib.get_close_matches(d, BRANDS, n=1, cutoff=0.86)
    return near[0] if near and abs(len(near[0]) - len(d)) <= 2 else None


def lookalike_brand(url):
    return _lookalike(_parts(url)[3].domain)


def is_popular(url):
    ext = _parts(url)[3]
    return ext.top_domain_under_public_suffix in POPULAR and ext.subdomain in ("", "www")


def text_norm(url):
    s = str(url).strip().lower()
    s = re.sub(r"^[a-z]+://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/")


def extract(url):
    url, u, p, ext = _parts(url)
    host = p.netloc.split("@")[-1].split(":")[0].lower()
    path, query = p.path, p.query
    low = url.lower()
    dom, sub = ext.domain.lower(), ext.subdomain.lower()
    n = max(len(url), 1)
    brand_host = int(any(b in (sub + "." + dom) for b in BRANDS) and dom not in BRANDS)
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": path.count("/"),
        "num_params": query.count("&") + 1 if query else 0,
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / n,
        "num_special": len(re.findall(r"[@?=&%~]", url)),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))),
        "has_port": int(":" in p.netloc.split("@")[-1]),
        "has_punycode": int("xn--" in host),
        "hex_encoded": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "num_subdomains": len([s for s in sub.split(".") if s]),
        "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "tld_length": len(ext.suffix),
        "risky_tld": int(ext.suffix.split(".")[-1] in RISKY_TLDS),
        "num_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "brand_in_host": brand_host,
        "brand_lookalike": int(_lookalike(dom) is not None),
        "is_shortener": int(ext.top_domain_under_public_suffix in SHORTENERS),
        "entropy_host": _entropy(host),
        "entropy_url": _entropy(low),
        "longest_token": max((len(t) for t in re.findall(r"[a-z0-9]+", host)), default=0),
        "has_www": int(host.startswith("www.")),
    }


def legacy_features(url):
    """The 10 columns that also exist in phishing_url_dataset.csv (used for pooled training)."""
    url, u, p, ext = _parts(url)
    low = url.lower()
    return {
        "url_length": len(url),
        "at_symbol": int("@" in url),
        "sensitive_words_count": sum(w in low for w in SUSPICIOUS_WORDS),
        "path_length": len(p.path),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_and": url.count("&"),
        "nb_www": low.count("www"),
        "nb_com": low.count(".com"),
        "nb_underscore": url.count("_"),
    }


def red_flags(url):
    """Human-readable warning signs: list of (severity, message)."""
    f = extract(url)
    _, _, p, ext = _parts(url)
    flags = []
    brand = _lookalike(ext.domain)
    if brand:
        flags.append(("high", f"Domain '{ext.domain}' imitates the brand '{brand}'"))
    elif f["brand_in_host"]:
        flags.append(("high", "A famous brand name appears in the host but this is not the brand's own domain"))
    if f["has_ip"]:
        flags.append(("high", "Uses a raw IP address instead of a domain name"))
    if f["has_at"]:
        flags.append(("high", "Contains '@', which can hide the real destination"))
    if f["has_punycode"]:
        flags.append(("medium", "Uses punycode (xn--), which can disguise look-alike characters"))
    if f["is_shortener"]:
        flags.append(("medium", "Link shortener: the real destination is hidden"))
    if f["risky_tld"]:
        flags.append(("medium", f"Uses '.{ext.suffix}', a domain ending often abused for scams"))
    if f["num_suspicious_words"] >= 2:
        flags.append(("medium", "Several pressure/credential words (login, verify, secure, ...)"))
    if f["num_subdomains"] >= 3:
        flags.append(("medium", "Unusually many subdomains"))
    if f["host_hyphens"] >= 2:
        flags.append(("low", "Many hyphens in the domain name"))
    if f["url_length"] > 100:
        flags.append(("low", "Very long URL"))
    if f["hex_encoded"] >= 3:
        flags.append(("low", "Lots of %-encoded characters"))
    if f["entropy_host"] > 3.5 and f["host_length"] > 14:
        flags.append(("low", "Random-looking domain name"))
    if f["has_port"]:
        flags.append(("low", "Uses an explicit port number"))
    return flags