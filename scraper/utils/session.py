import re
import requests
from config.headers import headers

_TOKEN_RE = re.compile(r'name="_token"\s+value="([^"]+)"')

BASE = "https://www.sharesansar.com"
TIMEOUT = 30


def extract_token(html):
    m = _TOKEN_RE.search(html)
    return m.group(1) if m else None


def make_session():
    session = requests.Session()
    session.headers.update(headers)
    return session


def prime_session(session, symbol="adbl"):
    resp = session.get(f"{BASE}/company/{symbol.lower()}", timeout=TIMEOUT)
    resp.raise_for_status()
    return extract_token(resp.text)
