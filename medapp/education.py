"""Patient-education links from the NIH/NLM MedlinePlus search web service (free, no API key)."""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from html import unescape

import requests

API = "https://wsearch.nlm.nih.gov/ws/query"


@lru_cache(maxsize=256)
def lookup(term: str) -> dict | None:
    """Top MedlinePlus health-topic hit for `term` -> {title, url, summary}; None on any failure."""
    try:
        r = requests.get(API, params={"db": "healthTopics", "term": term, "retmax": 1}, timeout=5)
        r.raise_for_status()
        doc = ET.fromstring(r.text).find(".//document")
        if doc is None:
            return None
        strip = lambda s: re.sub(r"<[^>]+>", "", unescape(s or "")).strip()
        field = lambda n: next((c.text for c in doc.findall("content") if c.get("name") == n), "")
        summary = strip(field("FullSummary"))
        return {"title": strip(field("title")), "url": doc.get("url"),
                "summary": summary[:280] + ("…" if len(summary) > 280 else "")}
    except (requests.RequestException, ET.ParseError):
        return None
