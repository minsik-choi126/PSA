"""Web search tool: thin wrapper around Serper.dev (Google Search API).

When SERPER_API_KEY is set in .env, search() returns a JSON-friendly dict of
top results. Without the key, returns an empty list so the agent can still
attempt closed-book QA.

Optional Crawl4AI fetcher for deep-dive on a result URL.
"""
from __future__ import annotations
import os, json, time, requests
from typing import List, Dict


def search(query: str, k: int = 5, timeout: int = 15) -> List[Dict]:
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        return []
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": k}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"error": f"serper failed: {type(e).__name__}: {e}"}]
    out = []
    for item in (data.get("organic") or [])[:k]:
        out.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link"),
        })
    if data.get("answerBox"):
        ab = data["answerBox"]
        out.insert(0, {"title": ab.get("title"), "snippet": ab.get("answer") or ab.get("snippet"), "link": ab.get("link")})
    return out


def fetch_url(url: str, timeout: int = 20, max_chars: int = 8000) -> str:
    """Best-effort plain-text fetcher. Tries Crawl4AI first, falls back to requests+BeautifulSoup."""
    try:
        from crawl4ai import WebCrawler  # type: ignore
        crawler = WebCrawler(verbose=False)
        crawler.warmup()
        result = crawler.run(url=url)
        text = (result.markdown or result.cleaned_html or "")[:max_chars]
        if text.strip():
            return text
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup  # type: ignore
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception as e:
        return f"[fetch_url failed: {type(e).__name__}: {e}]"
