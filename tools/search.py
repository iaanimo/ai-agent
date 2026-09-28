"""
Search Tool
============
Web search. Primary backend is Bing China (cn.bing.com) — reachable from
mainland networks without an API key. Falls back to DuckDuckGo if Bing fails.
"""

import asyncio
import time
import urllib.parse
import urllib.request
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class SearchTool:
    """Web search tool (Bing China primary, DuckDuckGo fallback)."""

    name = "search"
    description = ("Search the web for current information. Write short keyword queries "
                   "(e.g. 'AI 新闻 2026') — long natural-language sentences get worse results. "
                   "For recent news, include '新闻' or 'news'. Results may be mixed; "
                   "they usually include relative dates (e.g. '18 小时之前', '2026年8月1日') "
                   "— use those to judge how recent each result is. 2-3 searches are enough.")

    # Chinese markers of navigation/directory sites (noise), filtered out of results.
    _JUNK_MARKERS = [
        "导航", "网址大全", "网站大全", "工具集", "工具箱", "工具库",
        "导航站", "导航网", "资源导航", "网址导航",
        "网站汇总", "资源汇总", "网址推荐", "网站推荐", "大全", "合集",
    ]

    @staticmethod
    async def search(query: str, max_results: int = 5) -> str:
        """Perform a web search. Tries Bing China first, then DuckDuckGo."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, SearchTool._search_bing, query, max_results)
        except Exception as e:
            try:
                return await loop.run_in_executor(None, SearchTool._search_ddg, query, max_results)
            except Exception as e2:
                return (f"Search error: Bing: {type(e).__name__}: {str(e)[:120]} | "
                        f"DuckDuckGo: {type(e2).__name__}: {str(e2)[:120]}")

    # ─── Bing China ────────────────────────────────────────────

    @staticmethod
    def _search_bing(query: str, max_results: int) -> str:
        """Search via cn.bing.com and parse results from HTML.

        Retries up to 3 times with a short backoff, so transient network
        hiccups don't turn into "search failed".
        """
        from bs4 import BeautifulSoup

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                params = urllib.parse.urlencode({"q": query, "mkt": "zh-CN", "setlang": "zh-hans"})
                url = f"https://cn.bing.com/search?{params}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

                with urllib.request.urlopen(req, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")

                return SearchTool._parse_bing(html, max_results)
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise last_err if last_err else RuntimeError("search failed")

    @staticmethod
    def _is_junk(title: str, href: str) -> bool:
        """True if a result looks like a navigation/directory site (noise)."""
        text = f"{title} {href}".lower()
        return any(m in text for m in SearchTool._JUNK_MARKERS)

    @staticmethod
    def _parse_bing(html: str, max_results: int) -> str:
        """Parse Bing search results from HTML, filtering junk sites."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        results = []
        # Scan extra candidates so filtering junk doesn't reduce the count.
        for li in soup.select("li.b_algo"):
            h2 = li.find("h2")
            if not h2:
                continue
            a = h2.find("a", href=True)
            if not a:
                continue
            title = a.get_text(" ", strip=True)
            href = a["href"]
            if not href.startswith("http"):
                continue
            if SearchTool._is_junk(title, href):
                continue
            snippet = ""
            p = li.find("p")
            if p:
                snippet = p.get_text(" ", strip=True)
            results.append(f"{title}\n   {href}\n   {snippet}")
            if len(results) >= max_results:
                break

        if not results:
            return "No results found."
        return "\n\n".join(results)

    # ─── DuckDuckGo (fallback) ─────────────────────────────────

    @staticmethod
    def _search_ddg(query: str, max_results: int) -> str:
        """Search via DuckDuckGo (works where Bing is unavailable)."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(f"{i}. {r.get('title', 'No title')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
            return "\n\n".join(formatted)


def create_search_tool() -> StructuredTool:
    """Create a search tool instance."""
    return StructuredTool(
        name=SearchTool.name,
        description=SearchTool.description,
        func=SearchTool.search,
        coroutine=SearchTool.search,
        args_schema=SearchInput,
    )
