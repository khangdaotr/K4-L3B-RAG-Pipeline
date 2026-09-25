"""Collect public guidance pages about EU web accessibility."""

import asyncio
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
ARTICLE_URLS = [
    "https://digital-strategy.ec.europa.eu/en/policies/web-accessibility",
    "https://digital-strategy.ec.europa.eu/en/policies/web-accessibility-directive-standards-and-harmonisation",
    "https://digital-strategy.ec.europa.eu/en/policies/web-accessibility-monitoring",
    "https://www.w3.org/WAI/standards-guidelines/wcag/",
    "https://www.w3.org/WAI/test-evaluate/",
]


class _ReadableHTML(HTMLParser):
    """Dependency-free conversion of meaningful HTML blocks to Markdown."""

    BLOCKS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip = 0
        self._tag = ""
        self._buffer: list[str] = []
        self.lines: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCKS and not self._skip:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCKS and not self._skip:
            self._flush()
        if tag in {"script", "style", "svg", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif self._tag:
            self._buffer.append(text)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", " ".join(self._buffer)).strip()
        if text:
            if self._tag.startswith("h"):
                text = f"{'#' * int(self._tag[1])} {text}"
            elif self._tag == "li":
                text = f"- {text}"
            elif self._tag == "blockquote":
                text = f"> {text}"
            if not self.lines or self.lines[-1] != text:
                self.lines.append(text)
        self._buffer = []
        self._tag = ""


async def crawl_article(url: str) -> dict:
    """Fetch one public page and return the required landing schema."""

    def fetch() -> dict:
        response = requests.get(
            url,
            headers={"User-Agent": "K4-RAG-course-project/1.0 (article collection)"},
            timeout=60,
        )
        response.raise_for_status()
        # Some public-sector pages omit a charset and requests otherwise falls
        # back to ISO-8859-1, corrupting typographic punctuation.
        response.encoding = response.apparent_encoding or "utf-8"
        parser = _ReadableHTML()
        parser.feed(response.text)
        parser._flush()
        markdown = "\n\n".join(parser.lines).strip()
        if len(markdown) < 200:
            raise ValueError("Extracted article is unexpectedly short")
        first_heading = next(
            (line[2:].strip() for line in parser.lines if line.startswith("# ")),
            None,
        )
        return {
            "url": url,
            "title": first_heading or parser.title or url,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": markdown,
        }

    return await asyncio.to_thread(fetch)


async def crawl_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
