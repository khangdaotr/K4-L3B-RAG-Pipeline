"""Normalize legal documents and guidance pages to traceable Markdown."""

import json
from pathlib import Path

from markitdown import MarkItDown

from src.task1_collect_legal_docs import LEGAL_SOURCES

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        body = converter.convert(str(path)).text_content.strip()
        if not body:
            raise ValueError(f"Conversion produced empty content: {path}")
        source = LEGAL_SOURCES.get(path.name, "Unknown (locally supplied document)")
        header = f"<!-- landing: legal/{path.name} -->\n\n**Source:** {source}\n\n---\n\n"
        (output_dir / f"{path.stem}.md").write_text(header + body + "\n", encoding="utf-8")


def convert_news_articles() -> None:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path.name} missing fields: {sorted(missing)}")
        body = str(data["content_markdown"]).strip()
        if not body:
            raise ValueError(f"Empty article content: {path}")
        header = (
            f"<!-- landing: news/{path.name} -->\n\n"
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(header + body + "\n", encoding="utf-8")


def convert_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
