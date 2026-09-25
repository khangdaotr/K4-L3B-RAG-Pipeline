"""Collect the three official EU web-accessibility legal instruments."""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
LEGAL_SOURCES = {
    "directive_eu_2016_2102.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016L2102",
    "implementing_decision_eu_2018_1523.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32018D1523",
    "implementing_decision_eu_2018_1524.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32018D1524",
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Download official PDFs, validating responses before replacing files."""
    setup_directory()
    headers = {"User-Agent": "K4-RAG-course-project/1.0 (document collection)"}
    for filename, url in LEGAL_SOURCES.items():
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF-") or len(content) <= 1024:
            raise ValueError(f"EUR-Lex did not return a valid PDF for {url}")
        output = DATA_DIR / filename
        output.write_bytes(content)
        print(f"Saved: {output}")


if __name__ == "__main__":
    download_documents()
