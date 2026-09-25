"""Collect the three official EU web-accessibility legal instruments."""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
LEGAL_SOURCES = {
    "directive_eu_2016_2102.pdf": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Directive_(EU)_2016-2102_of_the_European_Parliament_and_of_the_Council_of_26_October_2016_on_the_accessibility_of_the_websites_and_mobile_applications_of_public_sector_bodies_(Text_with_EEA_relevance)_(EUDR_2016-2102).pdf",
    "implementing_decision_eu_2018_1523.pdf": "https://commons.wikimedia.org/wiki/Special:Redirect/file/EUD_2018-1523.pdf",
    # OJ L 256 contains both 2018/1523 (p. 103) and 2018/1524 (p. 108).
    "official_journal_l256_2018.pdf": "https://commons.wikimedia.org/wiki/Special:Redirect/file/OJ_L_256_of_2018_-_EN_English.pdf?download=1",
}

CANONICAL_LEGAL_SOURCES = {
    "directive_eu_2016_2102.pdf": "https://eur-lex.europa.eu/eli/dir/2016/2102/oj/eng",
    "implementing_decision_eu_2018_1523.pdf": "https://eur-lex.europa.eu/eli/dec_impl/2018/1523/oj/eng",
    "official_journal_l256_2018.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L:2018:256:TOC",
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Download official PDFs, validating responses before replacing files."""
    setup_directory()
    # Commons hosts unmodified OGL copies sourced from legislation.gov.uk.
    # A descriptive User-Agent is required by Wikimedia's automated-access policy.
    headers = {"User-Agent": "K4RAGCourseProject/1.0 (academic document collection)"}
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
