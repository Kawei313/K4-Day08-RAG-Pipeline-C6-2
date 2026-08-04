"""
Task 3 - Convert all files in data/landing/ to Markdown.

Input:
    data/landing/legal/*.pdf, *.docx, *.doc
    data/landing/news/*.json

Output:
    data/standardized/legal/*.md
    data/standardized/news/*.md
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

SUPPORTED_LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _write_markdown(output_path: Path, content: str):
    """Write Markdown with UTF-8 encoding and a trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")


def convert_legal_docs():
    """Convert PDF/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  Skipped: missing directory {legal_dir}")
        return

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in SUPPORTED_LEGAL_EXTENSIONS:
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        text_content = getattr(result, "text_content", "") or ""

        header = f"# {filepath.stem}\n\n"
        header += f"**Source file:** {filepath.name}\n"
        header += "**Document type:** legal\n\n"
        header += "---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        _write_markdown(output_path, header + text_content)
        print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  Skipped: missing directory {news_dir}")
        return

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))

        title = data.get("title") or filepath.stem
        source_url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        content_markdown = data.get("content_markdown") or data.get("content") or ""

        header = f"# {title}\n\n"
        header += f"**Source:** {source_url}\n"
        header += f"**Crawled:** {date_crawled}\n"
        header += "**Document type:** news\n\n"
        header += "---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        _write_markdown(output_path, header + content_markdown)
        print(f"  Saved: {output_path}")


def convert_all():
    """Convert all supported landing files to Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\nDone! Output at: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
