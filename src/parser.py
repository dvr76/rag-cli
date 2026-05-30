import pymupdf4llm  # type: ignore[import-untyped]
from pathlib import Path
from rich.console import Console
from typing import cast

console = Console()


def parse_pdf(pdf_path: str | Path) -> str:
    """
    Parse PDF into markdown

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Markdown string of the whole document
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    console.print(f"[bold]Parsing:[/bold] {pdf_path.name}")

    md_text: str = cast(str, pymupdf4llm.to_markdown(str(pdf_path)))

    console.print(f"[bold green]DONE:[/bold green] Extracted {len(md_text):,}")

    return md_text


if __name__ == "__main__":
    text = parse_pdf("data/input.pdf")
