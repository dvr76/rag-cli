from pathlib import Path
from rich.console import Console
from src.parser import parse_pdf
from src.chunker import chunk_text
from src.indexer import index_chunks, clear_collection


console = Console()


def ingest_pdf(pdf_path: str | Path) -> int:
    """
    Step 1: Ingest pdf into vector db

    Args:
        pdf_path: Path to the PDF file

    Returns:
        int: Number of chunks indexed
    """

    pdf_path = Path(pdf_path)

    console.rule(f"[bold] Ingesting: {pdf_path.name}[/bold]")

    text = parse_pdf(pdf_path)

    chunks = chunk_text(text, source_name=pdf_path.name)

    count = index_chunks(chunks)

    console.rule("[bold green]Ingestion complete[/bold green]")
    return count


def reset_index():
    clear_collection()
