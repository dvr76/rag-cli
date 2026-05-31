from pathlib import Path
from rich.console import Console
from src.parser import parse_pdf
from src.chunker import chunk_text
from src.indexer import index_chunks, clear_collection
from src.retriever import retrieve, retrieve_multi
from src.generator import generate, decompose_query

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

    console.rule(f"[bold]Ingesting: {pdf_path.name}[/bold]")

    text = parse_pdf(pdf_path)

    chunks = chunk_text(text, source_name=pdf_path.name)

    count = index_chunks(chunks)

    console.rule("[bold green]Ingestion complete[/bold green]")
    return count


def ask(question: str) -> dict:
    """
    Step 2: Ask question against indexed documents.

    Args:
        question: User's question

    Returns:
        dict: dictionary with answer, sources and questions.
    """
    console.rule("[bold]Querying[/bold]")

    sub_queries = decompose_query(question)

    if len(sub_queries) == 1:
        chunks = retrieve(sub_queries[0])
    else:
        chunks = retrieve_multi(
            queries=sub_queries, k_per_query=5, limit=10, min_score=0.4
        )

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found. Have you ingested a PDF yet?",
            "sources": [],
        }

    answer = generate(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "source": c["metadata"].get("source", "unknown"),
                "score": c["score"],
            }
            for c in chunks
        ],
    }


def reset_index():
    clear_collection()
