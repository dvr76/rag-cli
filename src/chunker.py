from rich.console import Console
from src.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter


console = Console()


def chunk_text(text: str, source_name: str = "unknown") -> list[dict]:
    """
    Split text into overlapping chunks with metadata

    Args:
        text: Markdown text of document
        source_name: File name for tracking

    Returns:
        list[dict]: list of chunk dicts

    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = []

    for i, chunk_text_content in enumerate(raw_chunks):
        stripped = chunk_text_content.strip()
        if not stripped or len(stripped) < 20:
            continue
        chunks.append(
            {
                "text": stripped,
                "metadata": {
                    "source": source_name,
                    "chunk_index": i,
                },
            }
        )

    console.print(
        f"[bold green]DONE:[/bold green] Chunks: {len(chunks)}"
        f"(size={settings.chunk_size}, overlan={settings.chunk_overlap})"
    )

    return chunks


if __name__ == "__main__":
    from src.parser import parse_pdf
    from rich.panel import Panel

    text = parse_pdf("data/input.pdf")

    chunks = chunk_text(text, "input.pdf")

    first_chunk = chunks[0]

    console.print(Panel(f"Text{first_chunk['text'][:60]}", title="First Chunk"))

    console.print(Panel(f"Text{first_chunk['metadata']}", title="Metadata"))
