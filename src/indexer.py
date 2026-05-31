import chromadb
from chromadb.api import ClientAPI
from rich.console import Console
from src.config import settings
from src.embedder import embed_texts


console = Console()


def get_chroma_client() -> ClientAPI:
    return chromadb.HttpClient(
        host=settings.chroma_host, port=settings.chroma_port, ssl=False
    )


def get_or_create_collection(client: ClientAPI):
    return client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: list[dict]) -> int:
    """
    Embed and store chunks into chromadb

    Args:
        chunks: list of chunk dicts

    Returns:
        int: number of chunks indexed
    """

    if not chunks:
        console.print("[bold yellow]WARN:[/bold yellow] No chunks to index.")
        return 0

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    texts = [c["text"] for c in chunks]

    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]

    metadatas = [c["metadata"] for c in chunks]

    console.print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # upsert into chromadb
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    count = collection.count()
    console.print(
        f"[bold green]DONE:[/bold green] Indexed {len(chunks)} chunks."
        f"Collection total: {count}"
    )
    return len(texts)


def clear_collection():
    """clear collection for reindexing"""
    client = get_chroma_client()
    client.delete_collection(settings.collection_name)
    console.print(
        f"[bold yellow]DELETED COLLECTION:[/bold yellow] {settings.collection_name}"
    )


if __name__ == "__main__":
    from src.parser import parse_pdf
    from src.chunker import chunk_text
    import questionary

    markdown_text = parse_pdf("data/input.pdf")
    chunks = chunk_text(markdown_text)

    index_chunks(chunks)

    answer = questionary.confirm("Clear the collection?").ask()

    if answer:
        clear_collection()
