from rich.console import Console
from src.config import settings
from src.indexer import get_chroma_client, get_or_create_collection
from src.embedder import embed_query


console = Console()


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    Args:
        query: User's question
        top_k: No of results to return

    Returns:
        list[dict]: list of dicts, each with text, metadata and score.
    """

    # default top_k if top_k not provided
    top_k = top_k or settings.top_k

    query_embedding = embed_query(query)

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # format results

    retrieved = []

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved.append(
                {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            )

    return retrieved


if __name__ == "__main__":
    from src.parser import parse_pdf
    from src.chunker import chunk_text
    from src.indexer import index_chunks
    import questionary

    markdown_text = parse_pdf("data/input.pdf")
    chunks = chunk_text(markdown_text)

    index_chunks(chunks)

    query = questionary.text("Question:").ask()

    findings = retrieve(query)

    print(type(findings))
    print(findings)
