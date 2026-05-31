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


def retrieve_multi(
    queries: list[str],
    k_per_query: int = 5,
    limit: int = 10,
    min_score: float = 0.4,
) -> list[dict]:
    """
    Run multiple subqueries, merge, dedup and rerank results

    Args:
        queries: List of sub-queries from decomp
        k_per_query: Results per subquery
        limit: Max chunks to return after merging
        min_score: Minimum relevance score

    Returns:
        list[dict]: Merged, deduplicated, reranked chunks
    """

    seen_texts: set[str] = set()
    merged: list[dict] = []

    for i, query in enumerate(queries):
        console.print(f"   [dim]Subquery {i + 1}/{len(queries)}:[/dim] {query}")
        results = retrieve(query, top_k=k_per_query)

        for chunk in results:
            text_key = chunk["text"].strip()
            if text_key not in seen_texts and chunk["score"] >= min_score:
                seen_texts.add(text_key)
                merged.append(chunk)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:limit]


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
