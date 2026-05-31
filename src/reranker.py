from sentence_transformers import CrossEncoder

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _model


def rerank(query: str, chunks: list[dict], limit: int = 5) -> list[dict]:
    """
    Rerank the retrieved chunks using cross encoder.

    Args:
        query: Original user question
        chunks: chunks from retrieve multi
        limit: output limit

    Returns:
        list[dict]: Top N reranked chunks
    """

    if not chunks:
        return chunks

    model = get_reranker()

    pairs = [(query, str(chunk["text"])) for chunk in chunks]
    scores = model.predict(pairs)  # type: ignore[arg-type]

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:limit]
