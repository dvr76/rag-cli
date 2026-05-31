from sentence_transformers import SentenceTransformer
from rich.console import Console
from src.config import settings


console = Console()

# Load model once
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Loads the embedding model
    """

    global _model

    if _model is None:
        console.print(
            f"[bold]Loading Embedding model:[/bold] {settings.embedding_model}"
        )

        _model = SentenceTransformer(settings.embedding_model)

        # gpu check
        device = _model.device
        console.print(f"[bold green]DONE:[/bold green] Model loaded on: {device}")

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of texts into embedding vectors.

    Args:
        texts: List of texts to embed.

    Returns:
        list[list[float]]: List of embedding vectors (384 dims).
    """

    model = _get_model()

    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: query string

    Returns:
        list[float]: Embedding vector
    """

    model = _get_model()

    embedding = model.encode(query, convert_to_numpy=True)

    return embedding.tolist()


if __name__ == "__main__":
    text_list = ["Hello world testing embedding", "python is a programming language"]
    query = "What is python?"

    embedded_text = embed_texts(text_list)
    embedded_query = embed_query(query)

    console.print(embed_texts)
