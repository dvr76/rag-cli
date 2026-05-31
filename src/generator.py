from openai import OpenAI
from rich.console import Console
from src.config import settings
import json

console = Console()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Use ONLY information from the context below to answer.
2. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer that."
3. Be info-dense, concise and direct.
4. When referencing information, mention which part of the context it came from.
5. Do not make up information or use your training data to fill gaps."""


DECOMPOSE_PROMPT = """You are a search query decomposer for a car owner's manual RAG system.

Break the user's question into 2-4 specific, focused sub-queries that will each retrieve 
different relevant sections of the manual. Each sub-query should target a distinct aspect.

Rules:
- Return ONLY a valid JSON array of strings
- Each string is one sub-query
- Use terminology likely found in a car manual
- If the question is already simple and specific, return an array with just that one question
- Maximum 4 sub-queries

Example input:
"What are all the ways a user account can be suspended, and who can reinstate it?"

Example output:
["account suspension conditions triggers", "automatic account suspension criteria", "account reinstatement process authority"]"""


def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )


def decompose_query(query: str) -> list[str]:
    """
    Use LLM to decompose a complex query into focused subqueries.

    Args:
        query: Original user question

    Returns:
        list[str]: List of sub-queries (falls back to [query] on failure)
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=256,
        )

        content = response.choices[0].message.content

        if content is None:
            return [query]

        sub_queries = json.loads(content)

        # validate it's a list of strings
        if isinstance(sub_queries, list) and all(
            isinstance(q, str) for q in sub_queries
        ):
            console.print(f"[dim]Decomposed into {len(sub_queries)} sub-queries[/dim]")
            return sub_queries

    except (json.JSONDecodeError, Exception) as e:
        console.print(
            f"[dim yellow]Query decomposition failed ({e}), using original query[/dim yellow]"
        )

    return [query]


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the user prompt with retrieved context

    Args:
        query: User's query
        context_chunks: List of retrieved chunks

    Returns:
        str: contextful prompt

    """

    context_parts = []

    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        score = chunk.get("score", 0)
        context_parts.append(
            f"[Chunk {i}] (source: {source}, relavance: {score})\n{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""Context from documents:
    {context_str}

    ---
    Question: {query}

    Answer based only on the context above:"""


def generate(query: str, context_chunks: list[dict]) -> str:
    """
    Send augmented prompt to LLM

    Args:
        query: User's question
        context_chunks: List of retrieved chunks

    Returns:
        str: LLM response
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key
    )

    user_prompt = build_prompt(query, context_chunks)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        content = response.choices[0].message.content

        if content is None:
            console.print("[bold red]ERROR:[/bold red] LLM returned empty response")
            return "LLM returned no content."

        return content

    except Exception as e:
        error_msg = f"LLM generation failed: {e}"
        console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
        raise


if __name__ == "__main__":
    import questionary
    from src.retriever import retrieve
    from rich.markdown import Markdown
    from rich.panel import Panel

    query = questionary.text("Question:").ask()

    context_chunks = retrieve(query)

    output = generate(query, context_chunks)

    console.print(
        Panel(
            Markdown(output),
            title="LLM Output",
        )
    )

    with open(".internal/output.md", "w", encoding="utf-8") as f:
        f.write(output)
