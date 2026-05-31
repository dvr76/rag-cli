from rich.console import Console
from rich.panel import Panel
import questionary
from rich.table import Table
from rich.markdown import Markdown
from src.rag import ingest_pdf, ask, reset_index
from pathlib import Path

console = Console()


def show_banner():
    console.print(
        Panel(
            "[bold green]RAG-CLI[/bold green] - Ask questions about your PDF\n"
            "[dim]Simple CLI for Simple RAG[/dim]",
            border_style="blue",
        )
    )


def handle_ingest():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        console.print(
            "[yellow]No PDF files found in ./data/ directory[/yellow]\n"
            "Place your PDFs in the [bold]data/[/bold] folder and try again."
        )
        return

    choices = [f.name for f in pdf_files]
    choices.append("Back")

    selected = questionary.select("Select PDF to ingest:", choices=choices).ask()

    if selected == "Back" or selected is None:
        return

    pdf_path = data_dir / selected

    if questionary.confirm(f"Ingest {selected}?").ask():
        with console.status("[bold]Processing...[/bold]"):
            count = ingest_pdf(pdf_path)
        console.print(f"\n[bold green]DONE:[/bold green] {count} chunks indexed.\n")


def handle_query():
    question = questionary.text(
        "Your question:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a question",
    ).ask()

    if not question:
        return

    with console.status("[bold]Thinking...[/bold]"):
        result = ask(question)

    console.print()
    console.print(
        Panel(
            Markdown(result["answer"]),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    if result["sources"]:
        table = Table(title="Sources", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Relevance", justify="right", style="green")
        table.add_column("Preview", max_width=60)

        for i, src in enumerate(result["sources"], 1):
            table.add_row(
                str(i),
                src["source"],
                f"{src['score']:.2%}",
                src["text"][:100] + "...",
            )

        console.print(table)
    console.print()


def handle_reset():
    if questionary.confirm(
        "Delete ALL indexed documents? This cannot be undone.", default=False
    ).ask():
        reset_index()
        console.print("[bold green]DONE:[/bold green] Collection cleared.")


def main():
    show_banner()

    while True:
        action = questionary.select(
            "Select an option:", choices=["Ingest a PDF", "Exit"]
        ).ask()

        if action is None or action == "Exit":
            break
        elif "Ingest" in action:
            handle_ingest()
        elif "Ask" in action:
            handle_query()
        elif "Reset" in action:
            handle_reset()
