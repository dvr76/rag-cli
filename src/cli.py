from rich.console import Console
from rich.panel import Panel
import questionary

console = Console()


def show_banner():
    console.print(
        Panel(
            "[bold green]RAG-CLI[/bold green] - Ask questions about your PDF\n"
            "[dim]Simple CLI for Simple RAG[/dim]",
            border_style="blue",
        )
    )


def main():
    show_banner()

    while True:
        action = questionary.select(
            "Select an option:", choices=["Ingest a PDF", "Exit"]
        ).ask()

        if action is None or action == "Exit":
            break
