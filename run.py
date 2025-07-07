#!/usr/bin/env python3
"""
KSF v4.x "Unified Resonance Framework" Demo Script
Demonstrates the new architecture with a VectorDBConnector and a three-tiered
knowledge resonance model. Allows for interactive exploration of emerged concepts.
"""
import sys
import os
import logging
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add project root to Python path to ensure module imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Logging Setup ---
# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ksf_demo.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# --- Import Core KSF Components ---
try:
    from ksf.core.orchestrator import KSFOrchestrator
    from ksf.k_module.data_structures import ResonancePacket, EmergedConcept, RerankedItem
    logger.info("✓ Core KSF components imported successfully.")
except ImportError as e:
    logger.error(f"❌ Failed to import core KSF components: {e}", exc_info=True)
    sys.exit(1)


def display_resonance_packet(packet: ResonancePacket, console: Console):
    """Renders the contents of the ResonancePacket in a visually appealing format."""
    
    if not any([packet.primary_atoms, packet.context_atoms, packet.emerged_concepts]):
        console.print(Panel("[dim]K-Module did not return any resonant items.[/dim]", title="[bold yellow]Resonance Packet[/bold yellow]", border_style="yellow", expand=False))
        return

    panel_content = ""
    
    # --- Primary Atoms ---
    if packet.primary_atoms:
        panel_content += f"\n[bold bright_green]⚛️ Primary Knowledge ({len(packet.primary_atoms)}):[/bold bright_green]\n"
        for atom in packet.primary_atoms:
            content_preview = atom.content.replace('\n', ' ').strip()
            if len(content_preview) > 70:
                content_preview = content_preview[:67] + "..."
            panel_content += (f"  - [green]ID[/green]: {atom.id} | "
                              f"[green]Score[/green]: {atom.final_score:.3f} "
                              f"([dim]Sq:{atom.original_similarity:.2f}, Ss:{atom.pagerank_weight:.2f}[/dim])\n"
                              f"    [dim]{content_preview}[/dim]\n")
    
    # --- Context Atoms ---
    if packet.context_atoms:
        panel_content += f"\n[bold bright_blue]🧭 Contextual Knowledge ({len(packet.context_atoms)}):[/bold bright_blue]\n"
        for atom in packet.context_atoms:
            content_preview = atom.content.replace('\n', ' ').strip()
            if len(content_preview) > 65:
                content_preview = content_preview[:62] + "..."
            panel_content += (f"  - [blue]ID[/blue]: {atom.id} | "
                              f"[blue]Score[/blue]: {atom.final_score:.3f} "
                              f"([dim]Sq:{atom.original_similarity:.2f}, Ss:{atom.pagerank_weight:.2f}[/dim])\n"
                              f"    [dim]{content_preview}[/dim]\n")

    # --- Emerged Concepts ---
    if packet.emerged_concepts:
        panel_content += f"\n[bold bright_magenta]💡 Emerged Concepts ({len(packet.emerged_concepts)}):[/bold bright_magenta]\n"
        for i, concept in enumerate(packet.emerged_concepts):
            panel_content += f"  [{i+1}] \"{concept.concept}\" ([dim]Score: {concept.score:.3f}[/dim])\n"
            
    console.print(Panel(panel_content.strip(), title="[bold yellow]Resonance Packet[/bold yellow]", border_style="yellow", expand=False))


def main():
    """Main function to run the interactive KSF demo."""
    console = Console()
    
    config_path = 'configs/ksf_config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Configuration loaded successfully from {config_path}.")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Configuration file not found at {config_path}.[/bold red]")
        logger.error(f"Configuration file not found: {config_path}")
        return
    except Exception as e:
        console.print(f"[bold red]Error parsing configuration file {config_path}:[/bold red]\n{e}")
        logger.error(f"Failed to parse {config_path}: {e}", exc_info=True)
        return

    try:
        framework = KSFOrchestrator(config)
        logger.info("✓ KSF Framework initialized successfully.")
    except Exception as e:
        console.print(f"[bold red]A critical error occurred during KSF framework initialization:[/bold red]\n{e}")
        logger.error("KSFOrchestrator initialization failed.", exc_info=True)
        return

    console.print(Panel("[bold green]Welcome to the KSF v4.0 Interactive Demo[/bold green]\n"
                        "This system uses a 'Unified Resonance Framework' to synthesize knowledge.\n"
                        "Type your query to begin, or 'exit'/'quit' to leave.",
                        title="Knowledge Synthesized Framework v4.0",
                        expand=False))

    last_concepts: List[EmergedConcept] = []
    while True:
        try:
            prompt_text = "Enter query or command > "
            user_input = Prompt.ask(prompt_text).strip()

            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold yellow]Thank you for using KSF. Goodbye![/bold yellow]")
                break
            
            if user_input.lower() == 'status':
                console.print(framework.get_system_status())
                continue

            query_text = user_input
            if user_input.isdigit() and last_concepts:
                try:
                    index = int(user_input) - 1
                    if 0 <= index < len(last_concepts):
                        concept_item = last_concepts[index]
                        query_text = concept_item.concept
                        console.print(f"🔍 [bold cyan]Deepening exploration on concept[/bold cyan]: \"{query_text}\"")
                    else:
                        console.print(f"❌ [bold red]Invalid index. Please enter a number between 1 and {len(last_concepts)}.[/bold red]")
                        continue
                except (ValueError, IndexError):
                    console.print(f"❌ [bold red]Invalid input. Please enter a valid query or index.[/bold red]")
                    continue
            
            if not query_text:
                continue

            with console.status("[bold green]Processing query...", spinner="dots"):
                result = framework.query(query_text)
            
            packet = ResonancePacket.from_dict(result.get("knowledge_packet", {}))
            display_resonance_packet(packet, console)
            
            console.print(Panel(Markdown(result["answer"]), title="[bold cyan]Final Synthesized Answer[/bold cyan]", border_style="cyan"))

            last_concepts = packet.emerged_concepts

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Operation interrupted by user. Type 'exit' or 'quit' to leave.[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred while processing the query:[/bold red]\n{e}")
            logger.error("An error occurred during query processing.", exc_info=True)


if __name__ == "__main__":
    main() 