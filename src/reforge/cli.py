"""Command-line interface for anime-reforge."""

from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel

from reforge.synthesis.orchestrator import ReforgeEngine

console = Console()


@click.group()
def main():
    """anime-reforge: High-precision cinematic restyling engine for animation."""
    pass


@main.command()
@click.option(
    "-i", "--input", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the source anime video (.mp4, .mkv, .mov).",
)
@click.option(
    "-o", "--output", "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Target path for the final stylized video (.mp4).",
)
@click.option(
    "-s", "--style", "style_name",
    type=click.Choice(["cinematic_unreal", "makoto_shinkai"], case_sensitive=False),
    default="cinematic_unreal",
    show_default=True,
    help="Target artistic rendering preset.",
)
@click.option(
    "--temporal-blend",
    type=float,
    default=0.25,
    show_default=True,
    help="Temporal feedback strength (0.0 to 0.5) to suppress flickering.",
)
@click.option(
    "--dof / --no-dof",
    default=True,
    show_default=True,
    help="Enable/disable depth-of-field optical bokeh pass.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Deterministic RNG seed.",
)
@click.option(
    "--clean-cache",
    is_flag=True,
    default=False,
    help="Delete intermediate extraction numpy arrays upon completion.",
)
def render(
    input_path: Path,
    output_path: Path,
    style_name: str,
    temporal_blend: float,
    dof: bool,
    seed: int,
    clean_cache: bool,
):
    """Execute an end-to-end cinematic restyle on an input video clip."""
    console.print(
        Panel.fit(
            f"[bold cyan]anime-reforge Engine[/bold cyan]\n"
            f"[dim]Input:[/dim]  {input_path}\n"
            f"[dim]Output:[/dim] {output_path}\n"
            f"[dim]Style:[/dim]  [yellow]{style_name}[/yellow]\n"
            f"[dim]Seed:[/dim]   {seed}",
            border_style="cyan",
        )
    )

    engine = ReforgeEngine(workspace_root=Path("workspace"))
    try:
        final_file = engine.process_video(
            input_video_path=input_path,
            output_video_path=output_path,
            style_name=style_name,
            temporal_blend=temporal_blend,
            apply_dof=dof,
            seed=seed,
            clean_cache=clean_cache,
        )
        console.print(f"\n[bold green]✓ Done![/bold green] Output written to: [bold]{final_file}[/bold]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline Failure:[/bold red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()