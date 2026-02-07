"""CLI entry point for the wpv pipeline."""

import typer

app = typer.Typer(name="wpv", help="Water Polo Video auto-reframe pipeline.")


@app.command()
def ingest(match_id: str = typer.Argument(..., help="Match identifier")):
    """Validate incoming footage and manifest."""
    typer.echo(f"[ingest] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def stitch(match_id: str = typer.Argument(..., help="Match identifier")):
    """Stitch/decode Insta360 files to equirectangular master."""
    typer.echo(f"[stitch] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def track(match_id: str = typer.Argument(..., help="Match identifier")):
    """Run ball detection and tracking on the equirectangular master."""
    typer.echo(f"[track] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def camera(match_id: str = typer.Argument(..., help="Match identifier")):
    """Generate virtual camera path from tracking data."""
    typer.echo(f"[camera] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def render(match_id: str = typer.Argument(..., help="Match identifier")):
    """Render reframed video from equirectangular master + camera path."""
    typer.echo(f"[render] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def highlights(match_id: str = typer.Argument(..., help="Match identifier")):
    """Extract highlight montage."""
    typer.echo(f"[highlights] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def upscale(match_id: str = typer.Argument(..., help="Match identifier")):
    """Apply AI upscaling to rendered output."""
    typer.echo(f"[upscale] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def publish(match_id: str = typer.Argument(..., help="Match identifier")):
    """Upload to YouTube."""
    typer.echo(f"[publish] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def run_all(match_id: str = typer.Argument(..., help="Match identifier")):
    """Run the full pipeline end-to-end."""
    typer.echo(f"[run-all] {match_id} — not yet implemented")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
