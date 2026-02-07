"""Smoke tests to verify the package is importable and CLI is wired up."""

import subprocess
import sys


def test_import():
    import wpv
    assert wpv.__version__ == "0.1.0"


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "typer", "wpv.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "wpv" in result.stdout.lower() or "usage" in result.stdout.lower()
