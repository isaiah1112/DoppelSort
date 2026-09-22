from click.testing import CliRunner

from doppelsort.cli import main


def test_cli_help_renders() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "scan" in result.output.lower()
    assert "match" in result.output.lower()


def test_scan_command_help_renders() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scan", "--help"])

    assert result.exit_code == 0
    assert "TARGET_DIR" in result.output
    assert "--db-path" in result.output
