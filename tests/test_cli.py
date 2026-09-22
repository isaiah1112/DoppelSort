from click.testing import CliRunner

from doppelsort.cli import main


def test_cli_help_renders() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Compare a source portrait to a directory of portraits" in result.output
    assert "--top-n" in result.output
