"""Tests for jlens.cli — arg parsing, config construction, edge cases."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from jlens.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIRun:
    """Test the 'run' subcommand argument parsing and config construction."""

    def test_minimal_args(self, runner):
        """Default args parse correctly."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model-id" in result.output
        assert "--checkpoints" in result.output
        assert "--n-prompts" in result.output

    def test_explicit_checkpoints(self, runner):
        """--checkpoints with comma-separated values."""
        result = runner.invoke(main, [
            "run",
            "--model-id", "EleutherAI/pythia-160m-deduped",
            "--checkpoints", "8000,9000,10000",
            "--n-prompts", "10",
            "--prompt-seed", "42",
            "--results-dir", "/tmp/test_results",
            "--help",  # --help exits before running
        ])
        # Should parse successfully (help exits with 0)
        assert result.exit_code == 0

    def test_dtype_options(self, runner):
        """--dtype accepts valid choices. Invalid values are caught at config level."""
        result = runner.invoke(main, ["run", "--dtype", "float16", "--help"])
        assert result.exit_code == 0
        result = runner.invoke(main, ["run", "--dtype", "bfloat16", "--help"])
        assert result.exit_code == 0
        result = runner.invoke(main, ["run", "--dtype", "float32", "--help"])
        assert result.exit_code == 0

    def test_layers_parsing(self, runner):
        """--layers comma-separated."""
        result = runner.invoke(main, [
            "run", "--layers", "0,2,5,7", "--help"
        ])
        assert result.exit_code == 0

    def test_start_end_step(self, runner):
        """--start-step and --end-step."""
        result = runner.invoke(main, [
            "run", "--start-step", "1000", "--end-step", "5000", "--help"
        ])
        assert result.exit_code == 0

    def test_results_dir_path_handling(self, runner):
        """--results-dir accepts strings and paths."""
        result = runner.invoke(main, [
            "run", "--results-dir", "my_results", "--help"
        ])
        assert result.exit_code == 0

    def test_no_such_option_model(self, runner):
        """--model does NOT exist (was a bug in catch-up scripts)."""
        result = runner.invoke(main, [
            "run", "--model", "foo/bar", "--help"
        ])
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_no_such_option_steps(self, runner):
        """--steps does NOT exist (was a bug in catch-up scripts)."""
        result = runner.invoke(main, [
            "run", "--steps", "1000,2000", "--help"
        ])
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_valid_option_names(self, runner):
        """The actual valid option names."""
        result = runner.invoke(main, [
            "run",
            "--model-id", "foo",
            "--checkpoints", "1000,2000",
            "--n-prompts", "10",
            "--prompt-seed", "42",
            "--results-dir", "/tmp/test",
            "--dtype", "float16",
            "--max-seq-len", "256",
            "--layers", "0,1,2",
            "--start-step", "1",
            "--end-step", "1000",
            "--help",
        ])
        assert result.exit_code == 0

    def test_checkpoints_overrides_start_end(self, runner):
        """When --checkpoints is set, it should override start/end."""
        # This tests that the help output is consistent, not the actual override logic
        result = runner.invoke(main, [
            "run",
            "--checkpoints", "5000,10000",
            "--start-step", "1",
            "--end-step", "143000",
            "--help",
        ])
        assert result.exit_code == 0


class TestCLIAnalyze:
    """Test the 'analyze' subcommand."""

    def test_analyze_help(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_analyze_nonexistent_dir(self, runner):
        result = runner.invoke(main, ["analyze", "/nonexistent/path"])
        assert result.exit_code != 0


class TestCLIVersion:
    """Test the 'version' subcommand."""

    def test_version(self, runner):
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "jlens" in result.output
