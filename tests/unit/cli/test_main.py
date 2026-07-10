import pytest

from cli.main import build_parser


def test_mcp_help_describes_sse_routes_and_scaling_constraint(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["mcp", "--help"])

    help_text = capsys.readouterr().out
    assert "OGMEM_MCP_PATH" in help_text
    assert "/messages/" in help_text
    assert "process-local" in help_text
