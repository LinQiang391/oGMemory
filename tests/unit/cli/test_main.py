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
    assert "OG_ROLE_CONTROL_ENABLED" in help_text
    assert "OG_ROOT_API_KEY" in help_text
    assert "X-API-Key" in help_text


def test_onboard_accepts_opengauss_sql_dialect():
    args = build_parser().parse_args(
        ["onboard", "--non-interactive", "--sql-dialect", "opengauss"]
    )

    assert args.sql_dialect == "opengauss"
