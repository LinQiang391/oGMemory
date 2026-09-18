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


def test_onboard_defaults_to_local_opengauss():
    args = build_parser().parse_args(["onboard", "--non-interactive"])

    assert args.mode is None
    assert args.vector_db is None
    assert args.sql_dialect == "opengauss"


def test_onboard_parser_exposes_docker_quickstart_options():
    args = build_parser().parse_args(
        ["onboard", "--mode", "docker", "--instance-name", "locomo",
         "--opengauss-port", "26432", "--port", "18090",
         "--mcp-port", "18091", "--openclaw-ce-port", "18790",
         "--openclaw-mcp-port", "18791"]
    )

    assert args.instance_name == "locomo"
    assert args.opengauss_port == 26432
    assert args.port == 18090
    assert args.mcp_port == 18091
