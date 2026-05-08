"""ogmem onboard command - unified setup wizard."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from cli.lib.colors import CYAN, GREEN, NC, YELLOW, fail, info, ok, warn
from cli.lib.prompt import confirm, input_masked, input_validated, select

# Provider defaults
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "embedding_model": "text-embedding-ada-002", "embedding_dim": 1536},
    "volcengine": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seed-2-0-code-preview-260215", "embedding_model": "doubao-embedding-vision-250615", "embedding_dim": 1024},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "embedding_model": "text-embedding-v3", "embedding_dim": 1024},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash", "embedding_model": "embedding-3", "embedding_dim": 1024},
    "mock": {"base_url": "", "model": "mock", "embedding_model": "mock", "embedding_dim": 1024},
}


def _print_header(step: int, title: str) -> None:
    print(f"\n{CYAN}{'='*60}{NC}\n{CYAN}Step {step}: {title}{NC}\n{CYAN}{'='*60}{NC}\n")


# ---------------------------------------------------------------------------
# Shared steps (all modes)
# ---------------------------------------------------------------------------

def _step1_check(args) -> bool:
    _print_header(1, "System Check")
    from cli.commands.check import run as check_run
    check_run(args)
    return confirm("Continue?", default=True)


def _step2_llm(args) -> dict:
    _print_header(2, "LLM Configuration")
    if args.non_interactive:
        if not args.provider:
            print(fail("--provider required in non-interactive mode"))
            sys.exit(1)
        provider, api_key = args.provider, args.api_key or ""
        base_url = args.base_url or PROVIDER_DEFAULTS.get(provider, {}).get("base_url", "")
        model = args.model or PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
    else:
        providers = list(PROVIDER_DEFAULTS.keys())
        provider_idx = select("LLM provider:", providers, default=0)
        provider = providers[provider_idx]
        defaults = PROVIDER_DEFAULTS[provider]
        api_key = input_masked("API key (press Enter to skip)")
        base_url = input_validated("Base URL", defaults.get("base_url", ""))
        model = input_validated("Model", defaults.get("model", ""))
    return {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}


def _step3_embedding(args, llm_config: dict) -> dict:
    _print_header(3, "Embedding Configuration")
    if args.non_interactive:
        provider = args.embedding_provider or llm_config["provider"]
        model = args.embedding_model or PROVIDER_DEFAULTS.get(provider, {}).get("embedding_model", "")
        dim = PROVIDER_DEFAULTS.get(provider, {}).get("embedding_dim", 1024)
    else:
        same = confirm("Use same provider for embedding?", default=True)
        provider = llm_config["provider"] if same else input_validated("Embedding provider", llm_config["provider"])
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        model = input_validated("Embedding model", defaults.get("embedding_model", ""))
        dim = int(input_validated("Dimension", str(defaults.get("embedding_dim", 1024))) or 1024)
    return {"provider": provider, "model": model, "dimension": dim}


def _step4_database(args) -> tuple[dict, dict]:
    """Vector DB + PostgreSQL configuration and schema init."""
    _print_header(4, "Database Configuration")

    # Vector DB type
    if getattr(args, "non_interactive", False):
        vdb_type = getattr(args, "vector_db", None) or "chroma"
    else:
        vdb_type = ["chroma", "opengauss"][select("Vector database:", ["chroma", "opengauss"], default=0)]

    vdb = {"type": vdb_type}
    if vdb_type == "opengauss":
        vdb["connection_string"] = input_validated("Connection string", "host=localhost port=5432 dbname=ogmem")
    elif vdb_type == "chroma":
        vdb["connection_string"] = ""

    # PostgreSQL for storage backend
    import getpass
    current_user = getpass.getuser()
    default_conn = f"host=127.0.0.1 port=5432 dbname=ogmemory user={current_user}"
    conn_str = getattr(args, "db_connection", None)
    if not conn_str and not getattr(args, "non_interactive", False):
        conn_str = input_validated("PostgreSQL connection string", default_conn)
    elif not conn_str:
        conn_str = default_conn

    db = {"connection_string": conn_str}

    # Check and init schema
    from cli.lib.db_setup import check_postgresql, init_schema
    if check_postgresql(conn_str):
        print(ok("PostgreSQL reachable"))
        if not getattr(args, "non_interactive", False) and confirm("Initialize schema?", default=True):
            if init_schema(conn_str):
                print(ok("Schema initialized"))
            else:
                print(warn("Schema init failed — you can run it later with ogmem check"))
    else:
        print(warn("PostgreSQL not reachable — ensure it's running"))
        print(info("You can re-run ogmem onboard or ogmem check after starting PostgreSQL"))

    return vdb, db


def _step5_target(args) -> str:
    """Select deployment target."""
    _print_header(5, "Deployment Target")
    if getattr(args, "non_interactive", False) and getattr(args, "mode", None):
        return args.mode
    targets = [
        "Docker (containerized)",
        "Agent Plugin (OpenClaw / Claude Code)",
        "Headless (CE server only)",
    ]
    idx = select("Choose deployment target:", targets, default=1)
    return ["docker", "plugin", "headless"][idx]


# ---------------------------------------------------------------------------
# Shared sub-steps (plugin & headless)
# ---------------------------------------------------------------------------

def _step_install_deps(args, embedding: dict, vdb: dict) -> None:
    """Install Python dependencies."""
    _print_header(6, "Install Dependencies")

    extras = []
    if embedding.get("provider") == "st":
        extras.append("sentence-transformers")
    if vdb.get("type") == "chroma":
        extras.append("chroma")

    if shutil.which("uv"):
        cmd = ["uv", "sync"]
        for extra in extras:
            cmd.extend(["--extra", extra])
        print(info(f"Running {' '.join(cmd)}..."))
        subprocess.run(cmd, check=False)
    elif confirm("Run pip install -e . ?", default=True):
        pip_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        if extras:
            pip_cmd.append("[" + ",".join(extras) + "]")
        subprocess.run(pip_cmd, check=False)
    else:
        print(warn("Skipping dependency installation"))
        if extras:
            print(info(f"Note: you'll need to manually install extras: {', '.join(extras)}"))


# ---------------------------------------------------------------------------
# Mode-specific steps
# ---------------------------------------------------------------------------

def _step_docker_setup(args, llm: dict, embedding: dict, vdb: dict) -> dict:
    """Docker mode: generate deploy.env + ogmemory.yaml, optionally start."""
    from cli.lib.deploy_helpers import generate_deploy_env, generate_docker_config
    from cli.lib.process import find_project_root

    root = find_project_root()
    result = {}

    # Generate deploy.env
    _print_header(6, "Generate deploy.env")
    db_config = {
        "enable_opengauss": vdb.get("type") == "opengauss",
        "port": "15432",
        "host_ip": "127.0.0.1",
    }
    env_path = generate_deploy_env(llm, db_config=db_config, output_path=root / "deploy" / "deploy.env")
    print(ok(f"Written: {env_path}"))

    # Generate ogmemory.yaml
    _print_header(7, "Generate ogmemory.yaml")
    config_path = generate_docker_config(llm, embedding, vdb, output_path=root / "deploy" / "ogmemory.yaml")
    print(ok(f"Written: {config_path}"))

    result["env_path"] = str(env_path)
    result["config_path"] = str(config_path)

    # Start containers
    _print_header(8, "Start Docker Containers")
    if confirm("Start containers now?", default=False):
        password = input_masked("OpenGauss password (8+ chars, 3+ types)") if db_config["enable_opengauss"] else None
        from cli.lib.deploy_helpers import deploy_docker
        rc = deploy_docker(env_file=env_path, password=password)
        if rc != 0:
            print(warn(f"deploy.sh exited with code {rc}"))
            result["deploy_failed"] = True
        else:
            print(ok("Containers started"))
    else:
        print(info("Skipped. Run: bash deploy/deploy.sh"))

    return result


def _step_plugin_install(args, llm: dict) -> dict:
    """Plugin mode: select agent, generate config, install into agent."""
    import json

    result = {}

    _print_header(7, "Select Agent")
    agent_types = ["OpenClaw (context-engine plugin)", "Claude Code (hooks)"]
    idx = select("Choose agent:", agent_types, default=0)
    agent = ["openclaw", "claude_hooks"][idx]
    result["plugin_type"] = agent

    if agent == "openclaw":
        _step_install_openclaw(args, llm, result)
    elif agent == "claude_hooks":
        _step_install_claude_hooks(args, llm, result)

    return result


def _step_install_openclaw(args, llm: dict, result: dict) -> None:
    """Generate OpenClaw gateway config and install plugin."""
    import json
    from cli.lib.config_loader import get_openclaw_template, load_template

    _print_header(8, "OpenClaw Plugin Installation")

    ogmem_url = "http://127.0.0.1:8090"
    if not getattr(args, "non_interactive", False):
        ogmem_url = input_validated("oG-Memory API URL", ogmem_url)
    result["ogmem_url"] = ogmem_url

    # Load and fill the gateway template
    template_str = load_template(get_openclaw_template())
    template = json.loads(template_str)

    _fill_json(template, "__LLM_PROVIDER__", llm["provider"])
    _fill_json(template, "__LLM_MODEL__", llm["model"])
    _fill_json(template, "__LLM_BASE_URL__", llm.get("base_url", ""))
    _fill_json(template, "__LLM_API_KEY__", llm.get("api_key", ""))
    _fill_json(template, "__OGMEM_URL__", ogmem_url)
    _fill_json(template, "__GATEWAY_PORT__", "18789")
    _fill_json(template, "__OG_AUTH_API_KEY__", "")
    _fill_json(template, "__OG_AUTH_ACCOUNT_ID__", "")

    out_path = Path.cwd() / "openclaw.plugin.json"
    out_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n")
    print(ok(f"OpenClaw gateway config written: {out_path}"))

    # Copy plugin into OpenClaw plugins dir if openclaw is installed
    import shutil
    openclaw_bin = shutil.which("openclaw")
    if openclaw_bin:
        print(ok(f"OpenClaw binary found: {openclaw_bin}"))
        print(info("Plugin will be loaded when gateway starts"))
    else:
        print(info("OpenClaw binary not in PATH — plugin config is ready for manual setup"))

    result["openclaw_config"] = str(out_path)


def _step_install_claude_hooks(args, llm: dict, result: dict) -> None:
    """Install Claude Code hooks into project .claude/settings.json."""
    import json
    from cli.lib.process import find_project_root

    _print_header(8, "Claude Code Hooks Installation")

    root = find_project_root()
    claude_plugin = root / "claude-plugin"
    hooks_json = claude_plugin / "hooks" / "hooks.json"

    if not hooks_json.exists():
        print(fail(f"hooks.json not found: {hooks_json}"))
        return

    # Read hooks template and resolve ${CLAUDE_PLUGIN_ROOT}
    hooks_cfg = json.loads(hooks_json.read_text())
    for event_name, matchers in hooks_cfg.get("hooks", {}).items():
        for matcher in matchers:
            for hook in matcher.get("hooks", []):
                if "command" in hook:
                    hook["command"] = hook["command"].replace(
                        "${CLAUDE_PLUGIN_ROOT}", str(claude_plugin.resolve())
                    )

    # Write into project .claude/settings.json
    settings_dir = root / ".claude"
    settings_file = settings_dir / "settings.json"
    settings = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}

    # Merge hooks — don't overwrite existing, append per-event
    existing_hooks = settings.get("hooks", {})
    for event_name, matchers in hooks_cfg["hooks"].items():
        if event_name not in existing_hooks:
            existing_hooks[event_name] = matchers
        else:
            existing_commands = set()
            for m in existing_hooks[event_name]:
                for h in m.get("hooks", []):
                    existing_commands.add(h.get("command", ""))
            for matcher in matchers:
                matcher_cmds = {h.get("command", "") for h in matcher.get("hooks", [])}
                if not matcher_cmds.intersection(existing_commands):
                    existing_hooks[event_name].append(matcher)

    settings["hooks"] = existing_hooks
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")

    hook_count = sum(len(m) for ms in hooks_cfg["hooks"].values() for m in ms)
    print(ok(f"Claude Code hooks installed to {settings_file}"))
    print(f"  {hook_count} hooks registered ({', '.join(hooks_cfg['hooks'].keys())})")

    result["hooks_file"] = str(settings_file)


def _fill_json(obj, placeholder: str, value: str) -> None:
    """Recursively replace placeholder strings in a JSON-like dict/list."""
    if isinstance(obj, dict):
        for k in obj:
            if isinstance(obj[k], str) and placeholder in obj[k]:
                obj[k] = obj[k].replace(placeholder, value)
            else:
                _fill_json(obj[k], placeholder, value)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and placeholder in v:
                obj[i] = v.replace(placeholder, value)
            else:
                _fill_json(v, placeholder, value)


# ---------------------------------------------------------------------------
# Write configuration
# ---------------------------------------------------------------------------

def _write_config(llm: dict, embedding: dict, vdb: dict, db: dict, *, port: int = 8090, plugin_type: str = "", openclaw_config: str = "") -> Path:
    """Write config/ogmem.yaml."""
    conn_str = db.get("connection_string", "")

    cfg = f"""# oG-Memory Configuration
# Generated by ogmem onboard

paths:
  data_root: .ogmem_data

llm:
  provider: {llm['provider']}
  api_key: "{llm['api_key']}"
  base_url: "{llm['base_url']}"
  model: "{llm['model']}"
  temperature: 0.1

embedding:
  provider: {embedding['provider']}
  model: "{embedding['model']}"

storage:
  backend: sql
  connection_string: "{conn_str}"
  pool_size: 5

vector_db:
  type: {vdb['type']}
  dimension: {embedding['dimension']}
"""
    if vdb["type"] == "chroma":
        cfg += "  chroma_persist_dir: .ogmem_data/chroma\n"
    elif vdb.get("connection_string"):
        cfg += f'  connection_string: "{vdb["connection_string"]}"\n'

    cfg += f"""
service:
  http_port: {port}
  workers: 2

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"
"""

    if plugin_type:
        cfg += f"""
plugin:
  type: "{plugin_type}"
"""
        if openclaw_config:
            cfg += f'  openclaw_config: "{openclaw_config}"\n'

    config_path = Path.cwd() / "config" / "ogmem.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cfg)
    return config_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(args) -> int:
    print(f"\n{GREEN}oG-Memory Setup Wizard{NC}\n{YELLOW}This wizard will guide you through the configuration.{NC}\n")

    # Step 1: System Check
    if not _step1_check(args):
        return 1

    # Step 2: LLM
    llm_config = _step2_llm(args)

    # Step 3: Embedding
    embedding_config = _step3_embedding(args, llm_config)

    # Step 4: Database (Vector DB + PostgreSQL)
    vdb_config, db_config = _step4_database(args)

    # Step 5: Deployment Target
    deployment = _step5_target(args)

    if deployment == "docker":
        docker_setup = _step_docker_setup(args, llm_config, embedding_config, vdb_config)
        print(ok("\nDocker configuration complete!"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  LLM:         {llm_config['provider']} / {llm_config['model']}")
        print(f"  Files:       {docker_setup.get('env_path', 'deploy/deploy.env')}")
        print(f"               {docker_setup.get('config_path', 'deploy/ogmemory.yaml')}")
        print(f"{'-'*60}")
        if not docker_setup.get("deploy_failed"):
            print(info("Manage with: bash deploy/deploy.sh"))

    elif deployment == "plugin":
        # Step 6: Install deps (shared with headless)
        _step_install_deps(args, embedding_config, vdb_config)

        # Step 7-8: Select agent & install plugin
        plugin_setup = _step_plugin_install(args, llm_config)
        plugin_type = plugin_setup.get("plugin_type", "")
        openclaw_config = plugin_setup.get("openclaw_config", "")

        # Write config
        config_path = _write_config(
            llm_config, embedding_config, vdb_config, db_config,
            plugin_type=plugin_type, openclaw_config=openclaw_config,
        )
        print(ok(f"\nConfiguration written: {config_path}"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  LLM:         {llm_config['provider']} / {llm_config['model']}")
        print(f"  Plugin:      {plugin_type}")
        print(f"{'-'*60}")

        if confirm("Start services now?", default=True):
            from cli.commands.start import start_plugin
            return start_plugin(type('Args', (), {
                'dry_run': False, 'daemon': True, 'health_timeout': 30,
            })())
        print(ok("\nSetup complete! Run 'ogmem start plugin' to begin."))

    elif deployment == "headless":
        # Step 6: Install deps
        _step_install_deps(args, embedding_config, vdb_config)

        # Write config
        config_path = _write_config(llm_config, embedding_config, vdb_config, db_config)
        print(ok(f"\nConfiguration written: {config_path}"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  LLM:         {llm_config['provider']} / {llm_config['model']}")
        print(f"  Embedding:   {embedding_config['provider']} / {embedding_config['model']}")
        print(f"  Vector DB:   {vdb_config['type']}")
        print(f"  Storage:     sql")
        print(f"{'-'*60}")

        if confirm("Start services now?", default=True):
            from cli.commands.start import start_headless
            return start_headless(type('Args', (), {
                'mode': 'headless', 'daemon': True, 'dry_run': False,
            })())
        print(ok("\nSetup complete! Run 'ogmem start headless' to begin."))

    return 0
