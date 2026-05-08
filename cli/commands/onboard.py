"""ogmem onboard command - interactive setup wizard with mode-aware flows."""

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


def _step2_deployment(args) -> str:
    _print_header(2, "Deployment Mode")
    if args.non_interactive and args.mode:
        return args.mode
    modes = ["local (development)", "docker (production)", "plugin (OpenClaw/Claude integration)"]
    idx = select("Choose deployment mode:", modes, default=0)
    return ["local", "docker", "plugin"][idx]


def _step3_llm(args) -> dict:
    _print_header(3, "LLM Configuration")
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


def _step4_embedding(args, llm_config: dict) -> dict:
    _print_header(4, "Embedding Configuration")
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


def _step5_vector_db(args) -> dict:
    _print_header(5, "Vector Database")
    if args.non_interactive:
        return {"type": args.vector_db or "chroma", "connection_string": ""}
    vdb_type = ["chroma", "opengauss"][select("Vector database:", ["chroma", "opengauss"], default=0)]
    return {
        "type": vdb_type,
        "connection_string": input_validated("Connection string", "host=localhost port=5432 dbname=ogmem") if vdb_type == "opengauss" else "",
    }


# ---------------------------------------------------------------------------
# Mode-specific steps
# ---------------------------------------------------------------------------

def _step_local_setup(args, llm: dict, vdb: dict, embedding: dict | None = None) -> dict:
    """Local mode: install deps, init DB, check AGFS."""
    storage_backend = getattr(args, "storage_backend", "sql") or "sql"
    result = {"storage_backend": storage_backend}

    # Step 6: Install Python dependencies
    _print_header(6, "Install Dependencies")

    # Determine which extras to install based on config choices
    extras = []
    if embedding and embedding.get("provider") == "st":
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

    # Step 7: Database initialization (if using SQL backend)
    if storage_backend == "sql":
        _print_header(7, "Database Initialization")
        default_conn = "host=127.0.0.1 port=5432 dbname=ogmemory user=postgres password=postgres"
        conn_str = getattr(args, "db_connection", None) or input_validated("PostgreSQL connection string", default_conn)
        result["connection_string"] = conn_str

        from cli.lib.db_setup import check_postgresql, init_schema
        if check_postgresql(conn_str):
            print(ok("PostgreSQL reachable"))
            if confirm("Initialize schema?", default=True):
                if init_schema(conn_str):
                    print(ok("Schema initialized"))
                else:
                    print(warn("Schema init failed — you can run it later with ogmem check"))
        else:
            print(warn("Cannot reach PostgreSQL — ensure it's running and connection string is correct"))
            print(info("You can re-run ogmem onboard or ogmem check after starting PostgreSQL"))

    # Step 8: AGFS binary check
    _print_header(8, "AGFS Binary Check")
    from cli.lib.process import find_project_root
    agfs_bin = find_project_root() / "agfs" / "agfs-server"
    if agfs_bin.exists():
        print(ok(f"AGFS binary found: {agfs_bin}"))
    else:
        if storage_backend == "sql":
            print(info("AGFS binary not found (not needed for SQL backend)"))
        else:
            print(warn("AGFS binary not found — SQL backend is recommended if AGFS is unavailable"))

    return result


def _step_docker_setup(args, llm: dict, embedding: dict, vdb: dict) -> dict:
    """Docker mode: generate deploy.env + ogmemory.yaml, optionally start."""
    from cli.lib.deploy_helpers import generate_deploy_env, generate_docker_config
    from cli.lib.process import find_project_root

    root = find_project_root()
    result = {}

    # Step 6: Generate deploy.env
    _print_header(6, "Generate deploy.env")
    db_config = {
        "enable_opengauss": vdb.get("type") == "opengauss",
        "port": "15432",
        "host_ip": "127.0.0.1",
    }
    env_path = generate_deploy_env(llm, db_config=db_config, output_path=root / "deploy" / "deploy.env")
    print(ok(f"Written: {env_path}"))

    # Step 7: Generate ogmemory.yaml
    _print_header(7, "Generate ogmemory.yaml")
    config_path = generate_docker_config(llm, embedding, vdb, output_path=root / "deploy" / "ogmemory.yaml")
    print(ok(f"Written: {config_path}"))

    result["env_path"] = str(env_path)
    result["config_path"] = str(config_path)

    # Step 8: Start containers
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


def _step_plugin_setup(args, llm: dict) -> dict:
    """Plugin mode: configure OpenClaw or Claude Code hooks."""
    result = {}

    _print_header(6, "Plugin Type")
    plugin_types = ["OpenClaw v2 (context-engine plugin)", "OpenClaw v1 (tool plugin)", "Claude Code hooks"]
    idx = select("Choose plugin type:", plugin_types, default=0)
    plugin_type = ["openclaw_v2", "openclaw_v1", "claude_hooks"][idx]
    result["plugin_type"] = plugin_type

    if plugin_type.startswith("openclaw"):
        _print_header(7, "OpenClaw Configuration")
        ogmem_url = input_validated("oG-Memory API URL", "http://127.0.0.1:8090")
        result["ogmem_url"] = ogmem_url

        # Load and fill the template
        from cli.lib.config_loader import get_openclaw_template, load_template
        import json
        template_str = load_template(get_openclaw_template())
        template = json.loads(template_str)

        # Fill in values
        _fill_json(template, "__LLM_PROVIDER__", llm["provider"])
        _fill_json(template, "__LLM_MODEL__", llm["model"])
        _fill_json(template, "__LLM_BASE_URL__", llm.get("base_url", ""))
        _fill_json(template, "__LLM_API_KEY__", llm.get("api_key", ""))
        _fill_json(template, "__OGMEM_URL__", ogmem_url)
        _fill_json(template, "__GATEWAY_PORT__", "18789")

        out_path = Path.cwd() / "openclaw.plugin.json"
        out_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n")
        print(ok(f"Written: {out_path}"))

    elif plugin_type == "claude_hooks":
        _print_header(7, "Claude Code Hooks")
        ogmem_url = input_validated("oG-Memory API URL", "http://127.0.0.1:8090")
        result["ogmem_url"] = ogmem_url

        hooks_script = Path(__file__).resolve().parent.parent.parent / "claude-plugin" / "scripts" / "install_hooks.sh"
        if hooks_script.exists():
            if confirm(f"Run {hooks_script}?", default=True):
                subprocess.run(["bash", str(hooks_script)], check=False)
        else:
            print(info("Claude Code hooks installation script not found"))
            print(info("See claude-plugin/README.md for manual setup"))

    return result


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

def _write_local_config(llm: dict, embedding: dict, vdb: dict, local_setup: dict, svc: dict | None = None) -> Path:
    """Write ogmem.yaml for local mode."""
    storage = local_setup.get("storage_backend", "sql")
    conn_str = local_setup.get("connection_string", "")
    svc = svc or {}

    cfg = f"""# oG-Memory Configuration
# Generated by ogmem onboard (local mode)

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
"""
    if storage == "sql" and conn_str:
        cfg += f"""
storage:
  backend: sql
  connection_string: "{conn_str}"
  pool_size: 5
"""
    cfg += f"""
vector_db:
  type: {vdb['type']}
"""
    if vdb["type"] == "chroma":
        cfg += "  chroma_persist_dir: .ogmem_data/chroma\n"
    elif vdb.get("connection_string"):
        cfg += f'  connection_string: "{vdb["connection_string"]}"\n'

    cfg += f"""
service:
  http_port: {svc.get('port', 8090)}
  workers: 2

identity:
  account_id: "{svc.get('account_id', 'acct-demo')}"
  user_id: "{svc.get('user_id', 'u-alice')}"
  agent_id: "{svc.get('agent_id', 'main')}"
"""
    config_path = Path.cwd() / "config" / "ogmem.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cfg)
    return config_path


def _write_plugin_config(llm: dict, plugin_setup: dict) -> Path:
    """Write ogmem.yaml for plugin mode (points to remote API)."""
    ogmem_url = plugin_setup.get("ogmem_url", "http://127.0.0.1:8090")

    cfg = f"""# oG-Memory Configuration
# Generated by ogmem onboard (plugin mode — connects to remote API)

service:
  base_url: "{ogmem_url}"

llm:
  provider: {llm['provider']}
  api_key: "{llm['api_key']}"
  base_url: "{llm['base_url']}"
  model: "{llm['model']}"
"""
    config_path = Path.cwd() / "config" / "ogmem.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cfg)
    return config_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(args) -> int:
    print(f"\n{GREEN}oG-Memory Setup Wizard{NC}\n{YELLOW}This wizard will guide you through the configuration.{NC}\n")
    if not _step1_check(args):
        return 1

    deployment = _step2_deployment(args)
    llm_config = _step3_llm(args)
    embedding_config = _step4_embedding(args, llm_config)
    vdb_config = _step5_vector_db(args)

    if deployment == "local":
        local_setup = _step_local_setup(args, llm_config, vdb_config, embedding_config)
        config_path = _write_local_config(llm_config, embedding_config, vdb_config, local_setup)
        print(ok(f"\nConfiguration written: {config_path}"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  LLM:         {llm_config['provider']} / {llm_config['model']}")
        print(f"  Embedding:   {embedding_config['provider']} / {embedding_config['model']}")
        print(f"  Vector DB:   {vdb_config['type']}")
        print(f"  Storage:     {local_setup.get('storage_backend', 'sql')}")
        print(f"{'-'*60}")
        if confirm("Start services now?", default=False):
            from cli.commands.start import start_local
            return start_local(type('Args', (), {'mode': 'local', 'daemon': True, 'dry_run': False})())
        print(ok("\nSetup complete! Run 'ogmem start local' to begin."))

    elif deployment == "docker":
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
        plugin_setup = _step_plugin_setup(args, llm_config)
        config_path = _write_plugin_config(llm_config, plugin_setup)
        print(ok(f"\nConfiguration written: {config_path}"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  Plugin:      {plugin_setup.get('plugin_type', 'unknown')}")
        print(f"{'-'*60}")
        print(ok("\nSetup complete!"))

    return 0
