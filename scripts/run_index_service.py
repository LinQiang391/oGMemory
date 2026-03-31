#!/usr/bin/env python3
"""Standalone IndexService runner.

Starts the OutboxScheduler as a long-running process that periodically
scans AGFS outbox events, generates embeddings, and upserts into openGauss.

Usage:
    # With environment variables set (same as OpenClaw container):
    python scripts/run_index_service.py

    # Or with nohup for background:
    nohup python scripts/run_index_service.py &

Environment variables (required):
    CONTEXTENGINE_PROVIDER  - "openai"
    OPENAI_BASE_URL         - e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_API_KEY          - API key
    OPENAI_EMBEDDING_MODEL  - e.g. "text-embedding-v2"
    VECTOR_DB_TYPE          - "opengauss"
    OPENGAUSS_CONNECTION_STRING - e.g. "host=127.0.0.1 port=8799 ..."
    AGFS_BASE_URL           - e.g. "http://127.0.0.1:1833"

Optional:
    OG_ACCOUNT_ID           - default "acct-demo"
    INDEX_INTERVAL          - polling interval in seconds (default 30)
    INDEX_WORKERS           - number of worker threads (default 1)
"""

import logging
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("index-service")


def main():
    from providers.config import ProviderConfig
    from pyagfs import AGFSClient
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from commit.outbox_store import OutboxStore
    from service.index_service import init_index_service

    agfs_base_url = os.environ.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
    mount_prefix = os.environ.get("AGFS_MOUNT_PREFIX", "/local/plugin")
    account_id = os.environ.get("OG_ACCOUNT_ID", "acct-demo")
    interval = int(os.environ.get("INDEX_INTERVAL", "30"))
    workers = int(os.environ.get("INDEX_WORKERS", "1"))

    config = ProviderConfig.from_env()
    embedder = config.create_embedder()
    vector_index = config.create_vector_index()

    agfs_client = AGFSClient(api_base_url=agfs_base_url)
    agfs = AGFSContextFS(client=agfs_client, mount_prefix=mount_prefix)
    outbox_store = OutboxStore(client=agfs_client, fs=agfs, mount_prefix=mount_prefix)

    logger.info("Config: provider=%s, embedding=%s, db=%s, agfs=%s",
                config.provider, config.openai_embedding_model,
                config.vector_db_type, agfs_base_url)
    logger.info("Params: account=%s, interval=%ds, workers=%d", account_id, interval, workers)

    from providers.llm.openai_llm import OpenAILLM
    llm = OpenAILLM(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
    )

    service = init_index_service(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=lambda: [account_id],
        interval_seconds=interval,
        worker_count=workers,
        fs=agfs,
        llm=llm,
    )

    def shutdown(signum, frame):
        logger.info("Received signal %d, stopping...", signum)
        service.stop(wait=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    service.start()
    logger.info("IndexService running (Ctrl+C to stop)")

    while True:
        time.sleep(60)
        stats = service.get_aggregated_stats()
        logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
