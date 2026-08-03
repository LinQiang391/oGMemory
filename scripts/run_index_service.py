#!/usr/bin/env python3
"""Standalone IndexService runner.

Starts the OutboxScheduler as a long-running process that periodically
scans AGFS outbox events, generates embeddings, and upserts into openGauss.

Usage:
    python scripts/run_index_service.py                    # uses config.yaml or env vars
    OGMEM_CONFIG=/path/to/config.yaml python scripts/run_index_service.py
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
    from pyagfs import AGFSClient

    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.config import ProviderConfig
    from providers.unified_config import get_config
    from service.index_service import init_index_service

    cfg = get_config()
    if cfg.retrieval_hybrid_enabled:
        raise RuntimeError(
            "hybrid retrieval is unsupported by the AGFS standalone index service; "
            "run the SQL-backed MemoryService so every outbox event dual-writes"
        )
    provider_cfg = ProviderConfig.from_ogmem_config(cfg)

    embedder = provider_cfg.create_embedder()
    vector_index = provider_cfg.create_vector_index()

    agfs_client = AGFSClient(api_base_url=cfg.agfs_base_url)
    agfs = AGFSContextFS(client=agfs_client, mount_prefix=cfg.agfs_mount_prefix)
    outbox_store = OutboxStore(client=agfs_client, fs=agfs, mount_prefix=cfg.agfs_mount_prefix)

    logger.info("Config: provider=%s, embedding=%s, db=%s, agfs=%s",
                cfg.provider, cfg.openai_embedding_model,
                cfg.vector_db_type, cfg.agfs_base_url)
    logger.info("Params: account=%s, interval=%ds, workers=%d",
                cfg.account_id, cfg.index_interval, cfg.index_workers)

    from providers.llm.openai_llm import OpenAILLM
    llm = OpenAILLM(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        model=cfg.openai_llm_model,
        json_mode=getattr(cfg, 'llm_json_mode', False),
    )

    service = init_index_service(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=lambda: [cfg.account_id],
        interval_seconds=cfg.index_interval,
        worker_count=cfg.index_workers,
        fs=agfs,
        llm=llm,
        directory_summary_enabled=getattr(cfg, 'directory_summary_enabled', False),
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
