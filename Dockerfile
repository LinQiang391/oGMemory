FROM alpine/openclaw:latest

USER root

# Switch to USTC Debian mirror for faster downloads in China
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list 2>/dev/null; \
    true

# Install pip, build deps, and Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-venv python3-dev gcc libpq-dev curl && \
    python3 -m pip install --break-system-packages \
      -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
      --timeout 120 \
      openai psycopg2-binary pyagfs rich requests apscheduler && \
    apt-get purge -y python3-pip python3-venv python3-dev gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy oG-Memory source
COPY --chown=node:node core/ /opt/ogmem/core/
COPY --chown=node:node retrieval/ /opt/ogmem/retrieval/
COPY --chown=node:node extraction/ /opt/ogmem/extraction/
COPY --chown=node:node commit/ /opt/ogmem/commit/
COPY --chown=node:node index/ /opt/ogmem/index/
COPY --chown=node:node providers/ /opt/ogmem/providers/
COPY --chown=node:node service/ /opt/ogmem/service/
COPY --chown=node:node fs/ /opt/ogmem/fs/
COPY --chown=node:node scripts/ /opt/ogmem/scripts/

# Copy plugin into OpenClaw extensions directory
COPY --chown=node:node openclaw_context_engine_plugin/ /app/extensions/og-memory-context-engine/

# Copy AGFS binary and config
COPY --chown=node:node agfs/agfs-server /opt/agfs/agfs-server
COPY --chown=node:node docker/agfs-config.yaml /opt/agfs/config.yaml

# Copy default openclaw.json
COPY --chown=node:node docker/openclaw.json /opt/defaults/openclaw.json

# Copy entrypoint
COPY --chown=node:node docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh /opt/agfs/agfs-server

# Create AGFS data directory (can be overridden by volume mount)
RUN mkdir -p /data/agfs && chown node:node /data/agfs

# Environment defaults
ENV PYTHONPATH=/opt/ogmem
ENV AGFS_BASE_URL=http://127.0.0.1:1833
ENV AGFS_MOUNT_PREFIX=/local/plugin
ENV CONTEXTENGINE_PROVIDER=openai
ENV VECTOR_DB_TYPE=opengauss
ENV INDEX_INTERVAL=15
ENV OG_ACCOUNT_ID=acct-demo
ENV OPENAI_EMBEDDING_MODEL=text-embedding-v2
ENV OPENCLAW_GATEWAY_TOKEN=test-token-123

USER node
EXPOSE 18789

ENTRYPOINT ["/opt/entrypoint.sh"]
