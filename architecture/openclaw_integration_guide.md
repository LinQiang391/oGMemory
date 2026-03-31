# oG-Memory × OpenClaw 部署启动指南

## 1. 安装依赖

```bash
# Python 3.11+
pip install openai psycopg2-binary pyagfs rich requests apscheduler
```

## 2. 启动 AGFS Server

```bash
cd /data1/sundechao/ogmem/oG-Memory/agfs
nohup ./agfs-server -c config.yaml > agfs.log 2>&1 &

# 验证
curl http://127.0.0.1:1833/serverinfo/version
```

## 3. 创建 OpenClaw 配置

```bash
mkdir -p /data1/sundechao/ogmem/openclaw-home
```

编辑 `openclaw-home/openclaw.json`：

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "dashscope/glm-5"
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "dashscope": {
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "<你的DashScope API Key>",
        "api": "openai-completions",
        "models": [
          {
            "id": "glm-5",
            "name": "glm-5",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 131072,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "gateway": {
    "http": {
      "endpoints": {
        "responses": { "enabled": true }
      }
    }
  },
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "pythonBin": "/data2/miniconda/bin/python"
        }
      },
      "memory-core": { "enabled": false }
    }
  }
}
```

## 4. 启动 OpenClaw 容器

```bash
docker run -d \
  --name openclaw-ogmem \
  --network host \
  --restart unless-stopped \
  -v /data2/miniconda:/data2/miniconda:ro \
  -v /home/sundechao/.local:/home/sundechao/.local:ro \
  -v /data1/sundechao/ogmem/openclaw-home:/home/node/.openclaw \
  -v /data1/sundechao/ogmem/oG-Memory/openclaw_context_engine_plugin:/app/extensions/og-memory-context-engine \
  -v /data1/sundechao/ogmem/oG-Memory:/data1/sundechao/ogmem/oG-Memory:ro \
  -e OG_PYTHON_BIN=/data2/miniconda/bin/python \
  -e OPENCLAW_GATEWAY_TOKEN=test-token-123 \
  -e AGFS_BASE_URL=http://127.0.0.1:1833 \
  -e CONTEXTENGINE_PROVIDER=openai \
  -e OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  -e OPENAI_API_KEY=<你的API Key> \
  -e OPENAI_EMBEDDING_MODEL=text-embedding-v2 \
  -e OPENAI_LLM_MODEL=glm-5 \
  -e VECTOR_DB_TYPE=opengauss \
  -e "OPENGAUSS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=Huawei@123" \
  -e "PYTHONPATH=/data1/sundechao/ogmem/oG-Memory:/home/sundechao/.local/lib/python3.13/site-packages" \
  alpine/openclaw:latest
```

## 5. 启动 IndexService

```bash
cd /data1/sundechao/ogmem/oG-Memory

export CONTEXTENGINE_PROVIDER=openai
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<你的API Key>"
export OPENAI_EMBEDDING_MODEL="text-embedding-v2"
export OPENAI_LLM_MODEL="glm-5"
export VECTOR_DB_TYPE=opengauss
export OPENGAUSS_CONNECTION_STRING="host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=Huawei@123"
export AGFS_BASE_URL="http://127.0.0.1:1833"
export OG_ACCOUNT_ID="acct-demo"
export INDEX_INTERVAL=15

setsid /data2/miniconda/bin/python scripts/run_index_service.py \
  > /tmp/index_service.log 2>&1 < /dev/null & disown
```

## 6. 验证

```bash
# 容器状态
docker logs openclaw-ogmem 2>&1 | tail -10

# Health
curl http://127.0.0.1:18789/health

# 发消息测试
curl -s -X POST http://127.0.0.1:18789/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token-123" \
  -d '{"model":"openclaw","input":"你好","user":"test-user"}'

# IndexService 日志
tail -20 /tmp/index_service.log

# openGauss 数据
gsql -U sundechao -W Huawei@123 -d postgres -p 8799 -r -c \
  "SELECT id, uri, level, substring(text,1,60) FROM vector_index ORDER BY created_at DESC LIMIT 5;"
```

## 7. 重启 / 重建

```bash
# 重启容器
docker restart openclaw-ogmem

# 重建容器（改配置后）
docker stop openclaw-ogmem && docker rm openclaw-ogmem
# 然后重新执行第 4 步

# 重启 IndexService
pkill -f run_index_service
# 然后重新执行第 5 步
```
