# oGMemory + openGauss启动说明

本文档分为两部分：

- 第一部分：启动 openGauss，作为 oGMemory 对接的向量数据库。
- 第二部分：配置并启动 oGMemory。

## 使用镜像

| 组件 | 镜像 |
| --- | --- |
| openGauss | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328` |
| oGMemory | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc1_428` |

## 第一部分：openGauss

### 1. 创建数据库目录

openGauss 数据会持久化到宿主机 `/your_directory/opengauss`：

```bash
mkdir -p /your_directory/opengauss
chmod 777 /your_directory/opengauss
```

### 2. 启动 openGauss 容器

启动命令：

```bash
docker run --name opengauss --network host --privileged=true -d \
  -e 'GS_PASSWORD=<OPENGAUSS_PASSWORD>' \
  -e 'GS_PORT=<OPENGAUSS_PORT>' \
  -v /your_directory/opengauss:/var/lib/opengauss \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328
```

参数说明：

- `<OPENGAUSS_PASSWORD>`：openGauss 数据库密码，后面需要写入 `ogmemory.yaml`。
- `<OPENGAUSS_PORT>`：openGauss 监听端口，例如 `5432` 或 `15432`。如果宿主机 `5432` 已被占用，可以改成其他空闲端口。


### 3. 更新 oGMemory 数据库配置

openGauss 启动后，需要在 `/your_directory/ogmemory.yaml` 中配置同一个端口和密码：

```yaml
vector_db:
  type: opengauss
  connection_string: "host=127.0.0.1 port=<OPENGAUSS_PORT> dbname=postgres user=gaussdb password=<OPENGAUSS_PASSWORD>"
  dimension: 1024
  table_name: vector_index
  pool_size: 5
```

注意：

- `port=<OPENGAUSS_PORT>` 必须和启动 openGauss 时的 `GS_PORT` 一致。
- `password=<OPENGAUSS_PASSWORD>` 必须和启动 openGauss 时的 `GS_PASSWORD` 一致。
- `dimension` 必须和 Embedding 模型输出维度一致，示例值为 `1024`。

## 第二部分：oGMemory

### 1. 创建 oGMemory 数据目录

oGMemory 的配置文件和 AGFS 记忆数据目录如下：

| 宿主机路径 | 容器路径 | 用途 |
| --- | --- | --- |
| `/your_directory/ogmemory.yaml` | `/etc/ogmem/config.yaml` | oGMemory 配置文件 |
| `/your_directory/ogmem` | `/data/agfs` | AGFS 记忆数据目录 |

首次部署时创建 AGFS 数据目录并赋予写权限：

```bash
mkdir -p /your_directory/ogmem
chmod 777 /your_directory/ogmem
```

### 2. 准备配置文件

复制示例配置并填写模型、Embedding 和数据库信息：

```bash
cp /your_directory/ogmemory.example.yaml \
  /your_directory/ogmemory.yaml
vim /your_directory/ogmemory.yaml
```

需要重点修改：

- `llm.api_key`：模型服务 API Key。
- `llm.base_url`：LLM 的 OpenAI-compatible API 地址，示例中为 `"<YOUR_LLM_BASE_URL>"`。
- `llm.model`：用户实际使用的 LLM 模型名。
- `embedding.api_key`：Embedding 服务 API Key；如果和 LLM 共用密钥，可填写同一个值。
- `embedding.base_url`：Embedding 服务 API 地址，示例中为 `"<YOUR_EMBEDDING_BASE_URL>"`。
- `embedding.model`：用户实际使用的 Embedding 模型名。
- `vector_db.connection_string`：按第一部分 openGauss 的端口和密码填写。
- `vector_db.dimension`：向量维度，必须和 Embedding 模型输出维度一致。

### 3. 启动 oGMemory 容器

```bash
docker run -d --name ogmem --network host \
  -e OGMEM_CONFIG=/etc/ogmem/config.yaml \
  -v /your_directory/ogmemory.yaml:/etc/ogmem/config.yaml:ro \
  -v /your_directory/ogmem:/data/agfs \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc1_428
```

说明：

- `--network host` 会让 oGMemory 直接使用宿主机网络。
- oGMemory HTTP API 默认监听宿主机 `8090` 端口。
- 配置文件以只读方式挂载到容器的 `/etc/ogmem/config.yaml`。
- 记忆数据会写入宿主机 `/your_directory/ogmem`。


### 4. 验证 oGMemory

查看容器状态：

```bash
docker ps --filter name=ogmem
```

查看启动日志：

```bash
docker logs -f ogmem
```

健康检查：

```bash
curl http://127.0.0.1:8090/api/v1/health
```

配置修改后需要重启容器才能生效：

```bash
docker restart ogmem
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `Unknown provider type` | `llm.provider` 使用 `openai`、`openai-cached` 或 `mock`。即使后端是兼容 OpenAI 协议的模型服务，LLM provider 也通常填写 `openai`。 |
| `OPENGAUSS_CONNECTION_STRING is required` | 使用 openGauss 时必须在 `vector_db.connection_string` 填写完整连接串。 |
| 无法连接 openGauss | 确认 `opengauss` 容器已启动，并且 `connection_string` 中的端口、密码分别和 `GS_PORT`、`GS_PASSWORD` 一致。 |
| 宿主机端口被占用 | host 网络模式下不要使用 `-p`，需要把 openGauss 的 `GS_PORT` 改成空闲端口，并同步修改 `connection_string` 的 `port`。 |
| Embedding 维度不匹配 | 修改 `vector_db.dimension`，设置为 Embedding 模型实际输出维度。 |
| AGFS 写入失败 | 检查 `/your_directory/ogmem` 是否存在且容器用户有写权限。 |

---

## 补充：复用一键部署流程接入 OpenClaw

本节只补充原文缺少的 OpenClaw、Context Engine 和 MCP 接入。原有 openGauss、oGMemory 手工部署步骤保持不变。

部署关系：

| 流程 | 部署内容 |
| --- | --- |
| [`deploy/README.md`](README.md) 一键脚本 | openGauss、oGMemory、Context Engine 版 OpenClaw |
| 本文补充的手工步骤 | MCP 版 OpenClaw |

MCP 版 OpenClaw 不合入一键脚本。

### 1. 当前验证镜像

原文的 `ogmemory:poc1_428` 不包含当前 MCP 能力。需要 MCP 时，在一键部署配置和后续命令中使用：

```bash
OGMEM_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc_0805'
OPENCLAW_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0805'
```

### 2. 复用一键部署

无需拉取完整代码仓。下载官方部署文件：

```bash
export OGMEM_STACK='/path/to/ogmemory-stack'
mkdir -p "$OGMEM_STACK"
cd "$OGMEM_STACK"

curl -fsSL https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/install.sh | bash
cd "$OGMEM_STACK/ogmem-deploy"
```

编辑 `deploy.env`，在原模板中补齐或替换以下值：

```bash
LLM_PROVIDER='openai'
LLM_API_KEY='替换为真实 API Key'
LLM_BASE_URL='替换为 OpenAI-compatible 地址'
LLM_MODEL='替换为模型名'

ENABLE_OPENGAUSS='true'
OG_HOST_PORT='15432'
OG_CONTAINER_NAME='opengauss'

OGMEM_CONTAINER_NAME='ogmem'
OGMEM_HTTP_PORT='8090'
OGMEM_MCP_ENABLED='true'
OGMEM_MCP_HOST='127.0.0.1'
OGMEM_MCP_PORT='8091'
OGMEM_MCP_PATH='/mcp'
OGMEM_MCP_DEFAULT_TOOLSET='all'
OGMEM_MCP_ENABLE_WRITE_TOOLS='true'
OGMEM_MCP_ENABLE_DELETE_TOOLS='false'
OGMEM_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc_0805'

OG_ROOT_API_KEY='替换为仅用于管理和初始化的 oGMemory Root Key'
OG_TENANT_API_KEY='替换为 acct-demo 租户级 Admin Key'
OG_ADMIN_API_KEYS="acct-demo:${OG_TENANT_API_KEY}"
OG_ACCOUNT_ID='acct-demo'
OG_USER_ID='ce-user'
OG_AGENT_ID='ce-agent'
OG_AUTH_API_KEY="${OG_TENANT_API_KEY}"
OG_AUTH_ACCOUNT_ID='acct-demo'

OPENCLAW_CONTAINER_NAME='openclaw_ce'
OPENCLAW_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0805'
GATEWAY_PORT='18790'
OPENCLAW_GATEWAY_TOKEN='替换为 CE Gateway Token'
OGMEM_URL='http://127.0.0.1:8090'

AGFS_DATA_DIR='/path/to/ogmemory-stack/agfs-data'
OPENCLAW_HOME_DIR='/path/to/ogmemory-stack/openclaw-ce'
```

Root Key 拥有跨租户最高权限，只用于管理、初始化和紧急诊断。生产环境不要把 Root Key 下发给 Context Engine 或 MCP Agent；应使用与 `OG_AUTH_ACCOUNT_ID`、`X-Account-ID` 匹配的租户级 Admin/Member Key。上例使用 `acct-demo` 的 Admin Key。仅本机临时调试时，才可将 `OG_TENANT_API_KEY` 临时设置为 Root Key。

复制并编辑 oGMemory 配置：

```bash
cp ogmemory.example.yaml ogmemory.yaml
vim ogmemory.yaml
```

除原文要求外，补充 MCP 配置：

```yaml
mcp:
  enabled: true
  host: "127.0.0.1"
  port: 8091
  path: "/mcp"
  default_toolset: "all"
  enable_write_tools: true
  enable_delete_tools: false
  allowed_hosts:
    - "127.0.0.1"
    - "localhost"

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "${OG_ADMIN_API_KEYS}"
```

执行：

```bash
export OG_DB_PASSWORD='替换为符合复杂度要求的 openGauss 密码'
bash deploy.sh -password "$OG_DB_PASSWORD"
```

查看状态：

```bash
bash deploy.sh --status
curl -fsS http://127.0.0.1:8090/api/v1/health
curl -fsS http://127.0.0.1:18790/health
```

> 当前一键脚本没有给 openGauss 挂载宿主机数据目录。功能验证可以直接复用；正式环境需要保留原文第一部分的持久化方式。

### 3. 补齐 Context Engine Hook 权限

当前镜像需要 `hooks.allowConversationAccess=true`。缺少该项时 Gateway 健康，但日志会出现 `typed hook ... blocked`，不会自动执行 `compose` 和 `afterTurn`。

```bash
cd "$OGMEM_STACK/ogmem-deploy"
set -a
source ./deploy.env
set +a

python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OPENCLAW_HOME_DIR"]) / "openclaw.json"
data = json.loads(path.read_text(encoding="utf-8"))
entry = data["plugins"]["entries"]["og-memory-context-engine"]
entry["hooks"] = {"allowConversationAccess": True}
entry["config"].update({
    "accountId": "acct-demo",
    "userId": "ce-user",
    "agentId": "ce-agent",
    "prefetchEnabled": True,
    "prefetchTopK": 5,
    "shortTermIndexMode": "sync",
})
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
path.chmod(0o600)
PY

docker restart openclaw_ce
```

确认 Hook 没有被阻止：

```bash
curl -fsS http://127.0.0.1:18790/health
docker logs openclaw_ce --since 2m 2>&1 | grep 'typed hook.*blocked' && exit 1 || true
```

### 4. 单独启动 MCP 版 OpenClaw

准备独立目录。MCP 与 Context Engine 不能共用 OpenClaw HOME：

```bash
cd "$OGMEM_STACK/ogmem-deploy"
set -a
source ./deploy.env
set +a

mkdir -p "$OGMEM_STACK/openclaw-mcp"
chmod 700 "$OGMEM_STACK/openclaw-mcp"
```

生成完整配置：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

cfg = {
    "agents": {
        "defaults": {
            "timeoutSeconds": 600,
            "model": {"primary": "ogm-openai/${LLM_MODEL}"},
        },
    },
    "models": {
        "mode": "merge",
        "providers": {
            "ogm-openai": {
                "baseUrl": "${LLM_BASE_URL}",
                "apiKey": "${LLM_API_KEY}",
                "api": "openai-completions",
                "timeoutSeconds": 300,
                "models": [{
                    "id": "${LLM_MODEL}",
                    "name": "${LLM_MODEL}",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                    "contextWindow": 131072,
                    "maxTokens": 8192,
                }],
            },
        },
    },
    "gateway": {
        "mode": "local",
        "bind": "loopback",
        "port": 18789,
        "auth": {"token": "${MCP_GATEWAY_TOKEN}"},
        "http": {"endpoints": {"responses": {"enabled": True}}},
    },
    "tools": {
        "profile": "minimal",
        "alsoAllow": [
            "ogmemory__ogm_query_memory",
            "ogmemory__ogm_read_memory",
            "ogmemory__ogm_write_memory",
            "ogmemory__resources_list",
            "ogmemory__resources_read",
        ],
    },
    "mcp": {
        "servers": {
            "ogmemory": {
                "url": "http://127.0.0.1:8091/mcp",
                "transport": "streamable-http",
                "headers": {
                    "X-API-Key": "${OG_TENANT_API_KEY}",
                    "X-Account-ID": "acct-demo",
                    "X-User-ID": "mcp-user",
                    "X-Agent-ID": "mcp-agent",
                },
                "connectTimeout": 30,
                "timeout": 300,
                "supportsParallelToolCalls": True,
                "toolFilter": {"include": ["ogm_*", "resources_*"]},
            },
        },
    },
    "plugins": {
        "slots": {"contextEngine": "legacy"},
        "entries": {
            "og-memory-context-engine": {
                "enabled": False,
                "config": {
                    "mode": "remote",
                    "memoryApiBaseUrl": "http://127.0.0.1:8090",
                    "authApiKey": "${OG_TENANT_API_KEY}",
                    "authAccountId": "acct-demo",
                },
            },
            "memory-core": {"enabled": False},
        },
    },
}

out = Path(os.environ["OGMEM_STACK"]) / "openclaw-mcp" / "openclaw.json"
out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
out.chmod(0o600)
print(out)
PY
```

`toolFilter` 使用 `ogm_*`，不能写成 `ogm.*`。

启动：

```bash
export OPENCLAW_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0805'
export MCP_GATEWAY_TOKEN='替换为 MCP Gateway Token'

docker run -d \
  --name openclaw_mcp \
  --restart unless-stopped \
  --network host \
  -v "$OGMEM_STACK/openclaw-mcp:/home/node/.openclaw" \
  -e "LLM_API_KEY=$LLM_API_KEY" \
  -e "LLM_BASE_URL=$LLM_BASE_URL" \
  -e "LLM_MODEL=$LLM_MODEL" \
  -e "OG_TENANT_API_KEY=$OG_TENANT_API_KEY" \
  -e "MCP_GATEWAY_TOKEN=$MCP_GATEWAY_TOKEN" \
  -e 'NO_PROXY=*' \
  -e 'no_proxy=*' \
  "$OPENCLAW_IMAGE" gateway
```

### 5. 验证两种接入

MCP：

```bash
curl -fsS http://127.0.0.1:18789/health

docker exec -u node -e HOME=/home/node \
  openclaw_mcp openclaw mcp probe ogmemory --json
```

预期 `diagnostics` 为空，并包含：

```text
ogmemory__ogm_query_memory
ogmemory__ogm_read_memory
ogmemory__ogm_write_memory
```

Context Engine：

```bash
cd "$OGMEM_STACK/ogmem-deploy"
set -a
source ./deploy.env
set +a

export CE_SMOKE_SESSION="ce-smoke-$(date +%s)"

curl -fsS http://127.0.0.1:18790/health

curl -fsS -X POST http://127.0.0.1:18790/v1/responses \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENCLAW_GATEWAY_TOKEN" \
  -H 'X-OpenClaw-Agent-ID: main' \
  -H "X-OpenClaw-Session-Key: $CE_SMOKE_SESSION" \
  -d '{
    "model": "openclaw",
    "input": "只回复：Context Engine 对接正常",
    "stream": false
  }'

docker logs openclaw_ce --since 5m 2>&1 | grep -E \
  '\[og-memory\].*(assemble|compose|afterTurn)' | tail -n 100

unset CE_SMOKE_SESSION
```

Responses API 应返回包含“Context Engine 对接正常”的回答，日志至少应出现 `assemble`、`compose` 和 `afterTurn called`。

### 6. 配置目录判断

不要假定所有镜像都使用相同用户。先检查：

```bash
docker inspect openclaw_mcp --format 'default_user={{json .Config.User}}'
docker exec openclaw_mcp sh -lc \
  'echo "exec_user=$(id -un) home=$HOME"; find /root /home -path "*/.openclaw/openclaw.json" 2>/dev/null'
```

本教程镜像的 Gateway 使用 `node` 用户和 `/home/node/.openclaw`，所以 probe 使用：

```bash
docker exec -u node -e HOME=/home/node \
  openclaw_mcp openclaw mcp probe ogmemory --json
```
