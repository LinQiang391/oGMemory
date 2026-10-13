# 使用 Docker Compose 部署 oGMemory

oGMemory 提供 Docker Compose 部署配置，可从源码构建并启动 oGMemory 服务和 openGauss 向量数据库。本指南介绍环境准备、模型配置、服务启动、客户端接入和数据维护。

按第 1–7 节依次操作，可完成部署并验证模型调用、记忆写入与召回、HTTP REST 接入及数据持久化。所有命令均在部署主机的终端执行。

- 首次部署：从第 1 节开始。
- 服务已启动：从第 5 节初始化身份，再完成第 6–7 节验证。
- 日常维护或部署失败：参阅第 8–9 节。

本指南使用 HTTP REST API 验证客户端接入，无需安装 OpenClaw 或其他 Agent。

## 1. 检查端口和部署环境

### 1.1 首先检查宿主端口

在准备部署 oGMemory 的主机上执行以下检查；此步骤不要求进入代码目录。

先检查计划使用的宿主端口。默认 HTTP 为 8090、MCP 为 8091；若已占用，选择其他空闲端口，并在后面的 `.env` 中填写。检查其他端口时，也相应修改下面命令中的端口号。

```sh
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

按部署主机的操作系统选择一条检查命令。通过 SSH 连接 Linux 服务器时，使用 Linux 命令：

```sh
# Linux：检查宿主机监听端口。
ss -ltn '( sport = :8090 or sport = :8091 )'
```

```sh
# macOS：使用下面命令代替 ss。
lsof -nP -iTCP:8090 -iTCP:8091 -sTCP:LISTEN
```

若输出中出现所选端口，表示已有监听或 Docker 端口映射。请改选端口并再次检查；例如检查 18110/18111：

```sh
# Linux
ss -ltn '( sport = :18110 or sport = :18111 )'
```

```sh
# macOS
lsof -nP -iTCP:18110 -iTCP:18111 -sTCP:LISTEN
```

确定可用端口后，再继续下面步骤，并将所选端口填入 `.env` 的 `OGMEM_HTTP_HOST_PORT` 和 `OGMEM_MCP_HOST_PORT`。

### 1.2 确认工作目录和运行条件

获取包含 `compose.yaml` 和配套部署文件的 oGMemory 源码版本，将其放置在自行选择的工作目录中。以下命令中的 `/path/to/oGMemory` 表示本地仓库路径，请替换为实际路径：

```sh
cd /path/to/oGMemory
ls compose.yaml .env.example docker/Dockerfile.compose deploy/compose/ogmemory.yaml
docker info --format '{{.Architecture}}'
docker compose version
```

后续所有 `docker compose` 命令均在该仓库根目录执行。执行前确认上述文件存在；`deploy/compose` 子目录不是命令执行目录。

需要准备：

- 正在运行的 Docker Engine 或 Docker Desktop，及 Docker Compose 插件，建议 Compose 2.20 或更新版本。
- **ARM64 环境**：当前默认 openGauss 镜像仅支持 ARM64，Compose 两个服务都指定 `linux/arm64`。可使用 ARM64 Linux 主机或 ARM64 Docker Desktop，部署后按本指南检查服务。
- Python 3（用于运行构建源检查，无需安装额外 Python 包）。
- 可以从容器访问的 LLM 和 Embedding 服务，以及各自的地址、模型名称和访问密钥。
- 首次构建可访问 Docker Hub、Go 模块源、apt、PyPI 和 openGauss 镜像仓库。宿主机不必安装 Go，也不必预先编译 AGFS。

## 2. 部署架构

Compose 启动两个容器，LLM 和 Embedding 服务需单独配置：

- **ogmemory 容器**：HTTP API（8090）提供生命周期与记忆接口；MCP（8091）提供客户端工具；AGFS（1833）保存记忆正文、摘要和控制数据；索引进程调用 Embedding，将向量写入数据库。
- **opengauss 容器**：openGauss（5432）保存向量索引。
- **LLM 与 Embedding 服务**：由用户提供，分别负责记忆提取与文本向量化，可使用云端 API 或宿主机模型服务。

```text
客户端 → 宿主机 HTTP 端口 → ogmemory:8090
       → 宿主机 MCP 端口  → ogmemory:8091/mcp
ogmemory 内部进程 → 127.0.0.1:1833 → AGFS → /data/agfs
                 → opengauss:5432 → openGauss
                 → 配置的模型 Base URL → LLM / Embedding
```

数据库与 oGMemory 通过项目专属 bridge 网络通信，使用服务名 `opengauss`，不依赖固定 IP。AGFS 和数据库端口默认不发布到宿主机。网络允许容器访问外部模型。

AGFS 的 `/local` 映射容器目录 `/data/agfs`，oGMemory 使用其下的 `/local/plugin`。正文与向量分开保存，因此后续需要保留、备份两个数据卷。

## 3. 创建并填写 .env

在代码目录中执行以下命令：`.env` 不存在时，从 `.env.example` 创建；已有 `.env` 时保留其内容。`chmod` 只调整文件权限，不改变配置。

```sh
if [ ! -e .env ]; then
  cp .env.example .env
fi
chmod 600 .env
```

随后打开 **`.env`** 填写配置，例如：

```sh
vi .env
```

### 3.1 按文件顺序填写配置

`.env.example` 按部署与端口、镜像与数据库、LLM、Embedding、身份与密钥、MCP 与记忆提取分成六组。下面的代码块与模板中的变量顺序、默认值和注释完全一致，可以从上到下逐项填写。注释标为“必填”的空值必须补齐，其他项按需要调整。

已有 `.env` 不会自动合并模板；如果缺少某一项，请参照 `.env.example` 补充，并保留已有值。

```dotenv
# 复制为 .env 后编辑。不要提交 .env。

# 1. 部署名称和宿主机端口
# 可保留默认值：项目名，默认 'ogmemory'；多实例使用不同名称，创建后保持不变。
COMPOSE_PROJECT_NAME=ogmemory
# 可保留默认值：宿主机监听地址，默认 '127.0.0.1'；需远程访问时按网络和访问控制要求调整。
PUBLISH_IP=127.0.0.1
# 可保留默认值：HTTP 宿主端口，默认 '8090'；已占用时换成空闲端口。
OGMEM_HTTP_HOST_PORT=8090
# 可保留默认值：MCP 宿主端口，默认 '8091'；已占用时换成空闲端口。
OGMEM_MCP_HOST_PORT=8091
# 可选：默认地址池耗尽时，确认无冲突后取消以下两项注释；多实例须使用不同网段。
# OGMEM_NETWORK_SUBNET=10.203.78.0/24
# COMPOSE_FILE=compose.yaml:deploy/compose/compose.network.yaml

# 2. 镜像与数据库
# 可保留默认值：openGauss 数据库镜像；模板中的镜像仅支持 ARM64。
OPENGAUSS_IMAGE=swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328
# 可保留默认值：从当前源码构建的本地镜像名称；多实例分别构建时使用不同名称。
OGMEM_IMAGE=ogmemory/standalone:compose-local
# 可保留默认值：Go 基础镜像；不能访问 Docker Hub 时填写可访问镜像仓库中的同版本 ARM64 镜像地址。
GO_BASE_IMAGE=golang:1.25-bookworm
# 可保留默认值：Python 基础镜像；替换时保留 Python 3.11、Debian bookworm 和 ARM64。
PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
# 按网络选择：Go 依赖源；可选 https://goproxy.cn,direct、https://proxy.golang.org,direct 或内部源。
GOPROXY=https://goproxy.cn,direct
# 可保留默认值：Go 依赖校验服务；代理源可转发该服务。特殊网络可填写经验证的校验服务配置。
GOSUMDB=sum.golang.org
# 按网络选择：Debian 软件源地址；可替换为 https://deb.debian.org/debian 或内部镜像地址。
DEBIAN_MIRROR=https://mirrors.ustc.edu.cn/debian
# 按网络选择：Debian 安全更新源地址；可替换为 https://deb.debian.org/debian-security 或内部镜像地址。
DEBIAN_SECURITY_MIRROR=https://mirrors.ustc.edu.cn/debian-security
# 按网络选择：Python 包索引；下载慢或失败时可换用 https://repo.huaweicloud.com/repository/pypi/simple、https://pypi.org/simple 或内部 PyPI 地址。
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# 必填：8–32 字符，满足数据库镜像复杂度要求，包含大小写字母、数字和特殊字符。
OG_DB_PASSWORD=

# 3. LLM 服务
# 可保留默认值：默认 'openai'，用于 OpenAI 兼容接口。
LLM_PROVIDER=openai
# 必填：填写容器可访问的模型 API Base URL；云端使用供应商地址，宿主机服务使用 http://host.docker.internal:端口/v1。
LLM_BASE_URL=
# 必填：供应商支持的实际模型名称。
LLM_MODEL=
# 必填：模型供应商的访问密钥。
LLM_API_KEY=
# 可留空：可选供应商参数 JSON；不需要时留空。
LLM_EXTRA_BODY=

# 4. Embedding 服务
# 可保留默认值：默认 'openai'，也支持现有 'volcengine' 适配器。
EMBEDDING_PROVIDER=openai
# 必填：填写容器可访问的 Embedding API Base URL，不附加 /embeddings；宿主机服务使用 http://host.docker.internal:端口/v1。
EMBEDDING_BASE_URL=
# 必填：实际使用的 Embedding 模型名称。
EMBEDDING_MODEL=
# 必填：Embedding 服务密钥；无鉴权的本地兼容服务可填其接受的占位值。
EMBEDDING_API_KEY=
# 必填：模型实际输出的向量维度，例如 '1024'；必须与数据库索引一致。
EMBEDDING_DIMENSION=

# 5. 身份与服务密钥
# 可保留默认值：账户标识，默认 'compose-account'。
OG_ACCOUNT_ID=compose-account
# 可保留默认值：用户标识，默认 'compose-user'。
OG_USER_ID=compose-user
# 可保留默认值：Agent 标识，默认 'compose-agent'。
OG_AGENT_ID=compose-agent
# 必填：初始化和管理服务使用的根密钥，使用下方命令生成。
OG_ROOT_API_KEY=
# 必填：客户端使用的租户管理密钥，另行生成，必须与根密钥不同。
OG_TENANT_API_KEY=

# 6. MCP 与记忆提取
# 可保留默认值：默认 'true'，启用 MCP 服务。
OGMEM_MCP_ENABLED=true
# 可保留默认值：MCP 允许的主机名列表；客户端使用其他主机名或 IP 时补充对应值。
OGMEM_MCP_ALLOWED_HOSTS=localhost,127.0.0.1,ogmemory,host.docker.internal
# 可保留默认值：自动记忆提取阈值，默认 '200'。
OGMEM_AFTER_TURN_THRESHOLD=200
```

后续示例使用项目名 `ogmemory` 和宿主端口 8090/8091。如果使用其他值，需同步调整客户端地址。部署多个实例时，各实例使用不同的项目名、本地镜像名和宿主端口。

生成两次随机值，分别填入 `OG_ROOT_API_KEY` 与 `OG_TENANT_API_KEY`：

```sh
openssl rand -hex 32
openssl rand -hex 32
```

模型密钥、数据库密码、服务访问密钥是三种不同的凭据。不要把模型密钥填进服务访问密钥字段。

`.env` 由 Compose 读取，不是 shell 脚本，不要执行 `source .env`。含 `$` 等特殊字符的值可用 Compose 支持的单引号字面值填写；例如 `OG_DB_PASSWORD='实际数据库密码'`。不要提交 `.env` 或分享展开后的 `docker compose config` 输出。启动时的解析配置保存在容器内权限为 600 的临时文件。

## 4. 检查配置、构建、启动

### 4.1 选择构建源并检查

构建需要访问镜像仓库、Go 依赖源、Debian 软件源和 Python 包索引。源地址在 `.env` 中填写，Compose 会自动用于镜像构建，无需修改 Dockerfile 或宿主机的 Go、pip 配置。默认地址可保留，但部署前应在当前网络中检查是否可用。

镜像相关配置见第 3 节的 `OPENGAUSS_IMAGE`、`GO_BASE_IMAGE` 和 `PYTHON_BASE_IMAGE`。无法访问对应仓库时，填写可访问仓库中的兼容 ARM64 镜像地址；Go、Python 基础镜像应保留模板中的版本和 Debian bookworm 系列。`OGMEM_IMAGE` 是本地构建结果的名称。

依赖源按第 3 节的顺序填写。例如，需要更换 Python 包索引时，可以将已有 `.env` 中对应项改为下面的组合；请修改已有值，不重复追加同名变量：

```dotenv
# 按网络选择：Go 依赖下载代理；也可使用 https://proxy.golang.org,direct 或内部源。
GOPROXY=https://goproxy.cn,direct
# 可保留默认值：Go 依赖校验服务，保留完整性校验。
GOSUMDB=sum.golang.org
# 按网络选择：Debian 软件源；也可使用 https://deb.debian.org/debian 或内部源。
DEBIAN_MIRROR=https://mirrors.ustc.edu.cn/debian
# 按网络选择：Debian 安全更新源；也可使用 https://deb.debian.org/debian-security 或内部源。
DEBIAN_SECURITY_MIRROR=https://mirrors.ustc.edu.cn/debian-security
# 按网络选择：此示例使用华为 Python 包索引。
# 也可使用模板中的 https://mirrors.aliyun.com/pypi/simple/、https://pypi.org/simple 或内部源。
PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
```

不同网络下同一源的速度可能不同。Python 包下载持续过慢时，优先更换 `PIP_INDEX_URL`，然后重新执行下面的检查；检查会实际安装本项目需要的依赖，不能只凭源首页能够打开判断可用。

若所在网络要求使用代理，可在 `.env` 中补充以下配置。将示例地址替换为构建容器和应用容器均可访问的实际代理地址：

```dotenv
# 按需填写：无需代理时不用添加；宿主机代理须监听容器能够访问的地址。
HTTP_PROXY=http://host.docker.internal:7890
HTTPS_PROXY=http://host.docker.internal:7890
# 使用代理时保留：本地服务、数据库和宿主机模型接口直接访问。
NO_PROXY=localhost,127.0.0.1,opengauss,ogmemory,host.docker.internal
```

以上代理配置用于构建和应用。Docker 拉取镜像所需的代理应在 Docker 服务中配置，不能仅靠 `.env` 中的代理项解决。

保存 `.env` 后，在项目根目录执行：

```sh
docker compose config --quiet
python3 scripts/check_build_network.py
```

检查成功会输出 `Build source check passed`。脚本确认三个 ARM64 镜像可用：已有本地镜像直接使用，缺失时从配置的仓库拉取。随后在实际 Docker 构建环境下载并校验 AGFS 依赖，更新 Debian 软件包索引、安装系统依赖和 Python 包；不启动应用或数据库，也不调用模型。检查会下载依赖并占用构建缓存空间。缓存镜像可用仅表示当前主机能够使用该镜像，不代表对应仓库可以在线访问；首次部署前请准备可访问的镜像地址。

检查失败时按输出中的步骤调整对应源，再重新检查。例如 Go 下载超时修改 `GOPROXY`，apt 的 DNS、连接或重定向失败修改两个 Debian 源，pip 下载失败修改 `PIP_INDEX_URL`。默认源不保证在所有网络可达；可替换为自行验证的源或内部镜像仓库。保留 Go 依赖校验，不通过关闭校验规避网络问题。

### 4.2 构建并启动

构建源检查通过后执行。Docker 默认自动分配项目网段；若启动提示地址池耗尽，按第 9.2 节配置无冲突网段后重试。

```sh
# 不打印密钥；若缺少必填变量，按错误提示补齐 .env。
docker compose config --quiet
# 从当前源码编译 AGFS，构建本地 oGMemory 镜像。
docker compose build ogmemory
# 首次数据库初始化可能需要数分钟。
docker compose up -d --wait --wait-timeout 900
docker compose ps
```

预期看到 `opengauss` 和 `ogmemory` 两个服务均为 `healthy`。首次构建耗时取决于镜像和依赖下载速度。

Compose 使用 `docker/Dockerfile.compose` 构建镜像，入口脚本负责容器内进程的启动和退出。数据库就绪后才启动应用；应用先等待 AGFS，再启动索引、可选 MCP 和 HTTP。关键子进程退出会触发容器退出，由 `unless-stopped` 策略重启。

在宿主机终端执行完整命令，检查容器内的本地依赖；该检查不调用模型。不要先单独运行 `python3` 进入 `>>>` 交互模式：

```sh
docker compose exec ogmemory python3 /opt/ogmem/scripts/check_deployment.py --local
```

默认启用 MCP 时，成功结果为：

```json
{"http_agfs":{"ok":true},"database_vector":{"ok":true},"mcp_listener":{"ok":true}}
```

`healthy` 只说明本地依赖可用，模型连接、身份和记忆写入继续按下文验证。

## 5. 初始化用户和 Agent

```sh
docker compose exec ogmemory python3 /opt/ogmem/scripts/bootstrap_compose.py
```

预期输出：

```json
{"identity_ready":true}
```

启用认证后，仅填写 Account/User/Agent 环境变量还不够：配置中的账户通过管理员配置登记，初始化脚本再通过管理 API 创建缺少的用户和归属于该用户的 Agent。这些控制数据保存在 AGFS 数据卷中。

脚本使用根密钥，不调用模型；重复执行不轮换已有用户密钥，也不修改 Agent 归属。已有 Agent 属于其他用户时会失败，需核对身份配置。

单用户部署可使用 `OG_TENANT_API_KEY` 接入客户端。它具有该账户管理员权限；多用户使用时应通过管理 API 另行发放用户密钥，根密钥仅用于初始化和管理。

## 6. 验证模型与记忆链路

### 6.1 验证模型连接

```sh
docker compose exec ogmemory python3 /opt/ogmem/scripts/check_deployment.py --models
```

这会真实调用 LLM 和 Embedding，可能产生费用。预期所有检查为 `ok:true`，并增加 `"models":{"ok":true}`。检查会校验 Embedding 向量维度与可用性。输出只包含阶段和异常类型，失败时按第 9 节排查。

### 6.2 写入虚构记忆并跨会话召回

```sh
docker compose exec ogmemory python3 /opt/ogmem/scripts/compose_memory_smoke.py --write
```

这会新增一条带随机标记的虚构项目测试输入，有真实模型调用和持久化副作用。提取等待最长设置为 600 秒；随后轮询索引召回，并读取实际记忆正文。索引轮询窗口为 120 秒，单次 API 请求另有超时，因此总运行时间可能更长。

通过条件：

- `status` 为 `completed`，`writes_completed` 大于 0，`writes_failed` 为 0。
- 随后输出 `recalled:true` 和实际记忆 `uri`。

写入数量随模型提取结果变化。仅收到 accepted、HTTP 200 或查询文本回显均不算成功。保存输出的 `marker` 字段值，下一节会使用它。成功输出的形式如下；`marker` 和 `uri` 每次不同，不能直接复制示例值进行验证：

```json
{"marker":"青柚部署验收<本次随机标记>"}
{"status":"completed","writes_completed":4,"writes_failed":0,"retryable":null}
{"recalled":true,"uri":"ogmemory://memories/...","marker":"青柚部署验收<本次随机标记>"}
```

只看到第一行 marker 时，表示脚本仍在等待提取或检索结果。可在另一个终端进入同一项目目录，用 `docker compose logs --tail 100 ogmemory` 查看进展。

失败时保留 marker 和数据，不直接重放写入。客户端超时不证明服务端没有完成写入；脚本没有自动重放写入逻辑。

## 7. 验证持久化和客户端访问

### 7.1 删除并重新创建容器

先在宿主机终端执行下面的命令。看到提示后，粘贴第 6.2 节输出中 `marker` 字段的实际值，再按回车。只粘贴字段值，不带 JSON、引号或 `marker` 字样：

```sh
printf '请输入上一轮写入成功时输出的 marker，然后按回车：'
IFS= read -r MEMORY_MARKER
```

完成输入后，在同一个终端执行下面的代码块。它保留两个数据卷，只操作 `.env` 项目名对应的实例；不要给 `down` 添加 `-v`：

```sh
if [ -n "$MEMORY_MARKER" ]; then
  docker compose down &&
    docker compose up -d --wait --wait-timeout 900 &&
    docker compose exec ogmemory python3 /opt/ogmem/scripts/compose_memory_smoke.py --marker "$MEMORY_MARKER"
else
  printf '%s\n' 'marker 为空，请先执行上面的输入步骤。'
fi
```

请分两步执行，等待 marker 输入完成后再复制第二个代码块。`MEMORY_MARKER` 只保存在当前终端，换终端后需重新输入。验证命令不新增记忆，但检索仍可能调用 Embedding。预期再次输出 `recalled:true`，说明重建后正文和索引仍可用。保留数据卷时不需要重新初始化用户和 Agent。

如果容器已经重建且均为 `healthy`，无需重复 `down/up`。输入实际 marker 后，只执行：

```sh
if [ -n "$MEMORY_MARKER" ]; then
  docker compose exec ogmemory python3 /opt/ogmem/scripts/compose_memory_smoke.py --marker "$MEMORY_MARKER"
else
  printf '%s\n' 'marker 为空，请先执行上面的输入步骤。'
fi
```

若误填了其他文本，脚本会等待召回并最终失败；这不能证明数据丢失。请核对上一轮输出的实际 marker 后重试。

### 7.2 从宿主机访问 HTTP

按本指南端口示例：

```sh
curl --fail --show-error http://127.0.0.1:8090/api/v1/health
```

预期响应 `status` 为 `ok`。健康接口不要求密钥，不能用它证明业务接口认证成功。

### 7.3 按官方 HTTP REST 方式验证接入

本节使用 [HTTP REST API 接入文档](../../docs/integrations/http-rest.md) 中的 `/api/v1/compose` 接口，从宿主机获取记忆上下文。Compose 已启动服务，接入文档中的服务启动命令无需再次执行。

客户端配置与 `.env` 对应：

```text
HTTP Base URL： http://127.0.0.1:8090     # 端口对应 OGMEM_HTTP_HOST_PORT
MCP URL：       http://127.0.0.1:8091/mcp # 端口对应 OGMEM_MCP_HOST_PORT
API Key：       OG_TENANT_API_KEY 的值
Account ID：    OG_ACCOUNT_ID 的值
User ID：       OG_USER_ID 的值
Agent ID：      OG_AGENT_ID 的值
```

在 **Bash 终端**依次执行以下两个代码块。先隐藏输入 `.env` 中的 `OG_TENANT_API_KEY`；输入时不显示字符，粘贴后按回车：

```bash
read -r -s -p '请输入 OG_TENANT_API_KEY，然后按回车：' CLIENT_API_KEY
```

等待密钥输入完成，再复制下面的代码块。以下命令使用默认身份和端口。如果修改过 `.env`，请同步修改 URL、请求头和请求体中的对应值：

```bash
curl --fail --show-error --max-time 180 \
  http://127.0.0.1:8090/api/v1/compose \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $CLIENT_API_KEY" \
  -H 'X-Account-ID: compose-account' \
  -H 'X-User-ID: compose-user' \
  --data-binary @- <<'JSON'
{
  "accountId": "compose-account",
  "userId": "compose-user",
  "agentId": "compose-agent",
  "sessionId": "compose-client-check",
  "prompt": "测试项目使用什么语言？",
  "tokenBudget": 32000
}
JSON
unset CLIENT_API_KEY
```

预期返回 JSON 上下文。完成第 6.2 节写入后，检查 `messages` 中带 `_ogmem:true` 的记忆消息或 `retrievedEvidence` 是否包含测试项目及 Python 信息。返回 JSON 但没有相关记忆，只能说明接口请求成功，仍需核对第 6.2 节的实际 marker、索引和身份。检索可能调用 Embedding，但此请求不提交新的对话用于记忆提取。

Account ID 与 Agent ID 是不同字段。完整生命周期、写入响应及客户端处理方式见 [HTTP REST API 接入文档](../../docs/integrations/http-rest.md)。其他交互方式见 [接入目录](../../docs/integrations/README.md)，包括 [Python API](../../docs/integrations/python-api.md)、[Codex](../../docs/integrations/codex.md) 和 [MCP](../../docs/integrations/mcp.md)。

部署验收通过后，如需运行 LoCoMo sample 0 等效果实验，使用仓库提供的 [Agent 插件评测指南](../../tests/e2e/AGENT_PLUGIN_LOCOMO.md)。效果评测另有数据集、Agent CLI 和评测模型要求，不属于部署必须步骤。

默认 MCP 工具集为只读，HTTP 生命周期写入接口仍可使用。需要 MCP 写入工具时，显式修改 `deploy/compose/ogmemory.yaml` 的工具集和 `enable_write_tools` 后重建应用容器。

远程客户端访问时，调整 `PUBLISH_IP`、宿主机防火墙及访问地址，并把实际服务器域名/IP 加入 `OGMEM_MCP_ALLOWED_HOSTS`。关闭 MCP 可设置 `OGMEM_MCP_ENABLED=false`；进程与健康检查会跳过 MCP，Compose 端口映射声明仍保留。

## 8. 后续操作和数据维护

按需要选择对应操作，不必依次执行全部命令。

查看最近日志（持续查看时加 `-f`，Ctrl+C 只退出日志查看）：

```sh
docker compose logs --tail 100 ogmemory opengauss
```

停止服务，保留容器与数据：

```sh
docker compose stop
```

恢复服务并等待就绪：

```sh
docker compose up -d --wait --wait-timeout 900
```

修改源码后重新构建并更新，保留数据卷：

```sh
docker compose build ogmemory &&
  docker compose up -d --wait --wait-timeout 900
```

修改 `.env` 或挂载配置后使用 `up -d`；单独修改挂载 YAML 时可明确使用 `docker compose up -d --force-recreate ogmemory`。普通 `restart` 不会更新容器的环境变量。Compose 镜像名对应本地构建结果，不代表已有官方发布镜像。

需要同时保留以下两个数据卷：

- `<项目名>_agfs-data` 挂载到 `/data/agfs`，保存正文、摘要、元数据、控制数据、会话状态和索引任务。
- `<项目名>_opengauss-data` 挂载到 `/var/lib/opengauss`，保存数据库及向量索引。

按本文项目名，卷名为 `ogmemory_agfs-data` 和 `ogmemory_opengauss-data`。项目名变化会使用另一组卷，可能表现为“原记忆不见了”。

备份时先停止本项目，同时备份两个数据卷、配置模板和私有 `.env`，保持数据库与正文为同一停机时间点。已有数据库卷的密码不会因修改 `.env` 自动更换。不要在已有索引上直接更换 Embedding 模型或维度；另建项目和数据卷，或执行经过验证的数据迁移。

`docker compose down` 保留数据；**`docker compose down -v` 会删除本项目的数据卷，仅在明确要删除全部实例数据时使用。**

当前配置采用单实例，HTTP 一个进程、多线程；不承诺多副本共享 AGFS 的一致性。数据库中断后仍需要应用重连，Compose 启动依赖不是持续可用性的保证。

## 9. 排查指南

### 9.1 按失败阶段检查

- **找不到 Compose 文件**：回到包含 `compose.yaml` 的仓库根目录。
- **缺少必填变量**：补齐 `.env` 中的模型、密码、密钥和维度；确认编辑的是 `.env`。
- **镜像拉取或构建失败**：重新运行 `python3 scripts/check_build_network.py`，按失败阶段检查镜像仓库、Go、apt 或 PyPI 配置。镜像拉取代理由 Docker 配置，构建与应用代理由 `.env` 配置。
- **数据库长期不健康**：查看 `docker compose logs --tail 100 opengauss`；检查 ARM64 环境、密码复杂度、磁盘空间和存储权限。
- **应用退出或反复重启**：查看 `docker compose logs --tail 100 ogmemory`；检查 AGFS、配置和数据卷权限。
- **HTTP 正常但模型失败**：核对 `.env` 中容器可访问的模型地址、宿主机监听地址、防火墙、代理、模型密钥与向量维度。容器中的 `127.0.0.1` 指向容器自身。
- **业务接口返回 401**：核对服务密钥，使用 `OG_TENANT_API_KEY`。
- **业务接口返回 403 或 Agent denied**：确认已执行第 5 节初始化，身份一致且 Agent 属于指定用户。
- **MCP 主机校验失败**：在 `OGMEM_MCP_ALLOWED_HOSTS` 中加入实际访问域名或 IP。
- **写入 completed 但无法召回**：核对实际 marker、身份、索引、模型维度及正文。仅 HTTP 200 不证明写入与召回完成。
- **出现 Python 的 `>>>` 提示符**：这是交互模式。输入 `exit()` 返回终端，然后执行指南中的完整 `docker compose exec ... python3 /opt/ogmem/scripts/...` 命令。
- **端口占用**：改用空闲宿主端口，并同步修改客户端地址；容器内部仍使用 8090/8091。

### 9.2 Docker 无法分配容器网段

默认情况下，Docker 自动为项目分配网段。如果启动时报 `could not find an available, non-overlapping IPv4 address pool`，说明默认池中没有可分配网段。可启用项目级网段配置，无需重建镜像。先检查现有 Docker 网段与主机路由，并与网络管理员确认不会覆盖需要访问的内网地址：

```sh
docker network ls -q | xargs docker network inspect --format '{{.Name}} {{range .IPAM.Config}}{{.Subnet}} {{end}}'
# Linux 主机路由；macOS 可使用 netstat -rn。
ip -4 route show table all
```

确认后，在已有 `.env` 中填写以下两项。此网段仅为示例，必须换成当前环境中无冲突的私有网段；多实例使用不同网段。`COMPOSE_FILE` 使后续普通 `docker compose` 命令自动合并网络配置：

```dotenv
# 按需填写：容器网络网段，须与 Docker 网络、主机路由及需访问的内网网段无重叠。
OGMEM_NETWORK_SUBNET=10.203.78.0/24
# 启用自定义网段时必填；Linux/macOS 使用冒号分隔两个 Compose 文件。
COMPOSE_FILE=compose.yaml:deploy/compose/compose.network.yaml
```

保存后执行 `docker compose config --quiet`，再重试启动。数据库仍通过服务名 `opengauss` 访问，不需要配置固定容器 IP。已运行的实例需要更改网段时，应安排停机并保留数据卷后重建网络。不要为解决此错误直接清理共享服务器上的其他项目网络。

日志可能包含业务内容和供应商错误，分享前检查脱敏。排错优先保留数据，不默认删卷或重复写入。

## Agent 工具与经验记忆

部署模板默认关闭 Agent 记忆抽取。需要启用时，按 [Agent 记忆部署说明](../agent-memory.md) 修改挂载 YAML、重启服务并验证配置和 Trace 抽取结果。
