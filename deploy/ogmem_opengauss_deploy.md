# oG-Memory + openGauss启动说明

本文档分为两部分：

- 第一部分：启动 openGauss，作为 oG-Memory 对接的向量数据库。
- 第二部分：配置并启动 oG-Memory。

## 使用镜像

| 组件 | 镜像 |
| --- | --- |
| openGauss | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328` |
| oG-Memory | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc1_428` |

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


### 3. 更新 oG-Memory 数据库配置

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

## 第二部分：oG-Memory

### 1. 创建 oG-Memory 数据目录

oG-Memory 的配置文件和 AGFS 记忆数据目录如下：

| 宿主机路径 | 容器路径 | 用途 |
| --- | --- | --- |
| `/your_directory/ogmemory.yaml` | `/etc/ogmem/config.yaml` | oG-Memory 配置文件 |
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

### 3. 启动 oG-Memory 容器

```bash
docker run -d --name ogmem --network host \
  -e OGMEM_CONFIG=/etc/ogmem/config.yaml \
  -v /your_directory/ogmemory.yaml:/etc/ogmem/config.yaml:ro \
  -v /your_directory/ogmem:/data/agfs \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:poc1_428
```

说明：

- `--network host` 会让 oG-Memory 直接使用宿主机网络。
- oG-Memory HTTP API 默认监听宿主机 `8090` 端口。
- 配置文件以只读方式挂载到容器的 `/etc/ogmem/config.yaml`。
- 记忆数据会写入宿主机 `/your_directory/ogmem`。


### 4. 验证 oG-Memory

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
