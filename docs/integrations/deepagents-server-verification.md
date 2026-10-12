# Deep Agents 服务器部署与验证记录

2026-10-12 已在 `123.60.114.206:10008` 部署并完成真实端到端验证。

| 项目 | 实际配置/结果 |
| --- | --- |
| 系统 | openEuler 24.03 LTS，ARM64 |
| Python | 3.13.12 |
| 部署目录 | `/data4/ogmemory-deepagents-20261012` |
| oGMemory 服务 | 已有容器 `ogmemory-ogmemory-1`，`http://127.0.0.1:18110` |
| 存储 | 服务端 AGFS + openGauss 向量索引 |
| 实际模型 | `deepseek-flash`，通过 `ChatOpenAI` 的兼容接口调用 |
| 依赖版本 | Deep Agents 0.7.23、LangChain 1.4.4、LangGraph 1.2.14、langchain-openai 1.7.0 |
| 回归测试 | **72 passed** |
| 依赖检查 | **106 packages，All installed packages are compatible** |
| 真实记忆验证 | 自动写回、同步跨会话召回、异步跨会话召回均通过 |

本次改动是 Agent 客户端适配层，服务端已有所需 REST 接口，因此部署新客户端
并连接现有服务即可。新客户端安装在独立虚拟环境，没有重启或替换已有服务。
服务器根分区已满，部署、缓存和临时文件均放在空间充足的 `/data4`。

## 1. 登录与环境检查

在本机执行，密码在 SSH 提示时输入：

```bash
ssh -p 10008 root@123.60.114.206
```

服务器检查命令：

```bash
uname -sm
cat /etc/os-release
python3 --version
df -h / /data4
docker ps --filter name='^ogmemory-ogmemory-1$' \
  --format '{{.Names}} | {{.Status}} | {{.Ports}}'
curl -fsS http://127.0.0.1:18110/api/v1/health
```

实际健康响应包含：

```json
{"status":"ok","storage_backend":"agfs","llm":"deepseek-flash","agfs":true}
```

## 2. 打包与上传本次修改

本机仓库根目录构建 wheel（本次使用项目本地虚拟环境中的 hatchling）：

```powershell
.\.venv-deepagents\Scripts\python -c "import hatchling.build; print(hatchling.build.build_wheel('.venv-deepagents/artifacts'))"
```

制品为 `context_engine-0.1.0-py3-none-any.whl`。本次部署包还包含：

- `scripts/verify_deepagents.py`：真实模型和真实记忆服务的验证脚本；
- Deep Agents 测试、CLI 回归测试及其源代码/插件配置 fixture；
- 接入文档与交互示例。

两个已生成的上传包位于本机 `.venv-deepagents/`，可用以下命令上传：

```powershell
scp -P 10008 .venv-deepagents/ogmemory-deepagents-20261012.tar.gz root@123.60.114.206:/data4/
scp -P 10008 .venv-deepagents/test-fixtures.tar.gz root@123.60.114.206:/data4/ogmemory-deepagents-test-fixtures.tar.gz
scp -P 10008 .venv-deepagents/configure_connection.py root@123.60.114.206:/data4/ogmemory-deepagents-20261012/
scp -P 10008 pyproject.toml root@123.60.114.206:/data4/ogmemory-deepagents-20261012/
```

本次实际使用 Python Paramiko 完成相同的 SSH/SFTP 上传，密码没有写入制品。
首次部署时，应先完成下面的目录创建与解压，再上传配置辅助脚本和 `pyproject.toml`。
部署目录现在已经存在，复测直接跳到第 6 步。

## 3. 建立独立环境并安装

以下是首次部署的服务器命令；已有目录不必重新创建：

```bash
mkdir -m 700 /data4/ogmemory-deepagents-20261012
cd /data4/ogmemory-deepagents-20261012
tar -xzf /data4/ogmemory-deepagents-20261012.tar.gz -C "$PWD"
tar -xzf /data4/ogmemory-deepagents-test-fixtures.tar.gz -C "$PWD"
mkdir -m 700 cache tmp
export UV_CACHE_DIR="$PWD/cache"
export TMPDIR="$PWD/tmp"
export PYTHONUTF8=1

/root/.local/bin/uv venv --python /root/miniconda3/bin/python3 .venv
/root/.local/bin/uv pip install --python .venv/bin/python \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  'deepagents==0.7.23' 'langchain-openai>=1.1,<2' pytest pytest-asyncio ruff
/root/.local/bin/uv pip install --python .venv/bin/python \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  context_engine-0.1.0-py3-none-any.whl
```

PyPI 直连下载在该服务器反复重试；切换到华为镜像后安装成功。
最终安装了 wheel 的完整依赖，并通过 `uv pip check` 验证一致性。

## 4. 设置专用测试身份与连接

部署目录中的 `configure_connection.py` 会读取已有容器的连接配置，在服务器
内存中取得密钥，然后调用真实管理 API：

```text
POST /api/v1/admin/accounts/compose-account/users
{"user_id":"deepagents-verify-20261012","role":"user"}

POST /api/v1/admin/accounts/compose-account/agents
{"agent_id":"deepagents-verify-agent-20261012","owner_user_id":"deepagents-verify-20261012"}
```

首次配置执行：

```bash
cd /data4/ogmemory-deepagents-20261012
.venv/bin/python configure_connection.py
stat -c '%a %n' connection.env
```

生成的 `connection.env` 权限为 `600`，包含以下环境变量：

```text
OGMEM_URL=http://127.0.0.1:18110
OGMEM_ACCOUNT_ID=compose-account
OGMEM_USER_ID=deepagents-verify-20261012
OGMEM_AGENT_ID=deepagents-verify-agent-20261012
OGMEM_API_KEY=<专用测试用户密钥>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_API_KEY=<复用已有服务的模型密钥>
LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
```

这里展示变量名称和非敏感配置；不要把占位符覆盖到现有文件。
`connection.env` 已生成后直接复用，辅助脚本会拒绝再次覆盖。

## 5. 运行回归测试与依赖检查

```bash
cd /data4/ogmemory-deepagents-20261012
export PYTHONUTF8=1
export TMPDIR="$PWD/tmp"
.venv/bin/python -m pytest \
  tests/unit/deepagents tests/unit/cli/test_plugin_config.py \
  --confcutdir=tests/unit -p no:cacheprovider -q
.venv/bin/ruff check scripts/verify_deepagents.py
/root/.local/bin/uv pip check --python .venv/bin/python
```

实际结果：

```text
72 passed in 5.45s
All checks passed!
Checked 106 packages
All installed packages are compatible
```

## 6. 真实模型、写回和跨会话召回验证

已部署的一键复测命令：

```bash
cd /data4/ogmemory-deepagents-20261012
./run-verification.sh
```

等价的展开命令：

```bash
cd /data4/ogmemory-deepagents-20261012
export PYTHONUTF8=1
export PYTHONWARNINGS=ignore
export TMPDIR="$PWD/tmp"
source ./connection.env
set -o pipefail
.venv/bin/python -u scripts/verify_deepagents.py 2>&1 | tee verification.log
```

脚本执行真实的 `create_deep_agent`，通过 `ChatOpenAI` 调用 DeepSeek，并执行：

1. 新建写入会话，发送随机唯一项目名及发布窗口。
2. 确认 middleware 自动调用 `after_turn`，包含本轮两条消息。
3. 结束写入会话，轮询服务端 `compose`，等待真实抽取和索引。
4. 新建同步读取会话，检查答案及模型请求中的实际记忆上下文。
5. 新建异步读取会话，重复验证。
6. 确认召回上下文未写入对话消息历史，输出 JSON 报告。

本次运行的实际结果：

| 验证 | 输入/输出 |
| --- | --- |
| 写入信息 | 项目 `DA验证75cac515225c` 的发布窗口是 **每周四 18:22** |
| 写入会话 | `da-write-75cac515225c` |
| 写入回复 | `已记住` |
| 自动写回结果 | `{"ok":true,"status":"processing"}` |
| 同步新会话 | `da-read-75cac515225c` → `每周四 18:22` |
| 异步新会话 | `da-async-75cac515225c` → `每周四 18:22` |
| 最终状态 | `passed` |

服务端实际召回证据包含：

```text
## Session Summary
### Facts
项目DA验证75cac515225c的发布窗口是每周四 18:22。

## Retrieved Memories
- [session_summary] ### Facts
项目DA验证75cac515225c的发布窗口是每周四 18:22。
```

`after_turn` 的 `processing` 和第一次 `dispose` 的 `dispose_deferred` 表示后台
仍在处理；脚本确认记忆可检索后才进行新会话验证。

## 7. 查看报告与确认服务状态

```bash
cd /data4/ogmemory-deepagents-20261012
cat deepagents-verification.json
cat verification.log
curl -fsS http://127.0.0.1:18110/api/v1/health
docker ps --filter name='^ogmemory-ogmemory-1$' \
  --format '{{.Names}} | {{.Status}} | {{.Ports}}'
```

完整结果已保存为仓库中的 [验证报告](deepagents-verification-20261012.json)。
测试记忆保留在专用测试用户下；再次运行脚本会产生新的唯一项目名和会话 ID。
验证脚本源代码见 [scripts/verify_deepagents.py](../../scripts/verify_deepagents.py)。
