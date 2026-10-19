# oGMemory LoCoMO 测试指南

完成[一键部署](../../deploy/user-quickstart/README_zh.md)后，在同一台 Linux ARM64 服务器上，使用执行部署的用户打开 Bash 终端。部署生成的 `$HOME/ogmemory-runtime.env` 提供实例名、模型配置和服务凭据。

## 1. 下载测试脚本

```bash
mkdir -p "$HOME/ogmemory-locomo"
cd "$HOME/ogmemory-locomo"
curl -fL 'https://raw.gitcode.com/opengauss/oGMemory/raw/dev/deploy/user-quickstart/test-locomo.sh?download=true' \
  -o test-locomo.sh
chmod 700 test-locomo.sh
```

脚本自动下载以下文件，评测程序使用固定版本 `263a8484fa1ba28638b5b64bbc48dad2f8e68963`：

| 文件 | 来源与用途 |
| --- | --- |
| `eval.py` | oGMemory 仓库 `tests/e2e/`，负责写入、问答和统计 |
| `judge_parallel.py` | 同目录，负责模型判分 |
| `locomo_mcp_prompts.py` | 同目录，提供 MCP 写入与检索提示词 |
| `locomo10.json` | [LoCoMO 官方数据](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)，包含全部 10 个 sample |

服务器需要能够访问 GitCode、GitHub 和部署时配置的模型服务。

## 2. 配置 locomo.env

在同一个目录创建测试配置：

```bash
cat > locomo.env <<'EOF'
OGMEMORY_RUNTIME_ENV_FILE="$HOME/ogmemory-runtime.env"
EOF
chmod 600 locomo.env
```

`OGMEMORY_RUNTIME_ENV_FILE` 填写一键部署生成的运行环境文件路径。部署时使用自定义路径的实例，将上面的路径改为该文件的实际路径。

问答和 Judge 默认使用部署时的聊天模型。使用独立 Judge 时，在 `locomo.env` 中添加：

```bash
LOCOMO_JUDGE_BASE_URL="https://your-model-service/v1"
LOCOMO_JUDGE_MODEL="your-judge-model"
LOCOMO_JUDGE_API_KEY="your-judge-api-key"
LOCOMO_JUDGE_API_MODE="chat"
```

`chat` 使用 Chat Completions；`responses` 使用 Responses。测试会产生写入、问答和判分的模型调用费用。

## 3. 执行测试

测试 sample 0 的全部对话和 category 1–4 问题：

```bash
bash ./test-locomo.sh --config ./locomo.env --sample 0
```

`--sample` 接受 `0–9`。测试全部 10 个 sample：

```bash
bash ./test-locomo.sh --config ./locomo.env --all
```

脚本依次完成环境检查、数据下载、对话注入、记忆检索问答、Judge 判分和准确率汇总。每个 sample 使用独立测试身份和测试 Gateway；评测程序运行在已部署的 oGMemory 容器中。

## 4. 观察检查项与进度

启动后，终端逐项显示 `[OK]`：测试配置、部署环境、身份与模型配置、命令依赖、Docker 服务、容器、网络、Python 依赖和 oGMemory HTTP 健康状态。数据下载后显示各 sample 的 session 与题目数量；每个 sample 的测试 Gateway 就绪后也显示 `[OK]`。

注入时，`[TEXT BEGIN]` 到 `[TEXT END]` 展示当前提交的完整对话文本，包括对话时间与说话人。每个 session 完成写入、提取和索引后，进度才增加。

问答阶段展示 `[QUESTION]` 和完成题数；Judge 阶段逐题展示 `CORRECT` 或 `WRONG`。各阶段显示进度条、已完成数量、百分比、耗时和预计剩余时间。等待模型或索引时，每 15 秒输出一次 `[WAIT]`。

输出样式如下，数量与耗时以当前运行数据为准：

```text
[OK] Docker 服务
[OK] 容器运行：demo_ogmemory
[OK] Sample 0 测试 Gateway 就绪
[RUN] ingest [....................] 0/2 0% elapsed=0s ETA=-- | ...session_1 写入
[TEXT BEGIN] ...session_1
[group chat conversation: ...]
Alice: My review day is Tuesday.
[TEXT END]
[WAIT] ingest [....................] 0/2 0% elapsed=15s ETA=-- | ...session_1 等待提取与索引
[OK] ingest [##########..........] 1/2 50% elapsed=20s ETA=20s | ...session_1 写入、提取、索引完成
[QUESTION] What day is the review?
[OK] qa [####################] 1/1 100% elapsed=8s ETA=0s | Q0 检索与回答
[OK] judge [####################] 1/1 100% elapsed=2s ETA=0s | Q0 CORRECT
[OK] 题目覆盖：1/1
[OK] Sample 进度：1/1
```

`[FAIL]` 表示运行中断，终端保留错误信息；已生成结果与日志保存在本次运行目录。再次执行命令会创建新的测试身份和结果目录。

## 5. 查看结果

终端打印结果目录 `$STACK_DIR/locomo/<运行ID>/results/`。

| 文件 | 内容 |
| --- | --- |
| `summary.json` | sample 编号、总题数、正确题数与整体准确率 |
| `sample-N/qa.csv` | 问题、标准答案、Agent 回答与宿主 Token 用量 |
| `sample-N/qa_judged.csv` | 逐题判分与理由 |
| `sample-N/coverage.txt`、`stat.txt` | 题目覆盖、总体与分类型准确率 |
| `sample-N/ingest.json`、`idle.json` | 写入结果与索引完成状态 |
| `sample-N/api_logs/`、`gateway.log` | 请求记录与测试 Gateway 日志 |

运行目录还保存 `run.log`、评测程序版本 `source-commit.txt` 和数据校验值 `dataset.sha256`。整体准确率按所有完成 sample 的正确题数除以总评分题数计算。
