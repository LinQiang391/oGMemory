# M3 安全续跑

- 默认不覆盖 profile 的音频配置。`--with-audio` / `--no-audio` 显式覆盖
  `episodic_memory.params.no_audio`；ASR 等独立阶段仍由 YAML 控制。
- smart resume 的索引探针核验当前账户的独立 VectorIndex 记录及向量，不要求
  ExtractedNode 存储 embedding。本地 episodic/ASR 文本变化会使索引重新执行。
- QA 只有当前 annotation 中的问题全部有非空、无错误的答案时才跳过。
  `qa.params.resume=true` 会重试失败/缺失题目；相同 `(video_id, question_id)`
  以最后一条记录为准。恢复成功的 `[Answer]` 内容将说明写入 `warning`，不写 `error`。
- `--reuse-graph --graph-store sql` 使用与正常流水线相同的有效配置：CLI 的
  `--connection-string` 优先，否则使用 YAML 解析出的图 DSN。向量库继续使用独立
  vector 配置。探测或 QA 异常会报错，不再静默回退到昂贵的完整流水线。

## 输入变更与旧工作目录

工作目录中的 `input_provenance.json` 绑定源文件路径及 SHA-256、账户、video_id、
截取时长和预处理/加载参数。trim/preprocess 的阶段清单另行记录实际输出文件的校验值，
只有清单和当前输入/输出一致才跳过。中断后不完整的输出不会被当作完成。

**更换视频、修改 `--seconds` 或 `clip_seconds` 时，使用新的工作目录和新的 video_id。**
例如：

```bash
python -m m3.run_video_pipeline \
  --video /data/example.mp4 --video-id example-60s-v2 \
  --seconds 60 --clip-seconds 15 --work-dir data/m3_runs/example-60s-v2 \
  --pipeline config/m3/pipelines/video_episodic_full.yaml \
  --config deploy/ogmemory.yaml --graph-store sql
```

检测到输入变化会明确报错，而不是只重新切视频后继续使用旧图中的抽取节点。
`--force` 不绕过这项检查，因为图中的 clip ID 与 video_id/clip_id 绑定，
强制重写文件并不能安全清除所有派生图节点和向量。

升级前生成的工作目录如果已有视频/片段但没有来源清单，同样拒绝自动信任。
请保留旧目录用于审计，以新的工作目录和 video_id 重建；不要手工伪造来源清单。
本次不自动删除旧图节点、向量或媒体文件。
