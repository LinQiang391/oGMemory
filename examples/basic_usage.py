"""ContextEngine 基础使用示例。

演示如何使用 ContextEngine 为 AI Agent 存储记忆。
注意: dev 分支只包含写入路径，读取功能在 phase1 分支。
"""

import os
from service.api import MemoryWriteAPI, init_write_api
from core.models import RequestContext
from pyagfs import AGFSClient
from fs.agfs_adapter import AGFSContextFS
from providers.llm import OpenAILLM


def main():
    """基础使用示例：写入记忆"""

    # 1. 配置 AGFS 连接
    agfs_url = os.environ.get("AGFS_BASE_URL", "http://localhost:1833")
    client = AGFSClient(api_base_url=agfs_url)
    fs = AGFSContextFS(client=client, mount_prefix="/local")

    # 2. 配置 LLM (用于抽取)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("请设置 OPENAI_API_KEY 环境变量")
        return

    llm = OpenAILLM(api_key=api_key, model="gpt-4")

    # 3. 初始化写 API
    write_api = init_write_api(fs=fs, llm=llm, outbox_store=None)

    # 4. 创建请求上下文
    ctx = RequestContext(
        account_id="demo-account",
        user_id="alice",
        agent_id="assistant",
        session_id="demo-session",
        trace_id="demo-trace"
    )

    # 5. 写入对话记忆
    messages = [
        {"role": "user", "content": "我叫张三，住在北京，是一名后端工程师"},
        {"role": "assistant", "content": "你好张三！很高兴认识你。"}
    ]

    write_result = write_api.commit_session(messages, ctx)
    print(f"✅ 写入完成:")
    print(f"  - 抽取候选: {write_result['candidates_extracted']} 条")
    print(f"  - 过滤掉: {write_result['candidates_filtered']} 条")
    print(f"  - 实际写入: {write_result['writes_completed']} 条")

    print("\n注意: 读取功能 (search_memory, read_memory) 在 phase1 分支可用")
    print("切换到 phase1 分支获取完整的读写功能:")
    print("  git checkout phase1")


if __name__ == "__main__":
    main()
