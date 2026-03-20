#!/usr/bin/env python3
"""
Bridge: called by OpenClaw plugin via child_process.
JSON on stdin → oG-Memory write → JSON on stdout.
"""
import sys
import json
import os
import uuid

# Disable proxy for local AGFS requests (prevents 502 errors)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# Add project root to path
project_root = os.environ.get("OG_MEMORY_ROOT", "")
if project_root:
    sys.path.insert(0, project_root)

from core.models import RequestContext, CandidateMemory
from fs.agfs_adapter import AGFSContextFS
from pyagfs import AGFSClient
from service.api import MemoryWriteAPI
from providers.config import ProviderConfig


# Category to owner_scope mapping per CLAUDE.md §1 URI Spec
# profile/preference/entity/event → user_space
# case/pattern/skill → agent_space
CATEGORY_OWNER_SCOPE = {
    "profile": "user",
    "preference": "user",
    "entity": "user",
    "event": "user",
    "case": "agent",
    "pattern": "agent",
    "skill": "agent",
}
DEFAULT_OWNER_SCOPE = "user"


def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        content = input_data.get("content", "")
        category = input_data.get("category", "preference")
        routing_key = input_data.get("routing_key", "")  # Bug 1 fix: read from input
        tags = input_data.get("tags", [])

        if not content:
            raise ValueError("content is required")

        # Initialize AGFS client
        # Use /local backend (returns 500 with "no such file" message for missing files)
        agfs_base_url = os.environ.get("AGFS_BASE_URL", "http://localhost:8080")
        client = AGFSClient(api_base_url=agfs_base_url)
        fs = AGFSContextFS(client, mount_prefix="/local")

        # Initialize LLM provider
        config = ProviderConfig.from_env()
        llm = config.create_llm()

        # Initialize Write API
        api = MemoryWriteAPI(fs=fs, llm=llm)

        # Create RequestContext
        account_id = os.environ.get("OG_MEMORY_ACCOUNT_ID", "default")
        user_id = os.environ.get("OG_MEMORY_USER_ID", "default")
        agent_id = os.environ.get("OG_MEMORY_AGENT_ID", "openclaw")

        ctx = RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        # Bug 2 fix: owner_scope from category mapping
        owner_scope = CATEGORY_OWNER_SCOPE.get(category, DEFAULT_OWNER_SCOPE)

        # Bug 1 fix: Use routing_key from input, default to category
        if not routing_key:
            routing_key = category

        # Bug 3 fix: Extract first sentence for abstract
        first_sentence = content.split("。")[0].split(".")[0].strip()
        abstract = first_sentence[:100] if first_sentence else content[:100]
        if len(content) > 100 and len(abstract) == 100:
            abstract += "..."

        overview = f"## {category.capitalize()}\n\n{content[:500]}"
        if len(content) > 500:
            overview += "..."

        # Create CandidateMemory from input
        candidate = CandidateMemory(
            category=category,
            owner_scope=owner_scope,  # Bug 2 fix: mapped from category
            routing_key=routing_key,  # Bug 1 fix: stable routing_key
            abstract=abstract,  # Bug 3 fix: first sentence extraction
            overview=overview,
            content=content,
            confidence=0.9,
        )

        # Write memory
        result = api.write_memory(candidate, ctx)

        output = {
            "ok": True,
            "message": "Memory written successfully",
            "uri": result.get("target_uri"),
            "action": result.get("action"),
        }
        print(json.dumps(output))

    except Exception as e:
        import traceback
        print(json.dumps({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
