# Memory Recall

Automatically retrieve relevant memories from your openGauss-based memory store. This skill runs in a forked subagent context, keeping all intermediate search operations isolated from the main conversation.

## When to Use

Claude will automatically invoke this skill when it determines that your question could benefit from historical context. You can also manually trigger it with `/memory-recall <query>`.

## What It Does

This skill performs a three-layer progressive disclosure:

1. **L1: Search** - Searches for relevant memories using semantic vector search
2. **L2: Expand** - Retrieves full markdown sections for promising results
3. **L3: Summarize** - Curates and summarizes the findings

Only the final curated summary is returned to the main conversation, keeping your context window clean.

## Progressive Disclosure

The skill autonomously evaluates search results and decides whether to drill deeper:

```bash
# L1: Search for relevant memories
$ python3 -m ogmemory search "your query" --top-k 5

# L2: Expand promising results (full context)
$ python3 -m ogmemory expand <chunk_id>

# L3: Return curated summary to main agent
```

## Examples

**Automatic invocation:**
```
User: What did we decide about the caching strategy?
[Claude automatically invokes memory-recall skill]
→ Finds relevant discussions about caching
→ Returns summary of decisions and implementations
```

**Manual invocation:**
```
/memory-recall database configuration
→ Searches and retrieves all context about DB setup
```

## Implementation Notes

- Runs in `context: fork` subagent
- Uses openGauss vector similarity search
- Filters by relevance score
- Expands top results for full context
- Returns concise summary to main conversation
