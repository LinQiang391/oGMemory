# OpenCode user summary filtering fix implementation plan

## Goal

Preserve normal OpenCode 1.18.6 user messages that carry diff-summary metadata,
while continuing to exclude generated assistant compaction summaries.

## Tasks

1. Add focused normalization tests in
   `opencode-plugin/tests/message-sync.test.mjs` using the real OpenCode 1.18.6
   message shapes. Verify that the user regression test fails before the fix.
2. Change generated-message classification in
   `opencode-plugin/lib/message-sync.mjs` so only assistant `summary: true`
   messages are excluded by the summary flag.
3. Run the focused test, full plugin suite, and package check.
4. Run a real OpenCode session in WSL and inspect oGMemory archive roles and
   index rows.
5. Review the complete implementation diff and report any backend-independent
   residual risk.
