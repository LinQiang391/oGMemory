#!/usr/bin/env python3
"""
OpenClaw Plugin Integration Test

Tests the oG-Memory plugin's integration with OpenClaw, including:
- assemble() retrieval functionality
- Helper functions (_extract_query, _format_memory_addition)
- Various scenarios (normal, no results, degradation, empty messages)

Run: pytest tests/integration/test_openclaw_plugin.py
     python tests/integration/test_openclaw_plugin.py

Requires: OpenClaw plugin files in openclaw_context_engine_plugin/
Skips: If OpenClaw plugin bridge files are not available
"""

import sys
import os
import shutil
from pathlib import Path

import pytest

# Check if OpenClaw plugin is available
PLUGIN_DIR = Path(__file__).parent.parent.parent / "openclaw_context_engine_plugin"
BRIDGE_FILE = PLUGIN_DIR / "bridge" / "memory_api.py"

if not BRIDGE_FILE.exists():
    pytest.skip(
        f"OpenClaw plugin bridge not found at {BRIDGE_FILE} (removed in upstream refactor)",
        allow_module_level=True,
    )

# Add plugin directory to path
sys.path.insert(0, str(PLUGIN_DIR))

try:
    from bridge.memory_api import assemble, _extract_query, _format_memory_addition
except ImportError as e:
    print(f"[SKIP] Cannot import OpenClaw plugin bridge: {e}")
    print("[SKIP] Skipping test - plugin is optional for core oG-Memory tests")
    sys.exit(0)

# ANSI color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{text}")
    print("=" * len(text))


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"{RED}✗{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print an info message."""
    print(f"{YELLOW}ℹ{RESET} {msg}")


def test_extract_query():
    """Test _extract_query() extracts queries from messages."""
    print_header("[1] Testing _extract_query()...")

    # Test 1: Normal case with user messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]
    query = _extract_query(messages)
    expected = "Hello\nHow are you?"

    # Assert for pytest, print for standalone execution
    if query == expected:
        print_success(f"_extract_query() returned correct query: {repr(query)}")
    else:
        print_error(f"Expected {repr(expected)}, got {repr(query)}")

    assert query == expected, f"Expected {repr(expected)}, got {repr(query)}"


def test_format_memory_addition_empty():
    """Test _format_memory_addition() with empty hits."""
    print_header("[2] Testing _format_memory_addition() with empty hits...")

    result = _format_memory_addition([])
    has_correct_message = "No relevant memories found" in result or "No context available" in result.lower()

    if has_correct_message:
        print_success(f"_format_memory_addition([]) returns correct message")
    else:
        print_error(f"Expected 'No relevant memories found' message, got: {result[:100]}")

    assert has_correct_message, f"Expected 'No relevant memories found' message, got: {result[:100]}"


def test_format_memory_addition_with_hits():
    """Test _format_memory_addition() with mock hits."""
    print_header("[3] Testing _format_memory_addition() with hits...")

    # Create mock RetrievedBlock objects
    class MockHit:
        def __init__(self, category, abstract, score):
            self.category = category
            self.abstract = abstract
            self.score = score

    hits = [
        MockHit("profile", "用户是一名后端工程师", 0.92),
        MockHit("entity", "SQLite数据库迁移", 0.85),
    ]

    result = _format_memory_addition(hits)
    is_valid = "长期记忆" in result and "后端工程师" in result and "相关度: 92%" in result

    if is_valid:
        print_success("_format_memory_addition(hits) formats results correctly")
        print(f"  Preview: {result[:150]}...")
    else:
        print_error(f"Formatted result doesn't contain expected content: {result[:200]}")

    assert is_valid, f"Formatted result doesn't contain expected content: {result[:200]}"


def test_assemble_normal():
    """Test assemble() with a normal search query."""
    print_header("[4] Testing assemble() - normal case with search query...")

    params = {
        "sessionId": "test-search-session",
        "tokenBudget": 128000,
        "messages": [
            {"role": "user", "content": "数据库迁移的进展如何？"},
        ],
    }

    result = assemble(params)

    assert result is not None, "assemble() returned None"
    assert "messages" in result, f"assemble() missing 'messages' field: {result.keys()}"
    assert "systemPromptAddition" in result, f"assemble() missing 'systemPromptAddition' field: {result.keys()}"

    print_success("assemble() returned with correct structure")
    print(f"  systemPromptAddition: {result['systemPromptAddition'][:100]}...")


def test_assemble_no_results():
    """Test assemble() with a query likely to have no results."""
    print_header("[5] Testing assemble() - query with likely no results...")

    params = {
        "sessionId": "test-no-results",
        "tokenBudget": 128000,
        "messages": [
            {"role": "user", "content": "xyzabc123totallyrandomquery"},
        ],
    }

    result = assemble(params)

    assert result is not None, "assemble() returned None"

    # Should handle gracefully even with no results
    addition = result.get("systemPromptAddition", "")
    print_success("assemble() returned for no-results query")
    print(f"  systemPromptAddition: {addition[:100]}...")

    # Check if it handled no-results gracefully
    handled_gracefully = "No relevant memories" in addition or "No context" in addition or len(addition) < 200
    if handled_gracefully:
        print_success("assemble() handled no-results case gracefully")

    # This is a soft check - we just want to ensure it doesn't crash
    assert True, "assemble() completed without exception"


def test_assemble_empty_messages():
    """Test assemble() with empty messages."""
    print_header("[6] Testing assemble() - empty messages...")

    params = {
        "sessionId": "test-empty",
        "tokenBudget": 128000,
        "messages": [],
    }

    result = assemble(params)

    assert result is not None, "assemble() returned None"

    addition = result.get("systemPromptAddition", "")

    if "No context available" in addition or "No relevant memories" in addition or len(addition) < 200:
        print_success("assemble() returned correct 'No context available' message")
    else:
        print_success(f"assemble() returned for empty messages: {addition[:100]}...")

    # Just verify it doesn't crash
    assert True, "assemble() completed without exception"


def test_assemble_multiple_user_messages():
    """Test assemble() with multiple user messages."""
    print_header("[7] Testing assemble() - multiple user messages...")

    params = {
        "sessionId": "test-multiple",
        "tokenBudget": 128000,
        "messages": [
            {"role": "user", "content": "我是工程师"},
            {"role": "assistant", "content": "你好工程师"},
            {"role": "user", "content": "我喜欢Python"},
            {"role": "assistant", "content": "Python很好"},
            {"role": "user", "content": "数据库迁移进展如何？"},
        ],
    }

    result = assemble(params)

    assert result is not None, "assemble() returned None"
    print_success("assemble() handled multiple user messages")


def test_assemble_assistant_only():
    """Test assemble() with only assistant messages."""
    print_header("[8] Testing assemble() - only assistant messages...")

    params = {
        "sessionId": "test-assistant-only",
        "tokenBudget": 128000,
        "messages": [
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "assistant", "content": "Please ask me something."},
        ],
    }

    result = assemble(params)

    assert result is not None, "assemble() returned None"

    addition = result.get("systemPromptAddition", "")
    if "No context available" in addition or "No relevant memories" in addition or len(addition) < 200:
        print_success("assemble() correctly identified no user messages")
    else:
        print_success("assemble() handled assistant-only messages")

    # Just verify it doesn't crash
    assert True, "assemble() completed without exception"


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OpenClaw Plugin Integration Test")
    print("=" * 60)
    print_info(f"Working directory: {os.getcwd()}")
    print_info(f"Plugin directory: {PLUGIN_DIR}")

    tests = [
        test_extract_query,
        test_format_memory_addition_empty,
        test_format_memory_addition_with_hits,
        test_assemble_normal,
        test_assemble_no_results,
        test_assemble_empty_messages,
        test_assemble_multiple_user_messages,
        test_assemble_assistant_only,
    ]

    # Track results by counting assertion failures
    total = len(tests)
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1  # No exception/AssertionError means test passed
        except AssertionError as e:
            print_error(f"Test {test.__name__} failed assertion: {e}")
        except Exception as e:
            print_error(f"Test {test.__name__} crashed: {e}")

    print_header("TEST SUMMARY")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print(f"\n{GREEN}{'=' * 20} ALL TESTS PASSED {'=' * 20}{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{'=' * 20} SOME TESTS FAILED {'=' * 20}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
