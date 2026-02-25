"""pytest 配置和共享 fixtures"""

import os
import pytest
from pathlib import Path
import tempfile


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_db: Tests requiring database")
    config.addinivalue_line("markers", "requires_openai: Tests requiring OpenAI API")


def pytest_collection_modifyitems(config, items):
    """无 OPENAI_API_KEY 时跳过 requires_openai 测试"""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_openai = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if "requires_openai" in item.keywords:
            item.add_marker(skip_openai)


@pytest.fixture(scope="session")
def test_config():
    """测试配置 fixture"""
    return {
        "db_host": os.environ.get("TEST_DB_HOST", "localhost"),
        "db_port": int(os.environ.get("TEST_DB_PORT", "5432")),
        "db_name": os.environ.get("TEST_DB_NAME", "test_memory_db"),
        "db_user": os.environ.get("TEST_DB_USER", "postgres"),
        "db_password": os.environ.get("TEST_DB_PASSWORD", "test_password"),
        "db_sslmode": os.environ.get("TEST_DB_SSLMODE") or None,
        "db_gssencmode": os.environ.get("TEST_DB_GSSENCMODE") or None,
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "embedding_model": os.environ.get("OG_EMBEDDING_MODEL") or os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
    }


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_markdown(temp_dir):
    """示例 markdown 文件 fixture"""
    md_file = temp_dir / "test.md"
    md_file.write_text("""# Test Document

## Section 1

This is a test document with multiple sections.

## Section 2

Another section with different content.

### Subsection 2.1

A subsection with more details.
""")
    return md_file


@pytest.fixture
def sample_markdown_large(temp_dir):
    """大型 markdown 文件 fixture（用于测试分块）"""
    md_file = temp_dir / "large_test.md"
    md_file.write_text("""#
""" + "\n".join([f"## Section {i}\n\nContent for section {i}.\n" for i in range(20)]))
    return md_file
