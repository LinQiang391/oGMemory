"""配置系统单元测试"""

import pytest
from config import Config, DatabaseConfig, EmbeddingConfig, ChunkingConfig


class TestConfig:
    """配置系统测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = Config()

        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkingConfig)

    def test_database_defaults(self):
        """测试数据库默认配置"""
        config = Config()

        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.database.database == "memory_db"
        assert config.database.user == "postgres"
        assert config.database.password == ""
        assert config.database.table_name == "vectors"

    def test_embedding_defaults(self, monkeypatch):
        """测试嵌入默认配置（无环境变量时的默认值）"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OG_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("OG_EMBEDDING_MODEL", raising=False)
        config = Config()

        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.embedding.api_key is None
        assert config.embedding.base_url is None

    def test_chunking_defaults(self):
        """测试分块默认配置"""
        config = Config()

        assert config.chunking.chunk_size == 1000
        assert config.chunking.chunk_overlap == 200
        assert config.chunking.preserve_structure is True

    def test_config_from_dict(self, monkeypatch):
        """测试从字典创建配置（显式传入值优先，不受环境变量覆盖）"""
        monkeypatch.delenv("OG_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = Config(
            database=DatabaseConfig(host="test-host", port=9999),
            embedding=EmbeddingConfig(model="test-model"),
            chunking=ChunkingConfig(chunk_size=500),
        )

        assert config.database.host == "test-host"
        assert config.database.port == 9999
        assert config.embedding.model == "test-model"
        assert config.chunking.chunk_size == 500

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = Config()
        config_dict = config.to_dict()

        assert "database" in config_dict
        assert "embedding" in config_dict
        assert "chunking" in config_dict
        assert config_dict["database"].get("table_name") == "vectors"
        assert "host" in config_dict["database"]
        assert "model" in config_dict["embedding"]

    def test_save_and_load_from_file(self, temp_dir, monkeypatch):
        """测试保存到文件并从文件加载（round-trip）"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OG_EMBEDDING_MODEL", raising=False)
        path = temp_dir / "config.json"
        config = Config(
            database=DatabaseConfig(host="saved-host", port=9999),
            embedding=EmbeddingConfig(model="saved-model"),
        )
        config.save_to_file(str(path))
        assert path.exists()
        loaded = Config.load_from_file(str(path))
        assert loaded.database.host == "saved-host"
        assert loaded.database.port == 9999
        assert loaded.embedding.model == "saved-model"

    def test_load_from_file_not_found(self):
        """测试从不存在文件加载应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            Config.load_from_file("/nonexistent/path/config.json")
        assert "Config file not found" in str(exc_info.value) or "nonexistent" in str(exc_info.value)

    def test_env_override_simulation(self, monkeypatch):
        """测试环境变量覆盖（模拟）"""
        monkeypatch.setenv("OG_DB_HOST", "env-host")
        monkeypatch.setenv("OG_DB_PORT", "9999")
        monkeypatch.setenv("OG_EMBEDDING_PROVIDER", "test-provider")

        config = Config()

        assert config.database.host == "env-host"
        assert config.database.port == 9999
        assert config.embedding.provider == "test-provider"
