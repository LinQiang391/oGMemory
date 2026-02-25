"""配置管理模块"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatabaseConfig:
    """数据库配置"""

    host: str = "localhost"
    port: int = 5432
    database: str = "memory_db"
    user: str = "postgres"
    password: str = ""
    table_name: str = "vectors"


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class ChunkingConfig:
    """分块配置"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_structure: bool = True


@dataclass
class Config:
    """主配置类"""

    database: DatabaseConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig

    def __init__(
        self,
        database: Optional[DatabaseConfig] = None,
        embedding: Optional[EmbeddingConfig] = None,
        chunking: Optional[ChunkingConfig] = None,
    ):
        self.database = database or DatabaseConfig()
        self.embedding = embedding or EmbeddingConfig()
        self.chunking = chunking or ChunkingConfig()

        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        if "OG_DB_HOST" in os.environ:
            self.database.host = os.environ["OG_DB_HOST"]
        if "OG_DB_PORT" in os.environ:
            self.database.port = int(os.environ["OG_DB_PORT"])
        if "OG_DB_NAME" in os.environ:
            self.database.database = os.environ["OG_DB_NAME"]
        if "OG_DB_USER" in os.environ:
            self.database.user = os.environ["OG_DB_USER"]
        if "OG_DB_PASSWORD" in os.environ:
            self.database.password = os.environ["OG_DB_PASSWORD"]

        if "OG_EMBEDDING_PROVIDER" in os.environ:
            self.embedding.provider = os.environ["OG_EMBEDDING_PROVIDER"]
        if "OG_EMBEDDING_MODEL" in os.environ:
            self.embedding.model = os.environ["OG_EMBEDDING_MODEL"]
        elif "OPENAI_EMBEDDING_MODEL" in os.environ:
            self.embedding.model = os.environ["OPENAI_EMBEDDING_MODEL"]
        if "OPENAI_API_KEY" in os.environ:
            self.embedding.api_key = os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            self.embedding.base_url = os.environ["OPENAI_BASE_URL"]

        if "OG_CHUNK_SIZE" in os.environ:
            self.chunking.chunk_size = int(os.environ["OG_CHUNK_SIZE"])
        if "OG_CHUNK_OVERLAP" in os.environ:
            self.chunking.chunk_overlap = int(os.environ["OG_CHUNK_OVERLAP"])

    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "database": asdict(self.database),
            "embedding": asdict(self.embedding),
            "chunking": asdict(self.chunking),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Config":
        """从文件加载配置"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls(
            database=DatabaseConfig(**config_dict.get("database", {})),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            chunking=ChunkingConfig(**config_dict.get("chunking", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "database": asdict(self.database),
            "embedding": asdict(self.embedding),
            "chunking": asdict(self.chunking),
        }
