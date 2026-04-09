"""向量嵌入接口"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBase(ABC):
    """嵌入模型基类"""

    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本编码为向量"""
        pass

    @abstractmethod
    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        pass


class EmbeddingFactory:
    """嵌入模型工厂"""

    _providers = {}

    @classmethod
    def register(cls, name: str, provider_class):
        """注册嵌入提供者"""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, name: str, **kwargs) -> EmbeddingBase:
        """创建嵌入实例"""
        if name not in cls._providers:
            raise ValueError(f"Unknown embedding provider: {name}")
        return cls._providers[name](**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有可用提供者"""
        return list(cls._providers.keys())
