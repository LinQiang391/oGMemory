"""文档分块处理器"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TextChunk:
    """文本块数据结构"""

    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """生成唯一ID"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        source_hash = hashlib.sha256(self.source.encode()).hexdigest()[:8]
        return f"{source_hash}_{content_hash}"


class DocumentProcessor:
    """文档处理器 - 负责读取和分块文档"""

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_structure: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure

    def load_document(self, file_path: str) -> str:
        """加载文档内容"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path.read_text(encoding="utf-8")

    def process(self, text: str, source: str = "") -> List[TextChunk]:
        """处理文本并生成块"""
        if self.preserve_structure:
            return self._chunk_by_structure(text, source)
        return self._chunk_by_size(text, source)

    def _chunk_by_structure(self, text: str, source: str) -> List[TextChunk]:
        """按文档结构（标题）分块"""
        chunks = []

        lines = text.split("\n")
        sections = self._extract_sections(lines)

        for section in sections:
            section_text = "\n".join(section["lines"]).strip()
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        content=section_text,
                        source=source,
                        chunk_id="",
                        metadata={
                            "heading": section["heading"],
                            "level": section["level"],
                            "start_line": section["start"],
                            "end_line": section["end"],
                        },
                    )
                )
            else:
                sub_chunks = self._split_large_text(
                    section_text, source, section["heading"], section["level"]
                )
                chunks.extend(sub_chunks)

        return chunks

    def _extract_sections(self, lines: List[str]) -> List[Dict]:
        """提取文档章节"""
        sections = []
        current_section = {"heading": "", "level": 0, "start": 0, "lines": []}

        for i, line in enumerate(lines):
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match:
                if current_section["lines"]:
                    current_section["end"] = i - 1
                    sections.append(current_section.copy())

                current_section = {
                    "heading": heading_match.group(2).strip(),
                    "level": len(heading_match.group(1)),
                    "start": i,
                    "lines": [line],
                }
            else:
                current_section["lines"].append(line)

        if current_section["lines"]:
            current_section["end"] = len(lines) - 1
            sections.append(current_section)

        return sections

    def _split_large_text(
        self, text: str, source: str, heading: str, level: int
    ) -> List[TextChunk]:
        """分割大文本"""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        source=source,
                        chunk_id="",
                        metadata={
                            "heading": heading,
                            "level": level,
                            "is_subchunk": True,
                        },
                    )
                )

                overlap = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap
                current_length = sum(len(" ".join(current_chunk)))

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    source=source,
                    chunk_id="",
                    metadata={"heading": heading, "level": level, "is_subchunk": True},
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """分割成句子"""
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """获取重叠句子"""
        if self.chunk_overlap <= 0:
            return []

        overlap_text = " ".join(sentences)
        if len(overlap_text) <= self.chunk_overlap:
            return sentences

        result = []
        total_length = 0

        for sentence in reversed(sentences):
            if total_length + len(sentence) > self.chunk_overlap:
                break
            result.insert(0, sentence)
            total_length += len(sentence)

        return result

    def _chunk_by_size(self, text: str, source: str) -> List[TextChunk]:
        """按固定大小分块"""
        chunks = []
        length = len(text)

        for i in range(0, length, self.chunk_size - self.chunk_overlap):
            end = min(i + self.chunk_size, length)
            chunk_text = text[i:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        source=source,
                        chunk_id="",
                        metadata={"start": i, "end": end, "method": "size_based"},
                    )
                )

        return chunks
