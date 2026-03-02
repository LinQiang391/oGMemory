"""文档处理器单元测试"""

import pytest
from core.document_processor import DocumentProcessor, TextChunk


class TestDocumentProcessor:
    """文档处理器测试"""

    def test_load_markdown_file(self, sample_markdown):
        """测试加载 markdown 文件"""
        processor = DocumentProcessor()
        text = processor.load_document(str(sample_markdown))

        assert "# Test Document" in text
        assert "## Section 1" in text
        assert len(text) > 0

    def test_process_preserve_structure(self, sample_markdown):
        """测试保留结构的分块"""
        processor = DocumentProcessor(preserve_structure=True)
        chunks = processor.process(
            sample_markdown.read_text(), source=str(sample_markdown)
        )

        assert len(chunks) > 0
        assert any("Section 1" in c.content for c in chunks)
        assert all(c.metadata.get("heading") for c in chunks)

    def test_process_fixed_size(self, sample_markdown_large):
        """测试固定大小分块"""
        processor = DocumentProcessor(
            chunk_size=100, chunk_overlap=20, preserve_structure=False
        )
        chunks = processor.process(
            sample_markdown_large.read_text(), source=str(sample_markdown_large)
        )

        assert len(chunks) > 0
        assert all(len(c.content) <= 120 for c in chunks)

    def test_chunk_id_uniqueness(self, sample_markdown):
        """测试 chunk ID 唯一性"""
        processor = DocumentProcessor()
        chunks = processor.process(
            sample_markdown.read_text(), source=str(sample_markdown)
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_empty_document(self):
        """测试空文档处理"""
        processor = DocumentProcessor()
        chunks = processor.process("", source="test.md")

        assert len(chunks) == 0

    def test_chunk_metadata(self, sample_markdown):
        """测试分块元数据"""
        processor = DocumentProcessor()
        chunks = processor.process(
            sample_markdown.read_text(), source=str(sample_markdown)
        )

        for chunk in chunks:
            assert chunk.source == str(sample_markdown)
            assert "source" not in chunk.metadata
            assert "chunk_id" not in chunk.metadata

    def test_load_document_file_not_found(self):
        """测试加载不存在文件应抛出 FileNotFoundError"""
        processor = DocumentProcessor()
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.load_document("/nonexistent/file.md")
        assert "not found" in str(exc_info.value).lower() or "file" in str(exc_info.value).lower()

    def test_process_whitespace_only(self):
        """测试仅空白/换行文档返回空列表"""
        processor = DocumentProcessor()
        assert processor.process("   \n\n  \n", source="") == []
        assert processor.process("", source="") == []

    def test_process_single_heading_no_content(self):
        """测试仅有一个标题无正文时仍产生一个 chunk"""
        processor = DocumentProcessor()
        chunks = processor.process("# Only Title\n", source="test.md")
        assert len(chunks) >= 1
        assert chunks[0].content.strip() == "# Only Title"
        assert chunks[0].metadata.get("heading") == "Only Title"

    def test_chunk_by_size_metadata(self, sample_markdown_large):
        """测试按大小分块时 metadata 包含 start/end/method"""
        processor = DocumentProcessor(
            chunk_size=80, chunk_overlap=10, preserve_structure=False
        )
        text = sample_markdown_large.read_text()
        chunks = processor.process(text, source=str(sample_markdown_large))
        assert len(chunks) > 0
        for c in chunks:
            assert "method" in c.metadata
            assert c.metadata["method"] == "size_based"
            assert "start" in c.metadata
            assert "end" in c.metadata
