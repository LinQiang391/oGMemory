from session.models import SessionMessage
from session.topic_detector import TopicDetector


class _Embedder:
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            if "debug" in text.lower() or "traceback" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class _ClusterEmbedder:
    def embed_texts(self, texts):
        mapping = {
            "latency p95 dashboard": [1.0, 0.0],
            "tail latency percentile": [0.95, 0.05],
            "billing invoice export": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


class _MalformedEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0], [0.95], [0.0, 1.0]]


def _msg(content: str) -> SessionMessage:
    return SessionMessage(id=content[:8], role="user", content=content)


def test_detect_topic_uses_keyword_fallback_without_embedder():
    detector = TopicDetector()

    topic = detector.detect([_msg("Please debug this failing pytest traceback")])

    assert topic.label == "debugging"
    assert topic.confidence > 0


def test_detect_topic_uses_embedding_cluster_when_embedder_available():
    detector = TopicDetector(embedder=_Embedder())

    topic = detector.detect([
        _msg("Debug the traceback"),
        _msg("Fix this pytest failure"),
        _msg("Review the architecture plan"),
    ])

    assert topic.label == "debugging"
    assert topic.message_count == 2


def test_detect_topic_switch_reports_changed_label():
    detector = TopicDetector()
    first = detector.detect([_msg("Please debug this failing test")])
    second = detector.detect([_msg("Write an implementation plan")], previous=first)

    assert first.label == "debugging"
    assert second.label == "planning"
    assert second.changed is True


def test_detect_topic_clusters_embeddings_before_labeling():
    detector = TopicDetector(embedder=_ClusterEmbedder(), similarity_threshold=0.8)

    topic = detector.detect([
        _msg("latency p95 dashboard"),
        _msg("tail latency percentile"),
        _msg("billing invoice export"),
    ])

    assert topic.label == "latency"
    assert topic.message_count == 2
    assert topic.confidence > 0.5


def test_detect_topic_falls_back_for_mixed_dimension_embeddings():
    detector = TopicDetector(embedder=_MalformedEmbedder(), similarity_threshold=0.8)

    topic = detector.detect([
        _msg("Please debug this traceback"),
        _msg("Fix this failing test"),
        _msg("Write an implementation plan"),
    ])

    assert topic.label == "debugging"
    assert topic.message_count == 2


def test_detect_topic_uses_generic_keyword_label_for_non_code_topics():
    detector = TopicDetector()

    topic = detector.detect([
        _msg("billing invoice export"),
        _msg("billing payment reconciliation"),
    ])

    assert topic.label == "billing"
    assert topic.message_count == 2


def test_detect_topic_keeps_general_for_single_generic_message():
    detector = TopicDetector()

    topic = detector.detect([_msg("status update invoice export")])

    assert topic.label == "general"
    assert topic.message_count == 1
