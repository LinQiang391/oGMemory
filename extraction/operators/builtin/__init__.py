"""Built-in multimodal extraction operators.

Each operator wraps a ``CVBackend`` (or pure-Python logic for text-only
operators) and produces a uniform ``ExtractionResult`` that becomes an
``ExtractedNode`` in the heterogeneous memory graph.

Available operators:

==================  =========  ===========================================
Operator Type       Raw Data   Backend default
==================  =========  ===========================================
``image_caption``   IMAGE      Hash-based stub (configure ``backend=``)
``object_detection`` IMAGE     MockBackend (configure ``backend=``)
``face_detection``   IMAGE     MockBackend (configure ``backend=``)
``pose_detection``   IMAGE     MockBackend (configure ``backend=``)
``ocr``              IMAGE     MockBackend (configure ``backend=``)
``document_structure`` DOCUMENT  None (pure-Python)
``video_frame_extractor`` VIDEO  OpenCV (no model)
``audio_extractor``      VIDEO  ffmpeg (no model)
``voice_segmentation``  AUDIO    RMS-energy VAD (no model)
``voice_asr``           AUDIO    faster-whisper (or openai_http / mock)
``speaker_diarization`` AUDIO    pyannote.audio 3.1 (or mock)
``speaker_embedding``   AUDIO    speechbrain ECAPA-TDNN (or mock)
==================  =========  ===========================================
"""

from extraction.operators.builtin.audio_extractor import AudioExtractor
from extraction.operators.builtin.document_extractor import DocumentExtractor
from extraction.operators.builtin.episodic_memory_generator import EpisodicMemoryGenerator
from extraction.operators.builtin.face_detection import FaceDetectionExtractor
from extraction.operators.builtin.image_caption import ImageCaptionExtractor
from extraction.operators.builtin.object_detection import ObjectDetectionExtractor
from extraction.operators.builtin.ocr_extractor import OCRExtractor
from extraction.operators.builtin.pose_detection import PoseDetectionExtractor
from extraction.operators.builtin.video_frame_extractor import VideoFrameExtractor
from extraction.operators.builtin.voice_segmentation import VoiceSegmentationExtractor

# Voice ASR (Whisper), speaker diarization (pyannote), and speaker
# embedding (speechbrain) are imported lazily because their default
# backends are heavy optional deps. Tests that don't have them
# installed still need ``import extraction.operators.builtin`` to
# succeed.
try:
    from extraction.operators.builtin.voice_asr import VoiceASRExtractor
except ImportError:  # pragma: no cover
    VoiceASRExtractor = None  # type: ignore[assignment]

try:
    from extraction.operators.builtin.speaker_diarization import (
        SpeakerDiarizationExtractor,
    )
except ImportError:  # pragma: no cover
    SpeakerDiarizationExtractor = None  # type: ignore[assignment]

try:
    from extraction.operators.builtin.speaker_embedding import (
        SpeakerEmbeddingExtractor,
    )
except ImportError:  # pragma: no cover
    SpeakerEmbeddingExtractor = None  # type: ignore[assignment]

__all__ = [
    "AudioExtractor",
    "DocumentExtractor",
    "EpisodicMemoryGenerator",
    "FaceDetectionExtractor",
    "ImageCaptionExtractor",
    "ObjectDetectionExtractor",
    "OCRExtractor",
    "PoseDetectionExtractor",
    "VideoFrameExtractor",
    "VoiceSegmentationExtractor",
    "VoiceASRExtractor",
    "SpeakerDiarizationExtractor",
    "SpeakerEmbeddingExtractor",
]


def register_all_defaults(registry) -> int:
    """Register every built-in operator on the given registry.

    Returns the number of operators registered. Useful in CLI setup
    and tests; the production path is to load from YAML instead.
    """
    from extraction.operators import PromptExtractor

    defaults = [
        PromptExtractor(),
        ImageCaptionExtractor(),
        ObjectDetectionExtractor(),
        FaceDetectionExtractor(),
        PoseDetectionExtractor(),
        OCRExtractor(),
        DocumentExtractor(),
        VideoFrameExtractor(),
        AudioExtractor(),
        VoiceSegmentationExtractor(),
        # EpisodicMemoryGenerator needs an LLM injected at construction,
        # so it's NOT registered as a default — drivers instantiate
        # it directly with the configured LLM backend.
        # The three voice-* operators are also opt-in: their default
        # backends (faster-whisper, pyannote, speechbrain) are heavy
        # optional deps. We only register when each import succeeded.
        VoiceASRExtractor(),
        SpeakerDiarizationExtractor(),
        SpeakerEmbeddingExtractor(),
    ]
    for op in defaults:
        if op is None:
            continue
        registry.register(op, override=True)
    return len(defaults)
