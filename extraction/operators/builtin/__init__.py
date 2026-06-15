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
==================  =========  ===========================================
"""

from extraction.operators.builtin.document_extractor import DocumentExtractor
from extraction.operators.builtin.face_detection import FaceDetectionExtractor
from extraction.operators.builtin.image_caption import ImageCaptionExtractor
from extraction.operators.builtin.object_detection import ObjectDetectionExtractor
from extraction.operators.builtin.ocr_extractor import OCRExtractor
from extraction.operators.builtin.pose_detection import PoseDetectionExtractor

__all__ = [
    "DocumentExtractor",
    "FaceDetectionExtractor",
    "ImageCaptionExtractor",
    "ObjectDetectionExtractor",
    "OCRExtractor",
    "PoseDetectionExtractor",
]


def register_all_defaults(registry) -> int:
    """Register every built-in operator on the given registry.

    Returns the number of operators registered. Useful in CLI setup
    and tests; the production path is to load from YAML instead.
    """
    from extraction.operators.builtin.face_detection import FaceDetectionExtractor
    from extraction.operators.builtin.object_detection import ObjectDetectionExtractor
    from extraction.operators.builtin.pose_detection import PoseDetectionExtractor
    from extraction.operators.builtin.image_caption import ImageCaptionExtractor
    from extraction.operators.builtin.ocr_extractor import OCRExtractor
    from extraction.operators.builtin.document_extractor import DocumentExtractor
    from extraction.operators import PromptExtractor

    defaults = [
        PromptExtractor(),
        ImageCaptionExtractor(),
        ObjectDetectionExtractor(),
        FaceDetectionExtractor(),
        PoseDetectionExtractor(),
        OCRExtractor(),
        DocumentExtractor(),
    ]
    for op in defaults:
        registry.register(op, override=True)
    return len(defaults)
