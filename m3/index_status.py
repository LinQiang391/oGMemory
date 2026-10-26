"""Check the dedicated vector store, never require graph-embedded vectors."""
from urllib.parse import quote


def episodic_uri(account_id, video_id, clip_id):
    return (f"m3://accounts/{quote(str(account_id), safe='')}/videos/"
            f"{quote(str(video_id), safe='')}/clips/{quote(str(clip_id), safe='')}/episodic")


def indexed_extraction(index, ext, raw, account_id):
    from core.models import IndexRecord
    metadata = raw.metadata or {}
    uri = episodic_uri(account_id, metadata.get("video_id", ""), metadata.get("clip_id"))
    record = index.get_record(IndexRecord.generate_id(uri, level=2))
    if record is None or record.filters.get("account_id") != account_id:
        return False
    if record.metadata.get("extracted_node_id") != ext.node_id:
        return False
    return bool(index.get_embeddings([uri], {"account_id": account_id}, level=2).get(uri))


def current_text_is_indexed(ctx, stage):
    """A completed vector may still predate refreshed ASR/episodic evidence."""
    from core.models import IndexRecord
    from m3.index_episodic import read_jsonl, build_retrieval_text, _load_asr_segments_by_clip
    path = ctx.artifact("episodic_jsonl", optional=True)
    if path is None or not path.exists() or ctx.vector_index is None:
        return True  # Graph-only reuse has no local manifest to compare.
    asr = _load_asr_segments_by_clip(ctx.artifact("voice_asr", optional=True))
    for rec in read_jsonl(path):
        key = (str(rec.get("video_id", "")), rec.get("clip_id"))
        text = build_retrieval_text(rec,
            asr_segments=rec.get("speech_transcript") or asr.get(key),
            transcript_first=stage.params.get("transcript_first", True))
        if not text:
            continue
        uri = episodic_uri(ctx.account_id, *key)
        saved = ctx.vector_index.get_record(IndexRecord.generate_id(uri, level=2))
        if (saved is None or saved.filters.get("account_id") != ctx.account_id
                or saved.text != text):
            return False
    return True
