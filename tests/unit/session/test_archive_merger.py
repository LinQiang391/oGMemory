from session.archive_merger import ArchiveMerger
from session.models import ArchiveEntry


def _archive(
    archive_id: str,
    created_at: str,
    overview: str,
    abstract: str,
    messages=None,
):
    return ArchiveEntry(
        archive_id=archive_id,
        session_id="sess1",
        overview=overview,
        abstract=abstract,
        messages=messages or [{"role": "user", "content": f"message {archive_id}"}],
        created_at=created_at,
        metadata={"archive_id": archive_id},
    )


def test_merge_preserves_newest_archive_messages_in_full():
    old = _archive("old", "2026-05-01T00:00:00+00:00", "old overview", "old abstract")
    new = _archive("new", "2026-05-02T00:00:00+00:00", "new overview", "new abstract")

    merged = ArchiveMerger(max_messages=10).merge([old, new])

    assert "new overview" in merged.overview
    assert "old overview" in merged.overview
    assert merged.messages[-1]["content"] == "message new"
    assert merged.metadata["source_archive_ids"] == ["old", "new"]
    assert merged.metadata["latest_archive_id"] == "new"


def test_merge_limits_old_messages_but_keeps_latest_archive_complete():
    old_messages = [{"role": "user", "content": f"old {i}"} for i in range(10)]
    new_messages = [{"role": "user", "content": "new important"}]
    old = _archive(
        "old",
        "2026-05-01T00:00:00+00:00",
        "old overview",
        "old abstract",
        old_messages,
    )
    new = _archive(
        "new",
        "2026-05-02T00:00:00+00:00",
        "new overview",
        "new abstract",
        new_messages,
    )

    merged = ArchiveMerger(max_messages=3).merge([old, new])

    assert merged.messages[-1]["content"] == "new important"
    assert len(merged.messages) == 3
    assert any(msg["content"] == "old 9" for msg in merged.messages)


def test_merge_abstract_keeps_all_source_abstracts():
    old = _archive(
        "old",
        "2026-05-01T00:00:00+00:00",
        "old overview",
        "old abstract " + ("x" * 80),
    )
    new = _archive(
        "new",
        "2026-05-02T00:00:00+00:00",
        "new overview",
        "new abstract " + ("y" * 80),
    )

    merged = ArchiveMerger().merge([old, new])

    assert old.abstract in merged.abstract
    assert new.abstract in merged.abstract


def test_merge_empty_archives_returns_empty_payload():
    merged = ArchiveMerger().merge([])

    assert merged.overview == ""
    assert merged.abstract == ""
    assert merged.messages == []
    assert merged.metadata["source_archive_ids"] == []
