"""Resolve memory subjects and reporting speakers from source evidence."""

import re
import unicodedata
from dataclasses import replace

from core.memory_names import canonical_memory_key
from core.models import CandidateMemory

_HEADER = re.compile(r"^\s*(?:\[(?P<bracket>[^\]\r\n:：]{1,60})\]|(?P<plain>[^:：\r\n]{1,60}))[:：]\s*")
_ROLES = {"user", "human", "assistant", "system", "developer", "agent", "bot",
          "用户", "助手", "系统"}
_USER_ROLES = {"user", "human", "用户"}
_UNTRUSTED_ROLE = "\0"
_FIELDS = {
    "address", "author", "category", "color", "content", "context", "conversation",
    "date", "deadline", "description", "email", "from", "id", "location", "message",
    "metadata", "name", "note", "notes", "occupation", "owner", "participants", "phone", "priority",
    "profession", "project", "role", "scope", "session", "session time", "size", "speaker", "status",
    "subject", "summary", "time", "timestamp", "title", "to", "topic", "url", "version",
    "标题", "作者", "日期", "时间", "地点", "正文", "内容", "摘要", "备注", "主题", "来源", "状态",
    "发件人", "收件人", "参与人", "项目", "版本", "说明", "描述", "类别", "优先级", "截止日期",
}
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b|我|咱|俺|本人", re.IGNORECASE)
# Word boundaries only against ASCII word characters so CJK names match inside CJK text.
_NOT_WORD_BEFORE = r"(?<![A-Za-z0-9_])"
_NOT_WORD_AFTER = r"(?![A-Za-z0-9_])"


def _identity(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _line_speaker(line: str):
    offset = 0
    for _ in range(3):
        match = _HEADER.match(line[offset:])
        if match is None:
            return None
        name = (match.group("bracket") or match.group("plain")).strip()
        key = _identity(name)
        offset += match.end()
        if key in _ROLES:
            continue
        if (key in _FIELDS or line[offset:].startswith(("//", "\\"))
                or len(name.split()) > 5 or not any(char.isalpha() for char in name)
                or any(not (char.isalnum() or char in " _-.'’") for char in name)):
            return None
        return key, name, bool(match.group("bracket"))
    return None


def _named_turns(source_text: str, user_id: str):
    headers = []
    has_role_header = False
    offset = 0
    for line in source_text.splitlines(keepends=True):
        match = _HEADER.match(line)
        role = _identity(match.group("bracket") or match.group("plain")) if match else ""
        speaker = _line_speaker(line) if role not in _ROLES or role in _USER_ROLES else None
        if speaker:
            headers.append((offset, *speaker))
        elif role in _ROLES:
            has_role_header = True
            key = _identity(user_id) if role in _USER_ROLES else _UNTRUSTED_ROLE
            headers.append((offset, key, user_id if role in _USER_ROLES else "", False))
        offset += len(line)
    names = {key: name for _, key, name, _ in headers if key != _UNTRUSTED_ROLE}
    explicit = any(bracketed for _, _, _, bracketed in headers)
    if not explicit and len(names) < 2 and not has_role_header:
        return {}, []
    turns = [(start, headers[index + 1][0] if index + 1 < len(headers) else len(source_text), key)
             for index, (start, key, _, _) in enumerate(headers)]
    return names, turns


def _quote_speakers(quote: str, source_text: str, turns: list[tuple]) -> set[str]:
    if not quote or not quote.strip():
        return set()
    pattern = r"\s+".join(re.escape(part) for part in quote.strip().split())
    speakers = set()
    for match in re.finditer(pattern, source_text):
        speakers.update(key for start, end, key in turns
                        if match.start() < end and match.end() > start)
    return speakers


def _mentions_name(text: str, name: str) -> bool:
    return bool(re.search(_NOT_WORD_BEFORE + re.escape(name) + _NOT_WORD_AFTER, text, re.IGNORECASE))


def _bind_preference(candidate: CandidateMemory, subject: str, speaker: str, user_id: str) -> CandidateMemory:
    own = _identity(subject) == _identity(user_id)
    # A third party's preference is identified by subject and topic; the user's keeps the plain topic key.
    return replace(candidate, who=subject, attributed_speaker=speaker,
                   attribution_basis="self_named" if own else "other_named",
                   scope_ref=candidate.scope_ref or ("" if own or candidate.target_uri else subject))


def _quote_body(quote: str, names: dict[str, str]) -> str:
    for _ in range(3):
        match = _HEADER.match(quote)
        if match is None:
            break
        key = _identity(match.group("bracket") or match.group("plain"))
        if key not in names and key not in _ROLES:
            break
        quote = quote[match.end():]
    return quote


def bind_source_attribution(
    candidate: CandidateMemory, source_text: str, user_id: str,
) -> CandidateMemory:
    """Resolve profile/preference ownership from source evidence and named turns.

    An ambiguous named transcript raises ValueError so extraction can repair its
    attribution using the source before preparing a write.
    """
    if candidate.category not in {"profile", "preference"}:
        return candidate
    names, turns = _named_turns(source_text, user_id)
    quote = candidate.evidence_quote or ""
    speakers = _quote_speakers(quote, source_text, turns)
    if _UNTRUSTED_ROLE in speakers:
        raise ValueError("Named-transcript attribution cannot use an assistant or system turn as evidence.")
    if not names:
        if (candidate.category == "preference" and candidate.who and candidate.who.strip()
                and _identity(candidate.who) not in _ROLES
                and _identity(candidate.who) != _identity(user_id)
                and not candidate.scope_ref and not candidate.target_uri):
            return replace(candidate, scope_ref=candidate.who.strip())
        return candidate

    subjects = set()
    if len(speakers) == 1:
        body = _quote_body(quote, names)
        # The extracted subject may be a person mentioned by another speaker.
        for hint in (candidate.who, candidate.attributed_speaker):
            named = _line_speaker(f"{hint}: ") if hint else None
            if named and _mentions_name(body, names.get(named[0], named[1])):
                names.setdefault(named[0], named[1])
                subjects = {named[0]}
                break
        if not subjects and _FIRST_PERSON.search(body):
            subjects = speakers
        body_subjects = {key for key, name in names.items()
                         if re.match(r"^\s*" + re.escape(name) + _NOT_WORD_AFTER,
                                     candidate.content, re.IGNORECASE)}
        if body_subjects and subjects and body_subjects != subjects:
            subjects = set()
    elif not speakers:
        subjects = {_identity(value) for value in (candidate.who, candidate.attributed_speaker)
                    if value and _identity(value) in names}
    if len(subjects) != 1:
        raise ValueError(
            "Named-transcript attribution needs one supported subject. Identify the person, group "
            "or object the fact describes in who. Provide a continuous supporting source quote "
            "from one reporting speaker; the subject can differ from that speaker. "
            f"Source reporting speakers: {', '.join(names.values())}."
        )

    key = next(iter(subjects))
    subject = names[key]
    speaker = names[next(iter(speakers))] if len(speakers) == 1 else subject
    if candidate.category == "preference":
        return _bind_preference(candidate, subject, speaker, user_id)
    if key == _identity(user_id) and (not speakers or speakers == {key}):
        return replace(candidate, attributed_speaker=speaker, who=subject,
                       attribution_basis="self_named")
    return replace(candidate, category="entity", owner_scope="user",
                   routing_key=canonical_memory_key(subject), object_name=subject,
                   who=subject, attributed_speaker=speaker, attribution_basis="other_named",
                   target_uri="", expected_version=None)
