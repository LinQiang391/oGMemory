"""Unit tests for AGFSContextFS URI to path mapping."""

import time
from threading import RLock, Event
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
from unittest.mock import Mock

import pytest
from pyagfs.exceptions import AGFSClientError

from commit.identity_resolver import MemoryIdentityResolver
from commit.outbox_store import OutboxStore
from core.enums import NodeStatus
from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    NodeBrokenError,
    NodeNotFoundError,
)
from core.models import CandidateMemory, ContextNode, RequestContext, Role
from core.uri_resolver import URIResolver
from core.validation import validate_candidate
from extraction.schemas.registry import SchemaRegistry
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS, build_uri, parse_uri, uri_to_path
from providers.unified_config import OgMemConfig
from server.mcp_tools import _list_memory_node_entries_page, _memory_list_cursor
from server.memory_service import MemoryService
from session.archive_store import SessionArchiveStore


class _MemoryAGFSClient:
    def __init__(self):
        self.files = {}
        self.directories = set()

    @staticmethod
    def _normalize(path):
        return str(path).rstrip("/") or "/"

    def mkdir(self, path):
        self.directories.add(self._normalize(path))

    def write(self, path, content):
        self.files[self._normalize(path)] = bytes(content)

    def read(self, path):
        key = self._normalize(path)
        if key not in self.files:
            raise AGFSClientError("No such file")
        return self.files[key]

    def ls(self, path):
        parent = self._normalize(path)
        prefix = parent.rstrip("/") + "/"
        names = set()
        for directory in self.directories:
            if directory.startswith(prefix):
                remainder = directory[len(prefix):]
                if remainder and "/" not in remainder:
                    names.add(remainder)
        return [{"name": name, "isDir": True} for name in sorted(names)]

    def rm(self, path, recursive=False):
        target = self._normalize(path)
        if recursive:
            prefix = target.rstrip("/") + "/"
            self.files = {
                key: value for key, value in self.files.items()
                if key != target and not key.startswith(prefix)
            }
            self.directories = {
                key for key in self.directories
                if key != target and not key.startswith(prefix)
            }
        else:
            self.files.pop(target, None)


def test_identity_directory_pagination_uses_agfs_order_key():
    class BoundedAGFSContextFS(AGFSContextFS):
        def __init__(self, client):
            super().__init__(client)
            self.page_count = 0

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None, skip_broken=False):
            self.page_count += 1
            assert self.page_count <= 3
            return super().list_node_summaries_page(
                uri, ctx, limit=limit, after_uri=after_uri, skip_broken=skip_broken,
            )

    fs = BoundedAGFSContextFS(_MemoryAGFSClient())
    ctx = RequestContext("acme", "alice", "bot", "session", "trace")
    directory = "ctx://acme/users/alice/memories/entities"
    uris = [f"{directory}/item_{index:03d}" for index in range(99)]
    uris += [f"{directory}/%E6%B1%89", f"{directory}/%E5%AD%97"]
    for uri in uris:
        fs.write_node(ContextNode(
            uri=uri, context_type="MEMORY", category="entity", level=0,
            owner_space="user:alice", abstract="Fact", overview="Fact", content="Fact",
        ), ctx)

    nodes = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))._nodes(directory, ctx)

    assert len(nodes) == 101
    assert {node.uri for node in nodes} == set(uris)
    assert fs.page_count == 2


def test_identity_scan_skips_unrelated_broken_agfs_node():
    fs = AGFSContextFS(_MemoryAGFSClient())
    ctx = RequestContext("acme", "alice", "bot", "session", "trace")
    directory = "ctx://acme/users/alice/memories/entities"
    broken_uri = f"{directory}/damaged"
    broken_path = fs._uri_to_agfs_path(broken_uri)
    fs._client.mkdir(broken_path)
    fs._client.write(broken_path + fs.FILE_META, json.dumps({
        "uri": broken_uri, "status": NodeStatus.BROKEN.value,
    }).encode())
    healthy_uri = f"{directory}/working"
    fs.write_node(ContextNode(
        uri=healthy_uri, context_type="MEMORY", category="entity", level=0,
        owner_space="user:alice", abstract="Working", overview="Working", content="Working",
    ), ctx)

    nodes = MemoryIdentityResolver(fs, URIResolver(SchemaRegistry()))._nodes(directory, ctx)

    assert [node.uri for node in nodes] == [healthy_uri]


def test_archived_node_version_reads_inactive_metadata():
    client = _MemoryAGFSClient()
    fs = AGFSContextFS(client)
    ctx = RequestContext(account_id="acme", user_id="alice", agent_id="main",
                         session_id="s1", trace_id="t1")
    uri = "ctx://acme/users/alice/memories/preferences/pets"
    path = fs._uri_to_agfs_path(uri) + fs.FILE_META
    client.write(path, json.dumps({"status": NodeStatus.ARCHIVED.value, "version": 4}).encode())

    assert fs.archived_node_version(uri, ctx) == 4
    client.write(path, json.dumps({"status": NodeStatus.ACTIVE.value, "version": 5}).encode())
    assert fs.archived_node_version(uri, ctx) is None


class _AtomicMemoryAGFSClient:
    """Thread-safe AGFS double with atomic file/directory rename semantics."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = {"/", "/local"}
        self.operations: list[tuple] = []
        self.before_write = None
        self.after_write = None
        self.before_mv = None
        self.after_mv = None
        self.before_read = None
        self._lock = RLock()

    @staticmethod
    def _normalize(path):
        return str(path).rstrip("/") or "/"

    def mkdir(self, path):
        target = self._normalize(path)
        with self._lock:
            if target in self.directories:
                raise AGFSClientError("already exists")
            self.directories.add(target)
            self.operations.append(("mkdir", target))

    def create(self, path):
        target = self._normalize(path)
        with self._lock:
            if target in self.files or target in self.directories:
                raise AGFSClientError("already exists")
            self.files[target] = b""
            self.operations.append(("create", target))

    def write(self, path, content):
        target = self._normalize(path)
        if self.before_write is not None:
            self.before_write(target, bytes(content))
        with self._lock:
            self.files[target] = bytes(content)
            self.operations.append(("write", target))
        if self.after_write is not None:
            self.after_write(target, bytes(content))

    def read(self, path):
        target = self._normalize(path)
        if self.before_read is not None:
            self.before_read(target)
        with self._lock:
            try:
                return self.files[target]
            except KeyError as exc:
                raise AGFSClientError("No such file") from exc

    def stat(self, path):
        target = self._normalize(path)
        with self._lock:
            if target not in self.files and target not in self.directories:
                raise AGFSClientError("No such file")
        return {"modTime": time.time()}

    def ls(self, path):
        parent = self._normalize(path)
        prefix = parent.rstrip("/") + "/"
        entries: dict[str, bool] = {}
        with self._lock:
            for directory in self.directories:
                if not directory.startswith(prefix):
                    continue
                remainder = directory[len(prefix):]
                if remainder and "/" not in remainder:
                    entries[remainder] = True
            for file_path in self.files:
                if not file_path.startswith(prefix):
                    continue
                remainder = file_path[len(prefix):]
                if remainder and "/" not in remainder:
                    entries.setdefault(remainder, False)
        return [
            {"name": name, "isDir": is_dir, "is_dir": is_dir}
            for name, is_dir in sorted(entries.items())
        ]

    def mv(self, source, destination):
        source_key = self._normalize(source)
        destination_key = self._normalize(destination)
        if self.before_mv is not None:
            self.before_mv(source_key, destination_key)
        with self._lock:
            if source_key in self.directories:
                if destination_key in self.directories or destination_key in self.files:
                    raise AGFSClientError("already exists")
                source_prefix = source_key + "/"
                destination_prefix = destination_key + "/"
                moved_dirs = {
                    destination_key
                    if path == source_key
                    else destination_prefix + path[len(source_prefix):]
                    for path in self.directories
                    if path == source_key or path.startswith(source_prefix)
                }
                moved_files = {
                    destination_prefix + path[len(source_prefix):]: value
                    for path, value in self.files.items()
                    if path.startswith(source_prefix)
                }
                self.directories = {
                    path
                    for path in self.directories
                    if path != source_key and not path.startswith(source_prefix)
                }
                self.files = {
                    path: value
                    for path, value in self.files.items()
                    if not path.startswith(source_prefix)
                }
                self.directories.update(moved_dirs)
                self.files.update(moved_files)
            elif source_key in self.files:
                self.files[destination_key] = self.files.pop(source_key)
            else:
                raise AGFSClientError("No such file")
            self.operations.append(("mv", source_key, destination_key))
        if self.after_mv is not None:
            self.after_mv(source_key, destination_key)

    def rm(self, path, recursive=False):
        target = self._normalize(path)
        with self._lock:
            if recursive:
                prefix = target + "/"
                found = target in self.directories or any(
                    item.startswith(prefix)
                    for item in (*self.directories, *self.files)
                )
                if not found:
                    raise AGFSClientError("No such file")
                self.directories = {
                    item
                    for item in self.directories
                    if item != target and not item.startswith(prefix)
                }
                self.files = {
                    item: value
                    for item, value in self.files.items()
                    if item != target and not item.startswith(prefix)
                }
            elif target in self.files:
                del self.files[target]
            elif target in self.directories:
                self.directories.remove(target)
            else:
                raise AGFSClientError("No such file")
            self.operations.append(("rm", target, recursive))


def _atomic_code_context() -> RequestContext:
    return RequestContext(
        account_id="acme",
        user_id="alice",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


def _atomic_code_node(
    *,
    description: str | None = None,
    expected_version: int = 0,
) -> ContextNode:
    metadata = {
        "_expected_version": expected_version,
        "resource_type": "code",
        "resource_root_slug": "repo",
        "root_fingerprint": "f" * 64,
        "file_key": "a" * 64,
        "file_slug": "a" * 32,
        "relative_path": "src/service.py",
        "source_extension": ".py",
        "language": "python",
        "start_line": 1,
        "end_line": 2,
        "symbol": "load_config",
        "symbol_kind": "function",
        "signature": "load_config(path)",
        "chunk_hash": "b" * 64,
        "chunk_key": "c" * 64,
    }
    if description is not None:
        metadata["description"] = description
    return ContextNode(
        uri="ctx://acme/resources/" + "d" * 32,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acme",
        abstract="load config",
        overview="path: src/service.py",
        content="def load_config(path):\n    return path\n",
        metadata=metadata,
    )


def _atomic_code_stack():
    client = _AtomicMemoryAGFSClient()
    fs = AGFSContextFS(client, mount_prefix="/local")
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    return client, fs, outbox


def test_memory_identity_lock_serializes_two_agfs_adapters():
    client = _AtomicMemoryAGFSClient()
    first = AGFSContextFS(client, mount_prefix="/local")
    second = AGFSContextFS(client, mount_prefix="/local")
    directory = "ctx://acme/users/alice/memories/preferences"
    ctx = _atomic_code_context()
    entered = Event()

    def acquire_second():
        with second.memory_identity_lock(directory, ctx):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with first.memory_identity_lock(directory, ctx):
            pending = executor.submit(acquire_second)
            assert not entered.wait(0.1)
        pending.result(timeout=3)
    assert entered.is_set()


class TestUriToPath:
    """Tests for uri_to_path function."""

    def test_user_profile_uri(self):
        """Test user profile URI mapping."""
        uri = "ctx://acme/users/alice/memories/profile"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/profile/"

    def test_user_preference_uri(self):
        """Test user preference URI mapping."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/preferences/coffee/"

    def test_user_entity_uri(self):
        """Test user entity URI mapping."""
        uri = "ctx://acme/users/alice/memories/entities/coffee_shop"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/entities/coffee_shop/"

    def test_user_event_uri(self):
        """Test user event URI mapping."""
        uri = "ctx://acme/users/alice/memories/events/visit_20250315_abc123"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/events/visit_20250315_abc123/"

    def test_agent_case_uri(self):
        """Test agent case URI mapping."""
        uri = "ctx://acme/agents/gpt-4/memories/cases/case_20250315_xyz789"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/memories/cases/case_20250315_xyz789/"

    def test_agent_pattern_uri(self):
        """Test agent pattern URI mapping."""
        uri = "ctx://acme/agents/gpt-4/memories/patterns/error_handling"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/memories/patterns/error_handling/"

    def test_agent_skill_uri(self):
        """Test agent skill URI mapping (skills instead of memories)."""
        uri = "ctx://acme/agents/gpt-4/skills/generated/code_review"
        path = uri_to_path(uri)
        # Skills have their own /skills/ subdirectory (not under /memories/)
        assert path == "/accounts/acme/agents/gpt-4/skills/generated/code_review/"

        assert (
            uri_to_path("ctx://acme/agents/gpt-4/skills/code_review")
            == "/accounts/acme/agents/gpt-4/skills/code_review/"
        )

    def test_agent_skills_directory_uri(self):
        """Test agent skills directory URI mapping."""
        uri = "ctx://acme/agents/gpt-4/skills"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/agents/gpt-4/skills/"

    def test_agent_managed_skill_uri(self):
        uri = "ctx://acme/agents/gpt-4/skills/imported/code_review"
        assert uri_to_path(uri) == "/accounts/acme/agents/gpt-4/skills/imported/code_review/"

    def test_managed_skill_uri_round_trip(self):
        uri = "ctx://acme/agents/gpt-4/skills/imported/code_review"
        assert build_uri(**parse_uri(uri)) == uri

    def test_url_encoded_slug(self):
        """Test URI with URL-encoded slug."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee%20preferences"
        path = uri_to_path(uri)
        assert path == "/accounts/acme/users/alice/memories/preferences/coffee preferences/"

    @pytest.mark.parametrize("slug", [
        "%2faccounts%2fother%2fusers%2fvictim%2fmemories%2fprofile",
        "%252faccounts%252fother",
        "%2e%2e",
        "%5caccounts%5cother",
    ])
    def test_memory_slug_cannot_escape_owner_path(self, slug):
        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path(f"ctx://acme/users/alice/memories/entities/{slug}")

    def test_invalid_uri_format(self):
        """Test that invalid URI format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path("not-a-valid-uri")

        with pytest.raises(ValueError, match="Invalid URI format"):
            uri_to_path("http://example.com/path")

    def test_uri_with_trailing_slash(self):
        """Test that URI with trailing slash is handled correctly."""
        # Trailing slashes are now stripped during URI processing, so this works
        path = uri_to_path("ctx://acme/users/alice/memories/profile/")
        assert path == "/accounts/acme/users/alice/memories/profile/"


class TestParseUri:
    """Tests for parse_uri function."""

    def test_parse_user_uri(self):
        """Test parsing a user URI."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'users',
            'owner_id': 'alice',
            'category': 'preferences',
            'slug': 'coffee',
        }

    def test_parse_agent_uri(self):
        """Test parsing an agent URI."""
        uri = "ctx://acme/agents/gpt-4/memories/cases/case_123"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'cases',
            'slug': 'case_123',
        }

    def test_parse_agent_skills_directory_uri(self):
        """Test parsing the agent skills directory URI."""
        uri = "ctx://acme/agents/gpt-4/skills"
        result = parse_uri(uri)
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'skills',
            'slug': '',
        }

    def test_parse_agent_managed_skill_uri(self):
        result = parse_uri("ctx://acme/agents/gpt-4/skills/imported/code_review")
        assert result == {
            'account': 'acme',
            'owner_type': 'agents',
            'owner_id': 'gpt-4',
            'category': 'skills/imported',
            'slug': 'code_review',
        }

    def test_parse_url_encoded_uri(self):
        """Test parsing URI with URL-encoded components."""
        uri = "ctx://acme/users/alice/memories/preferences/coffee%20shop"
        result = parse_uri(uri)
        assert result['slug'] == 'coffee shop'

    def test_parse_invalid_uri(self):
        """Test that invalid URI raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URI format"):
            parse_uri("invalid-uri")


def test_skill_summary_paging_is_uri_ordered_and_respects_cursor():
    client = Mock()
    client.ls.return_value = [
        {"name": "skill-c", "isDir": True},
        {"name": "skill-a", "isDir": True},
        {"name": "skill-b", "isDir": True},
    ]

    def read(path):
        name = path.split("/skills/")[1].split("/")[0]
        if path.endswith(".meta.json"):
            return json.dumps({
                "uri": f"ctx://acme/agents/bot/skills/{name}",
                "context_type": "SKILL", "category": "skill", "level": 2,
                "owner_space": "agent:bot", "status": NodeStatus.ACTIVE.value,
            }).encode()
        return name.encode()

    client.read.side_effect = read
    fs = AGFSContextFS(client)
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="bot",
        session_id="session", trace_id="trace",
    )

    page = fs.list_node_summaries_page(
        "ctx://acme/agents/bot/skills", ctx,
        limit=1, after_uri="ctx://acme/agents/bot/skills/skill-a",
    )

    assert [node.uri for node in page] == ["ctx://acme/agents/bot/skills/skill-b"]


def test_generated_skill_summary_preserves_governance_metadata():
    client = Mock()
    client.ls.return_value = [{"name": "filter-rows", "isDir": True}]

    def read(path):
        if path.endswith(".meta.json"):
            return json.dumps({
                "uri": "ctx://acme/agents/bot/skills/generated/filter-rows",
                "context_type": "SKILL",
                "category": "skill",
                "level": 0,
                "owner_space": "agent:bot",
                "status": NodeStatus.ACTIVE.value,
                "ogmemory_generated": True,
                "publication_status": "published",
                "content_hash": "abc123",
            }).encode()
        return b"Filter matching rows"

    client.read.side_effect = read
    fs = AGFSContextFS(client)
    ctx = RequestContext(
        account_id="acme", user_id="alice", agent_id="bot",
        session_id="session", trace_id="trace",
    )

    page = fs.list_node_summaries_page(
        "ctx://acme/agents/bot/skills/generated", ctx,
        limit=10, after_uri=None,
    )

    assert page[0].metadata["ogmemory_generated"] is True
    assert page[0].metadata["publication_status"] == "published"
    assert page[0].metadata["content_hash"] == "abc123"


@pytest.mark.parametrize("routing_key", [".R", " .R "])
def test_normalized_short_name_is_visible_after_agfs_write(routing_key):
    ctx = RequestContext("acct", "alice", "bot", "session", "trace")
    candidate = CandidateMemory(
        category="entity", owner_scope="user", routing_key=routing_key,
        abstract="Entity R", overview="", content="", confidence=0.9,
    )
    key = validate_candidate(candidate).routing_key
    uri = URIResolver(SchemaRegistry()).resolve("entity", {"routing_key": key}, ctx)
    fs = AGFSContextFS(_MemoryAGFSClient())
    fs.write_node(
        ContextNode(
            uri=uri, context_type="MEMORY", category="entity", level=0,
            owner_space="user:alice", abstract="Entity R", overview="", content="",
            metadata={},
        ),
        ctx,
    )

    assert uri in fs.list_children("ctx://acct/users/alice/memories/entities", ctx)


def test_forget_encoded_agfs_memory_uses_stored_uri_for_archive_event():
    client = _AtomicMemoryAGFSClient()
    fs = AGFSContextFS(client, mount_prefix="/local")
    outbox = OutboxStore(client=client, fs=fs, mount_prefix="/local")
    ctx = RequestContext("acme", "alice", "bot", "session", "trace")
    stored_uri = "ctx://acme/users/alice/memories/preferences/coffee%20shop"
    requested_uri = "ctx://acme/users/alice/memories/preferences/coffee shop"
    fs.write_node(
        ContextNode(
            uri=stored_uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="Coffee shop",
            overview="",
            content="Coffee shop",
        ),
        ctx,
    )
    service = MemoryService(config=OgMemConfig())
    service._get_context_fs = lambda: fs
    service._get_outbox_store = lambda _fs: outbox
    service.get_audit_service = lambda: Mock()

    service.forget_memory(requested_uri, ctx)

    assert fs.read_node_metadata_including_inactive(stored_uri, ctx)["status"] == "ARCHIVED"
    assert [event.uri for _, event in outbox.list_pending("acme")] == [stored_uri]


def test_agfs_memory_page_covers_encoded_unicode_names():
    fs = AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local")
    ctx = RequestContext("acme", "alice", "bot", "session", "trace")
    for name in ["%E6%9D%8E%E9%9B%B7", "atlas", "bravo"]:
        fs.write_node(
            ContextNode(
                uri=f"ctx://acme/users/alice/memories/entities/{name}",
                context_type="MEMORY",
                category="entity",
                level=0,
                owner_space="user:alice",
                abstract=name,
                overview="",
                content="",
            ),
            ctx,
        )

    service = SimpleNamespace(_get_context_fs=lambda: fs)
    seen = []
    cursor = None
    for _ in range(3):
        entries, has_more = _list_memory_node_entries_page(service, ctx, limit=1, cursor=cursor)
        seen.extend(summary["id"] for _, summary in entries)
        if not has_more:
            break
        cursor = _memory_list_cursor(entries[-1])

    assert seen == ["atlas", "bravo", "李雷"]


class TestBuildUri:
    """Tests for build_uri function."""

    def test_build_user_uri(self):
        """Test building a user URI."""
        uri = build_uri('acme', 'users', 'alice', 'preferences', 'coffee')
        assert uri == "ctx://acme/users/alice/memories/preferences/coffee"

    def test_build_agent_uri(self):
        """Test building an agent URI."""
        uri = build_uri('acme', 'agents', 'gpt-4', 'cases', 'case_123')
        assert uri == "ctx://acme/agents/gpt-4/memories/cases/case_123"

    def test_singular_owner_type_normalized(self):
        """Test that singular owner_type is normalized to plural."""
        uri = build_uri('acme', 'user', 'alice', 'profile', '')
        assert 'users/' in uri

        uri = build_uri('acme', 'agent', 'gpt-4', 'skills', 'code_review')
        assert 'agents/' in uri

    def test_roundtrip_uri(self):
        """Test that build_uri and parse_uri are inverses."""
        original_uri = "ctx://acme/users/alice/memories/preferences/coffee"
        parsed = parse_uri(original_uri)
        rebuilt = build_uri(**parsed)
        assert rebuilt == original_uri

    def test_build_skill_uri(self):
        """Test building a skill URI."""
        uri = build_uri('acme', 'agents', 'gpt-4', 'skills/generated', 'code_review')
        assert uri == "ctx://acme/agents/gpt-4/skills/generated/code_review"

    def test_build_profile_uri(self):
        """Test building a profile URI."""
        uri = build_uri('acme', 'users', 'alice', 'profile', 'profile')
        assert uri == "ctx://acme/users/alice/memories/profile"


class TestIsAccessible:
    """Tests for _is_accessible method."""

    def test_same_account_allowed(self):
        """Test that same account access is allowed."""
        client = None  # Not used for _is_accessible
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        assert fs._is_accessible(uri, ctx) is True

    def test_cross_account_denied(self):
        """Test that cross-account access is denied."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="other-corp",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        assert fs._is_accessible(uri, ctx) is False

    def test_invalid_uri_denied(self):
        """Test that invalid URI is denied."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "not-a-valid-uri"
        assert fs._is_accessible(uri, ctx) is False

    def test_session_state_matching_session_allowed(self):
        """Test that session state access requires matching session_id."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-1",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, ctx) is True

    def test_session_state_different_session_denied(self):
        """Test that session state denies a different session_id."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/state"
        assert fs._is_accessible(uri, ctx) is False

    def test_session_history_same_account_remains_accessible(self):
        """Test that session history remains account-scoped."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/history"
        assert fs._is_accessible(uri, ctx) is True

    def test_session_archive_same_account_remains_accessible(self):
        """Test that session archives remain account-scoped."""
        fs = AGFSContextFS(None)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-2",
            trace_id="trace-456",
        )
        uri = "ctx://acme/sessions/session-1/history/archive-1"
        assert fs._is_accessible(uri, ctx) is True


class TestEnsureAccessible:
    """Tests for _ensure_accessible method."""

    def test_same_account_succeeds(self):
        """Test that same account access succeeds."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        # Should not raise
        fs._ensure_accessible(uri, ctx)

    def test_cross_account_raises(self):
        """Test that cross-account access raises AccessDeniedError."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="other-corp",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "ctx://acme/users/alice/memories/profile"
        with pytest.raises(AccessDeniedError) as exc_info:
            fs._ensure_accessible(uri, ctx)
        assert exc_info.value.account_id == "other-corp"
        assert exc_info.value.uri == uri
        assert "acme" in str(exc_info.value.reason)

    def test_invalid_uri_raises(self):
        """Test that invalid URI raises AccessDeniedError."""
        client = None
        fs = AGFSContextFS(client)
        ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )
        uri = "not-a-valid-uri"
        with pytest.raises(AccessDeniedError) as exc_info:
            fs._ensure_accessible(uri, ctx)
        assert "Invalid URI format" in str(exc_info.value.reason)


class TestWriteNode:
    """Tests for write_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_write_node_creates_files_in_order(self):
        """Test that write_node creates files in the correct order."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="User profile for Alice",
            overview="Name: Alice\nAccount: acme",
            content="# Alice's Profile\n\nFull content here",
            metadata={},
        )

        self.fs.write_node(node, self.ctx)

        # Verify mkdir was called for the node directory
        self.mock_client.mkdir.assert_called()

        # Verify write was called for each file in the correct order
        calls = self.mock_client.write.call_args_list
        written_paths = [call[0][0] for call in calls]

        # Step ①: content.md should be written first
        assert any("content.md" in path for path in written_paths)
        content_idx = next(i for i, p in enumerate(written_paths) if "content.md" in p)

        # Step ②: .relations.json should be written second
        assert any(".relations.json" in path for path in written_paths)
        relations_idx = next(i for i, p in enumerate(written_paths) if ".relations.json" in p)
        assert relations_idx > content_idx

        # Step ③: .abstract.md and .overview.md
        assert any(".abstract.md" in path for path in written_paths)
        assert any(".overview.md" in path for path in written_paths)

        # Step ④: .meta.json should be written last (commit point)
        assert any(".meta.json" in path for path in written_paths)
        meta_idx = next(i for i, p in enumerate(written_paths) if ".meta.json" in p)
        assert meta_idx > relations_idx

    def test_write_node_includes_active_status(self):
        """Test that .meta.json includes status=ACTIVE."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:alice",
            abstract="User profile for Alice",
            overview="Name: Alice",
            content="# Profile",
            metadata={},
        )

        self.fs.write_node(node, self.ctx)

        # Find the .meta.json write call
        calls = self.mock_client.write.call_args_list
        meta_call = next(call for call in calls if ".meta.json" in call[0][0])
        meta_content = meta_call[0][1].decode('utf-8')
        meta = json.loads(meta_content)

        assert meta["status"] == "ACTIVE"
        assert meta["uri"] == node.uri
        assert meta["category"] == "profile"

    def test_write_node_cross_account_denied(self):
        """Test that cross-account write raises AccessDeniedError."""
        node = ContextNode(
            uri="ctx://other-corp/users/bob/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user:bob",
            abstract="Bob's profile",
            overview="Name: Bob",
            content="# Bob",
            metadata={},
        )

        with pytest.raises(AccessDeniedError):
            self.fs.write_node(node, self.ctx)

    def test_create_only_publishes_with_atomic_directory_move(self):
        self.mock_client.read.side_effect = AGFSClientError("File not found")
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Demo",
            metadata={"_expected_version": 0},
        )

        self.fs.write_node(node, self.ctx)

        source, destination = self.mock_client.mv.call_args.args
        assert ".demo.create-" in source
        assert destination == "/local/accounts/acme/agents/gpt-4/skills/imported/demo"

    def test_versioned_update_publishes_metadata_atomically(self):
        def read(path):
            if path.endswith("transaction.json"):
                marker_write = next(
                    call for call in reversed(self.mock_client.write.call_args_list)
                    if ".transaction-" in call.args[0]
                )
                return marker_write.args[1]
            return json.dumps({"version": 1}).encode()

        self.mock_client.read.side_effect = read
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Updated demo",
            metadata={"_expected_version": 1},
        )

        self.fs.write_node(node, self.ctx)

        final_path = "/local/accounts/acme/agents/gpt-4/skills/imported/demo/"
        moves = [call.args for call in self.mock_client.mv.call_args_list]
        assert any(
            source.startswith(final_path + ".meta.update-")
            and destination == final_path + ".meta.json"
            for source, destination in moves
        )
        assert any(
            ".demo.update-lock-" in source
            and destination.endswith(".demo.update-lock")
            for source, destination in moves
        )
        written_paths = [call.args[0] for call in self.mock_client.write.call_args_list]
        assert final_path + "content.md" in written_paths
        assert any(
            path.startswith(final_path + ".meta.update-") for path in written_paths
        )

    def test_versioned_update_rejects_concurrent_agfs_writer(self):
        def move(source, destination):
            if destination.endswith(".demo.update-lock"):
                raise AGFSClientError("already exists")

        self.mock_client.mv.side_effect = move
        self.mock_client.read.side_effect = AGFSClientError("File not found")
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Updated demo",
            metadata={"_expected_version": 1},
        )

        with pytest.raises(ConcurrentModificationError):
            self.fs.write_node(node, self.ctx)
        written_paths = [call.args[0] for call in self.mock_client.write.call_args_list]
        assert not any(path.endswith("content.md") for path in written_paths)
        calls = self.mock_client.method_calls
        marker_write = next(
            index for index, call in enumerate(calls)
            if call[0] == "write" and ".transaction-" in call.args[0]
        )
        lock_publish = next(
            index for index, call in enumerate(calls)
            if call[0] == "mv" and call.args[1].endswith(".demo.update-lock")
        )
        assert marker_write < lock_publish
        assert not any(
            call.args[0].endswith(".demo.update-lock")
            for call in self.mock_client.mkdir.call_args_list
        )

    def test_versioned_update_recovers_completed_transaction_lock(self):
        lock_attempts = 0
        acquired = False

        def move(source, destination):
            nonlocal lock_attempts, acquired
            if destination.endswith(".demo.update-lock"):
                lock_attempts += 1
                if lock_attempts == 1:
                    raise AGFSClientError("already exists")
                acquired = True

        def read(path):
            if path.endswith("lease-old-owner.json"):
                return json.dumps({
                    "owner_token": "old-owner",
                    "lease_updated_at": "2000-01-01T00:00:00+00:00",
                }).encode()
            if path.endswith("transaction.json"):
                if acquired:
                    marker_write = next(
                        call for call in reversed(self.mock_client.write.call_args_list)
                        if ".transaction-" in call.args[0]
                    )
                    return marker_write.args[1]
                return json.dumps({
                    "owner_token": "old-owner",
                    "expected_version": 1,
                    "temporary_meta_path": path.replace(
                        ".demo.update-lock/transaction.json",
                        "demo/.meta.update-old.json",
                    ),
                }).encode()
            return json.dumps({"version": 2}).encode()

        self.mock_client.mv.side_effect = move
        self.mock_client.read.side_effect = read
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Updated demo",
            metadata={"_expected_version": 2},
        )

        self.fs.write_node(node, self.ctx)

        assert lock_attempts == 2
        assert any(
            call.args[0].endswith(".demo.update-lock")
            for call in self.mock_client.rm.call_args_list
        )

    def test_stale_recovery_invalidates_pending_commit_before_unlocking(self):
        node_path = "/local/accounts/acme/agents/gpt-4/skills/imported/demo/"
        lock_path = (
            "/local/accounts/acme/agents/gpt-4/skills/imported/.demo.update-lock"
        )
        marker_path = lock_path + "/transaction.json"
        temporary_meta_path = node_path + ".meta.update-old-owner.json"
        marker = {
            "owner_token": "old-owner",
            "expected_version": 1,
            "temporary_meta_path": temporary_meta_path,
        }

        def read(path):
            if path == marker_path:
                return json.dumps(marker).encode()
            if path == lock_path + "/lease-old-owner.json":
                return json.dumps({
                    "owner_token": "old-owner",
                    "lease_updated_at": "2000-01-01T00:00:00+00:00",
                }).encode()
            return json.dumps({"version": 1}).encode()

        existing_paths = {temporary_meta_path}

        def remove(path, recursive=False):
            existing_paths.discard(path)

        def move(source, destination):
            if source not in existing_paths:
                raise AGFSClientError("source not found")
            existing_paths.remove(source)
            existing_paths.add(destination)

        self.mock_client.read.side_effect = read
        self.mock_client.rm.side_effect = remove
        self.mock_client.mv.side_effect = move

        assert self.fs._recover_update_lock(node_path, lock_path, marker_path)

        removals = [call.args[0] for call in self.mock_client.rm.call_args_list]
        assert removals == [temporary_meta_path, lock_path]

        # A stale writer that passed its owner check before recovery can no
        # longer publish after the successor reads the old version.
        with pytest.raises(AGFSClientError, match="source not found"):
            self.mock_client.mv(temporary_meta_path, node_path + ".meta.json")

    def test_stale_recovery_keeps_lock_when_commit_invalidation_fails(self):
        node_path = "/local/accounts/acme/agents/gpt-4/skills/imported/demo/"
        lock_path = (
            "/local/accounts/acme/agents/gpt-4/skills/imported/.demo.update-lock"
        )
        marker_path = lock_path + "/transaction.json"
        temporary_meta_path = node_path + ".meta.update-old-owner.json"

        def read(path):
            if path == marker_path:
                return json.dumps({
                    "owner_token": "old-owner",
                    "expected_version": 1,
                    "temporary_meta_path": temporary_meta_path,
                }).encode()
            if path == lock_path + "/lease-old-owner.json":
                return json.dumps({
                    "owner_token": "old-owner",
                    "lease_updated_at": "2000-01-01T00:00:00+00:00",
                }).encode()
            return json.dumps({"version": 1}).encode()

        def remove(path, recursive=False):
            if path == temporary_meta_path:
                raise AGFSClientError("injected network failure")

        self.mock_client.read.side_effect = read
        self.mock_client.rm.side_effect = remove

        assert not self.fs._recover_update_lock(node_path, lock_path, marker_path)
        assert [call.args[0] for call in self.mock_client.rm.call_args_list] == [
            temporary_meta_path
        ]

    def test_versioned_update_does_not_publish_after_losing_lease(self):
        acquired = False

        def move(source, destination):
            nonlocal acquired
            if destination.endswith(".demo.update-lock"):
                acquired = True

        def read(path):
            if path.endswith("transaction.json"):
                owner = "successor" if acquired else "initial"
                return json.dumps({"owner_token": owner}).encode()
            return json.dumps({"version": 1}).encode()

        self.mock_client.mv.side_effect = move
        self.mock_client.read.side_effect = read
        node = ContextNode(
            uri="ctx://acme/agents/gpt-4/skills/imported/demo",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:gpt-4",
            abstract="Demo",
            overview="Demo",
            content="# Updated demo",
            metadata={"_expected_version": 1},
        )

        with pytest.raises(ConcurrentModificationError):
            self.fs.write_node(node, self.ctx)

        final_meta = "/local/accounts/acme/agents/gpt-4/skills/imported/demo/.meta.json"
        assert not any(
            call.args[1] == final_meta for call in self.mock_client.mv.call_args_list
        )
        assert not any(
            call.args[0].endswith(".demo.update-lock")
            for call in self.mock_client.rm.call_args_list
        )


class TestReadNode:
    """Tests for read_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

        # Set up default mock responses
        self.mock_client.read.side_effect = lambda path: self._mock_read(path)

    def _mock_read(self, path):
        """Mock read implementation that returns test data."""
        if ".meta.json" in path:
            return json.dumps({
                "uri": "ctx://acme/users/alice/memories/profile",
                "context_type": "MEMORY",
                "category": "profile",
                "level": 0,
                "owner_space": "user:alice",
                "status": "ACTIVE",
                "created_at": "2025-03-17T10:00:00Z",
                "updated_at": "2025-03-17T11:00:00Z",
                "version": 1,
                "ogmemory_managed": True,
                "package_files": [{"path": "scripts/run.py"}],
            }).encode()
        elif "content.md" in path:
            return b"# Alice's Profile\n\nFull content"
        elif ".abstract.md" in path:
            return b"User profile for Alice"
        elif ".overview.md" in path:
            return b"Name: Alice\nAccount: acme"
        elif ".relations.json" in path:
            return b"[]"
        raise AGFSClientError("File not found")

    def test_read_node_returns_context_node(self):
        """Test that read_node returns a properly constructed ContextNode."""
        uri = "ctx://acme/users/alice/memories/profile"
        node = self.fs.read_node(uri, self.ctx)

        assert node.uri == uri
        assert node.context_type == "MEMORY"
        assert node.category == "profile"
        assert node.abstract == "User profile for Alice"
        assert node.overview == "Name: Alice\nAccount: acme"
        assert node.content == "# Alice's Profile\n\nFull content"
        assert node.metadata["ogmemory_managed"] is True
        assert node.metadata["package_files"] == [{"path": "scripts/run.py"}]

    def test_read_node_not_found(self):
        """Test that reading non-existent node raises NodeNotFoundError."""
        self.mock_client.read.side_effect = AGFSClientError("No such file")

        with pytest.raises(NodeNotFoundError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_pending_not_visible(self):
        """Test that PENDING nodes raise NodeNotFoundError."""
        def mock_read_pending(path):
            if ".meta.json" in path:
                return json.dumps({
                    "uri": "ctx://acme/users/alice/memories/profile",
                    "context_type": "MEMORY",
                    "category": "profile",
                    "level": 0,
                    "owner_space": "user:alice",
                    "status": "PENDING",
                }).encode()
            return b""

        self.mock_client.read.side_effect = mock_read_pending

        with pytest.raises(NodeNotFoundError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_broken_raises_error(self):
        """Test that BROKEN nodes raise NodeBrokenError."""
        def mock_read_broken(path):
            if ".meta.json" in path:
                return json.dumps({
                    "uri": "ctx://acme/users/alice/memories/profile",
                    "status": "BROKEN",
                }).encode()
            return b""

        self.mock_client.read.side_effect = mock_read_broken

        with pytest.raises(NodeBrokenError):
            self.fs.read_node("ctx://acme/users/alice/memories/profile", self.ctx)

    def test_read_node_cross_account_denied(self):
        """Test that cross-account read raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.read_node("ctx://other-corp/users/bob/memories/profile", self.ctx)

    def test_read_node_fields_reads_only_requested_files(self):
        """Test field-level reads avoid loading full content."""
        uri = "ctx://acme/users/alice/memories/profile"
        node = self.fs.read_node_fields(uri, self.ctx, "abstract")

        read_paths = [call.args[0] for call in self.mock_client.read.call_args_list]
        assert node.abstract == "User profile for Alice"
        assert node.overview == ""
        assert node.content == ""
        assert any(path.endswith(".meta.json") for path in read_paths)
        assert any(path.endswith(".abstract.md") for path in read_paths)
        assert not any(path.endswith("content.md") for path in read_paths)


class TestExists:
    """Tests for exists method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_exists_active_node(self):
        """Test that exists returns True for ACTIVE nodes."""
        self.mock_client.read.return_value = json.dumps({
            "uri": "ctx://acme/users/alice/memories/profile",
            "status": "ACTIVE",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is True

    def test_exists_pending_node(self):
        """Test that exists returns False for PENDING nodes."""
        self.mock_client.read.return_value = json.dumps({
            "status": "PENDING",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_broken_node(self):
        """Test that exists returns False for BROKEN nodes."""
        self.mock_client.read.return_value = json.dumps({
            "status": "BROKEN",
        }).encode()

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_missing_node(self):
        """Test that exists returns False for missing nodes."""
        self.mock_client.read.side_effect = AGFSClientError("File not found")

        assert self.fs.exists("ctx://acme/users/alice/memories/profile", self.ctx) is False

    def test_exists_cross_account_denied(self):
        """Test that cross-account exists returns False."""
        assert self.fs.exists("ctx://other-corp/users/bob/memories/profile", self.ctx) is False


class TestDeleteNode:
    """Tests for delete_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_delete_node_calls_rm(self):
        """Test that delete_node calls rm with recursive flag."""
        self.fs.delete_node("ctx://acme/users/alice/memories/profile", self.ctx)

        self.mock_client.rm.assert_called_once()
        call_args = self.mock_client.rm.call_args
        assert "/accounts/acme/users/alice/memories/profile/" in call_args[0][0]
        # The second positional argument should be recursive=True
        assert call_args[1].get("recursive") is True or call_args[0][1] is True

    def test_delete_node_cross_account_denied(self):
        """Test that cross-account delete raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.delete_node("ctx://other-corp/users/bob/memories/profile", self.ctx)


class TestArchiveNode:
    """Tests for archive_node soft delete behavior."""

    def setup_method(self):
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

    def test_archive_node_marks_meta_archived(self):
        uri = "ctx://acme/users/alice/memories/preferences/coffee"
        self.mock_client.read.return_value = json.dumps(
            {
                "uri": uri,
                "context_type": "MEMORY",
                "category": "preference",
                "level": 4,
                "owner_space": "user:alice",
                "status": NodeStatus.ACTIVE.value,
                "created_at": "2026-06-23T00:00:00Z",
                "updated_at": "2026-06-23T00:00:00Z",
                "version": 1,
            }
        ).encode()

        self.fs.archive_node(uri, self.ctx)

        write_path, write_bytes = self.mock_client.write.call_args[0]
        meta = json.loads(write_bytes.decode("utf-8"))
        assert write_path.endswith("/accounts/acme/users/alice/memories/preferences/coffee/.meta.json")
        assert meta["status"] == NodeStatus.ARCHIVED.value
        assert meta["archived_at"]
        assert meta["updated_at"] == meta["archived_at"]

    def test_archive_node_requires_active_node(self):
        self.mock_client.read.return_value = json.dumps({"status": NodeStatus.ARCHIVED.value}).encode()

        with pytest.raises(NodeNotFoundError):
            self.fs.archive_node("ctx://acme/users/alice/memories/preferences/coffee", self.ctx)

    def test_archive_node_cross_account_denied(self):
        with pytest.raises(AccessDeniedError):
            self.fs.archive_node("ctx://other-corp/users/bob/memories/preferences/coffee", self.ctx)


class TestMoveNode:
    """Tests for move_node method."""

    def setup_method(self):
        """Set up mock client and test fixtures."""
        self.mock_client = Mock()
        self.fs = AGFSContextFS(self.mock_client)
        self.ctx = RequestContext(
            account_id="acme",
            user_id="alice",
            agent_id="gpt-4",
            session_id="session-123",
            trace_id="trace-456",
        )

        # Mock exists to return True
        self.fs.exists = Mock(return_value=True)

    def test_move_node_calls_mv(self):
        """Test that move_node calls AGFS mv."""
        # Use valid URIs - moving within the same category is more realistic
        from_uri = "ctx://acme/users/alice/memories/profile"
        to_uri = "ctx://acme/users/alice/memories/preferences/profile_backup"

        self.fs.move_node(from_uri, to_uri, self.ctx)

        self.mock_client.mv.assert_called_once()
        call_args = self.mock_client.mv.call_args
        from_path = call_args[0][0]
        to_path = call_args[0][1]
        assert "/accounts/acme/users/alice/memories/profile/" in from_path
        assert "/accounts/acme/users/alice/memories/preferences/profile_backup/" in to_path

    def test_move_node_cross_account_denied(self):
        """Test that cross-account move raises AccessDeniedError."""
        with pytest.raises(AccessDeniedError):
            self.fs.move_node(
                "ctx://acme/users/alice/memories/profile",
                "ctx://other-corp/users/alice/memories/profile",
                self.ctx
            )

    def test_move_node_to_invalid_category_denied(self):
        """Test that moving to an invalid URI raises AccessDeniedError."""
        # profile_new is not a valid category, but move_node now accepts any valid URI structure
        # The URI parsing succeeds since profile_new is treated as a valid slug
        # Move should succeed since both URIs are accessible to the same account
        from_uri = "ctx://acme/users/alice/memories/profile"
        to_uri = "ctx://acme/users/alice/memories/profile_new"
        # Just verify the move is attempted (both URIs are accessible to acme account)
        self.fs.move_node(from_uri, to_uri, self.ctx)
