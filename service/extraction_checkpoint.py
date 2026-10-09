"""Durable source batches and accepted candidates for extraction recovery."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace

from commit.write_receipts import operation_digest
from core.models import CandidateMemory

EXTENSION_KEY = "memory_extraction_recovery"


class RecoveryNeedsConfirmationError(RuntimeError):
    """No durable evidence distinguishes an uncommitted write from deletion."""


class RecoveryPersistenceError(RuntimeError):
    """A checkpoint was not durably acknowledged; no new write is allowed."""


def recovery_owner(ctx):
    return {"account_id": ctx.account_id, "user_id": ctx.user_id,
            "agent_id": ctx.agent_id, "session_id": ctx.session_id}


def request_message_id(ctx, request_id, index):
    payload = {"owner": recovery_owner(ctx), "request_id": request_id, "index": index}
    return "msg_request_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExtractionBatchJournal:
    def __init__(self, manager, buffer, ctx, snapshot, *, archive_id,
                 messages, session_time, session_summary, tool_stats_text, client_request_id="",
                 request_messages=None, lifecycle=None):
        self.manager, self.buffer, self.ctx = manager, buffer, ctx
        extension = buffer.runtime_extensions.get(EXTENSION_KEY, {})
        batch = extension.get("batch")
        if batch is not None:
            if (extension.get("version") != 1 or not isinstance(batch, dict)
                    or batch.get("owner") != recovery_owner(ctx)):
                raise RecoveryNeedsConfirmationError("pending extraction batch has a different owner or format")
            actual = [asdict(message) for message in snapshot]
            if actual != batch.get("snapshot"):
                raise RecoveryNeedsConfirmationError("pending extraction source snapshot changed")
            stored_lifecycle = batch.get("lifecycle")
            if ("lifecycle" in batch and (not isinstance(stored_lifecycle, dict)
                    or stored_lifecycle.get("kind") not in {"compaction", "dispose"})):
                raise RecoveryNeedsConfirmationError("pending extraction lifecycle is invalid")
            if ((stored_lifecycle or {}).get("kind") != (lifecycle or {}).get("kind")):
                raise RecoveryNeedsConfirmationError("pending extraction batch belongs to another lifecycle")
        else:
            batch = {"owner": recovery_owner(ctx), "archive_id": archive_id,
                     "client_request_id": client_request_id,
                     "request_messages": deepcopy(request_messages or []),
                     "message_ids": [message.id for message in snapshot],
                     "snapshot": [asdict(message) for message in snapshot],
                     "messages": deepcopy(messages),
                     "session_time": session_time.isoformat() if hasattr(session_time, "isoformat") else session_time,
                     "session_summary": session_summary, "tool_stats_text": tool_stats_text,
                     "chunks": None, "slots": {}}
            if lifecycle is not None:
                batch["lifecycle"] = deepcopy(lifecycle)
            self._replace(batch)

    @property
    def batch(self):
        return self.buffer.runtime_extensions[EXTENSION_KEY]["batch"]

    def _replace(self, batch):
        with self.buffer.session_state_save_lock:
            with self.buffer.extraction_lock:
                prior = deepcopy(self.buffer.runtime_extensions.get(EXTENSION_KEY))
                if batch is None:
                    self.buffer.runtime_extensions.pop(EXTENSION_KEY, None)
                else:
                    self.buffer.runtime_extensions[EXTENSION_KEY] = {"version": 1, "batch": deepcopy(batch)}
            if not self.manager.save_session_state(self.buffer.session_id, self.ctx):
                with self.buffer.extraction_lock:
                    if prior is None:
                        self.buffer.runtime_extensions.pop(EXTENSION_KEY, None)
                    else:
                        self.buffer.runtime_extensions[EXTENSION_KEY] = prior
                raise RecoveryPersistenceError("extraction checkpoint could not be saved")

    def mutate(self, update):
        batch = deepcopy(self.batch)
        update(batch)
        self._replace(batch)

    def chunks(self, factory):
        chunks = self.batch.get("chunks")
        if chunks is None:
            chunks = factory()
            self.mutate(lambda batch: batch.update(chunks=deepcopy(chunks)))
        return deepcopy(chunks)

    def checkpoint(self, slot):
        return CandidateCheckpoint(self, str(slot))

    def finish(self, *, reset_window_state=False):
        """Clear source claim, pending drafts and request ID in one durable state."""
        with self.buffer.session_state_save_lock:
            with self.buffer.extraction_lock:
                rollback = self.buffer.capture_rollback_state()
                old_completed = list(self.buffer.completed_request_ids)
                old_extension = deepcopy(self.buffer.runtime_extensions[EXTENSION_KEY])
                client_request_id = self.batch.get("client_request_id", "")
                self.buffer.remove_messages_by_id(set(self.batch["message_ids"]))
                if (self.batch.get("lifecycle", {}).get("kind") == "compaction"
                        and self.buffer.compaction_prepare_archive_id == self.batch["archive_id"]):
                    self.buffer.compaction_prepare_archive_id = ""
                if reset_window_state and not self.buffer.messages:
                    from session.models import SessionWindowState
                    self.buffer.window_state = SessionWindowState()
                if client_request_id:
                    self.buffer.mark_request_completed(client_request_id)
                self.buffer.runtime_extensions.pop(EXTENSION_KEY)
            if not self.manager.save_session_state(self.buffer.session_id, self.ctx):
                with self.buffer.extraction_lock:
                    self.buffer.restore_rollback_state(rollback)
                    self.buffer.completed_request_ids = old_completed
                    self.buffer.runtime_extensions[EXTENSION_KEY] = old_extension
                raise RecoveryPersistenceError("completed extraction checkpoint could not be saved")


class CandidateCheckpoint:
    def __init__(self, journal, slot):
        self.journal, self.slot = journal, slot

    @property
    def record(self):
        return self.journal.batch["slots"].get(self.slot)

    def load(self):
        record = self.record
        if record is None:
            return None
        try:
            candidates = [CandidateMemory(**value) for value in record["candidates"]]
            for index, candidate in enumerate(candidates):
                expected = hashlib.sha256(
                    f"{self.journal.batch['archive_id']}:{self.slot}:{index}".encode()).hexdigest()
                if candidate.write_operation_id != expected:
                    raise ValueError("operation identity changed")
                if operation_digest(candidate, self.journal.ctx) != record["candidate_digests"][index]:
                    raise ValueError("accepted operation payload changed")
            return candidates, record["extracted"], record["filtered"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryNeedsConfirmationError("accepted extraction checkpoint is invalid") from exc

    def accept(self, candidates, extracted, filtered):
        if self.record is not None:
            raise RecoveryNeedsConfirmationError("accepted extraction candidates cannot be replaced")
        batch = self.journal.batch
        accepted = [replace(candidate, write_operation_id=hashlib.sha256(
            f"{batch['archive_id']}:{self.slot}:{index}".encode()).hexdigest())
                    for index, candidate in enumerate(candidates)]
        record = {"candidates": [asdict(candidate) for candidate in accepted],
                  "candidate_digests": [operation_digest(candidate, self.journal.ctx) for candidate in accepted],
                  "extracted": extracted, "filtered": filtered, "operations": {}}
        self.journal.mutate(lambda value: value["slots"].update({self.slot: record}))
        return accepted

    def _operation(self, index):
        return self.record["operations"].get(str(index), {})

    def completed(self, index):
        operation = self._operation(index)
        return deepcopy(operation["plan"]) if operation.get("status") == "complete" else None

    def outcome_uncertain(self, index):
        return self._operation(index).get("status") == "writing"

    def _set_operation(self, index, operation):
        self.journal.mutate(lambda batch: batch["slots"][self.slot]["operations"].update(
            {str(index): operation}))

    def begin(self, index):
        self._set_operation(index, {"status": "writing"})

    def reject(self, index, error):
        self._set_operation(index, {"status": "rejected_before_write", "error": error})

    def complete(self, index, plan):
        # Recovery needs the acknowledged action and target, not another copy of
        # full memory bodies and accumulated identity metadata in session state.
        saved = {"action": plan.action, "target_uri": plan.target_uri,
                 "merged_fields": {"status": plan.merged_fields["status"]}
                 if "status" in plan.merged_fields else {}, "relation_edges": []}
        self._set_operation(index, {"status": "complete", "plan": saved})
