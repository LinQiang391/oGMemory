"""Compose bounded scene decisions over a canonical evidence snapshot."""

from __future__ import annotations

from core.interfaces import LLM
from core.models import ContextNode
from scene.config import SceneConfig
from scene.decision_application import SceneDecisionApplier
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import DecisionLimits, SceneCallBudgetError, SceneDecisionResult
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_preflight import ScenePreflight
from scene.decision_proposal import SceneProposalStage
from scene.decision_sources import node_source_refs
from scene.models import SceneSnapshot


class SceneDecisionEngine:
    """Run preflight, proposal, quality review and versioned snapshot projection."""

    def __init__(
        self, llm: LLM, config: SceneConfig | None = None, *, limits: DecisionLimits | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SceneConfig()
        self.limits = limits or DecisionLimits()
        self._evidence = SceneEvidenceBuilder(self.config, self.limits)
        caller = SceneModelCaller(llm, self.limits)
        self._preflight = ScenePreflight(caller, self.config)
        self._proposal = SceneProposalStage(caller, self.config)
        self._applier = SceneDecisionApplier(self.config, self.limits)

    def for_config(self, config: SceneConfig) -> SceneDecisionEngine:
        """Bind another configuration while sharing the model and input limits."""
        return SceneDecisionEngine(self.llm, config, limits=self.limits)

    def decide(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode], embeddings: dict[str, list[float]],
    ) -> SceneSnapshot:
        return self.decide_with_result(snapshot, nodes, embeddings).snapshot

    def decide_with_result(
        self, snapshot: SceneSnapshot, nodes: list[ContextNode], embeddings: dict[str, list[float]],
    ) -> SceneDecisionResult:
        """Return a detached projection and bounded diagnostics for the next batch."""
        self._evidence.validate_input(snapshot, nodes, embeddings)
        if not nodes:
            return SceneDecisionResult(snapshot, "WAIT", reason="empty input")
        known = {node.uri for node in nodes}
        if len(nodes) < self.config.min_evidence_nodes:
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(known)),
                reason="evidence count below configured threshold",
            )

        packet = self._evidence.build(snapshot, nodes, embeddings)
        canonical = {node.uri: node for node in nodes}
        for card in packet["memories"]:
            card["source_refs"] = list(node_source_refs(canonical[card["uri"]]))
        budget = CallBudget(self.config.max_llm_calls_per_batch)
        offered = set(packet["offered_member_uris"])
        preflight = self._preflight.enforce_gate(packet, self._preflight.run(packet, budget))
        if preflight.status == "WAIT":
            return SceneDecisionResult(
                snapshot, "WAIT", calls=budget.used,
                pending_member_uris=tuple(sorted(offered | (set(preflight.evidence_uris) & known))),
                reason=preflight.reason or "; ".join(preflight.missing),
            )

        decision = self._proposal.decide(snapshot, packet, budget)
        if decision.action == "WAIT":
            return SceneDecisionResult(
                snapshot, "WAIT", calls=budget.used,
                pending_member_uris=tuple(sorted(offered | (set(decision.unassigned_member_uris) & known))),
                reason=decision.reason or "; ".join(decision.conflicts),
            )

        floor = self._proposal.deterministic_quality_gate(decision, packet)
        if floor.decision == "DEFER":
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(offered)),
                calls=budget.used, reason=floor.reason,
            )
        if self.config.max_llm_calls_per_batch == 3:
            if budget.remaining == 0:
                raise SceneCallBudgetError("quality")
            gate = self._proposal.review(decision, packet, budget)
        else:
            gate = floor
        if gate.decision == "DEFER":
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=tuple(sorted(offered)),
                calls=budget.used, reason=gate.reason,
            )
        if gate.decision == "ADJUST" and decision.draft is not None:
            decision = decision.model_copy(update={
                "draft": decision.draft.model_copy(update={
                    "title": gate.title or decision.draft.title,
                    "description": gate.description or decision.draft.description,
                }),
            })
        replacement = self._applier.apply(snapshot, nodes, embeddings, decision)
        pending = tuple(sorted(set(decision.unassigned_member_uris)))
        if replacement == snapshot:
            return SceneDecisionResult(
                snapshot, "WAIT", pending_member_uris=pending,
                calls=budget.used, reason="no projection change",
            )
        return SceneDecisionResult(
            replacement, decision.action, pending_member_uris=pending, calls=budget.used,
        )
