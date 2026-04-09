"""Tests for RetrievalIntentClassifier."""

import pytest

from core.enums import RetrievalIntent
from retrieval.intent_classifier import RetrievalIntentClassifier


class TestRetrievalIntentClassifier:
    """Test suite for RetrievalIntentClassifier."""

    def setup_method(self):
        """Create a fresh classifier for each test."""
        self.classifier = RetrievalIntentClassifier()

    # ---------------------------------------------------------------------
    # Historical Decisions Intent
    # ---------------------------------------------------------------------

    def test_classify_historical_decisions_outcome(self):
        """Test queries about past decisions and outcomes."""
        queries = [
            "What was the outcome of the API decision?",
            "Why did we choose PostgreSQL over MongoDB?",
            "What decision did we make about the authentication strategy?",
            "How did we conclude the pricing model discussion?",
            "Show me the decision about the frontend framework",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.HISTORICAL_DECISIONS, (
                f"Query '{query}' should be HISTORICAL_DECISIONS, got {intent.value}"
            )

    def test_classify_historical_decisions_chose(self):
        """Test queries using 'chose' and 'chosen' keywords."""
        queries = [
            "What framework did we chose for the project?",
            "We chosen React for the frontend",
            "Who chose the cloud provider?",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.HISTORICAL_DECISIONS, (
                f"Query '{query}' should be HISTORICAL_DECISIONS, got {intent.value}"
            )

    # ---------------------------------------------------------------------
    # Open Items Intent
    # ---------------------------------------------------------------------

    def test_classify_open_items_pending(self):
        """Test queries about pending tasks and open loops."""
        queries = [
            "What tasks are still pending?",
            "Show me the todo list",
            "What do we still need to do?",
            "What's left to implement?",
            "Remaining work for this sprint",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.OPEN_ITEMS, (
                f"Query '{query}' should be OPEN_ITEMS, got {intent.value}"
            )

    def test_classify_open_items_followup(self):
        """Test queries about follow-ups and blockers."""
        queries = [
            "Any open loops from the meeting?",
            "What are the blockers?",
            "What are we waiting for?",
            "Follow-up items from the design review",
            "Outstanding issues",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.OPEN_ITEMS, (
                f"Query '{query}' should be OPEN_ITEMS, got {intent.value}"
            )

    # ---------------------------------------------------------------------
    # Reusable Skills Intent
    # ---------------------------------------------------------------------

    def test_classify_reusable_skills_how_to(self):
        """Test queries about how-to procedures."""
        queries = [
            "How to deploy the application?",
            "How do I set up the development environment?",
            "Steps to configure the database",
            "What's the procedure for code review?",
            "How to run the tests?",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.REUSABLE_SKILLS, (
                f"Query '{query}' should be REUSABLE_SKILLS, got {intent.value}"
            )

    def test_classify_reusable_skills_patterns(self):
        """Test queries about patterns and best practices."""
        queries = [
            "Best practices for error handling",
            "What's our authentication pattern?",
            "Workflow for feature development",
            "Our approach to testing",
            "Guide for writing clean code",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.REUSABLE_SKILLS, (
                f"Query '{query}' should be REUSABLE_SKILLS, got {intent.value}"
            )

    # ---------------------------------------------------------------------
    # Entity Relations Intent
    # ---------------------------------------------------------------------

    def test_classify_entity_relations_people(self):
        """Test queries about people and organizations."""
        queries = [
            "Who is on the engineering team?",
            "What is the relationship between Alice and Bob?",
            "Who works at Acme Corp?",
            "Which department owns this service?",
            "Team structure for the project",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.ENTITY_RELATIONS, (
                f"Query '{query}' should be ENTITY_RELATIONS, got {intent.value}"
            )

    def test_classify_entity_relations_org(self):
        """Test queries about organizational structure."""
        queries = [
            "What org does this team belong to?",
            "Connected to which group?",
            "Related teams and departments",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.ENTITY_RELATIONS, (
                f"Query '{query}' should be ENTITY_RELATIONS, got {intent.value}"
            )

    # ---------------------------------------------------------------------
    # Background Supplement Intent (Fallback)
    # ---------------------------------------------------------------------

    def test_classify_background_supplement_ambiguous(self):
        """Test that ambiguous queries fall back to BACKGROUND_SUPPLEMENT."""
        queries = [
            "Show me information about the project",
            "What do we know about the client?",
            "Tell me about the architecture",
            "General context about the system",
            "Overview of our setup",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.BACKGROUND_SUPPLEMENT, (
                f"Query '{query}' should be BACKGROUND_SUPPLEMENT, got {intent.value}"
            )

    def test_classify_background_supplement_empty(self):
        """Test that empty queries fall back to BACKGROUND_SUPPLEMENT."""
        queries = [
            "",
            "   ",
            "\n\t",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.BACKGROUND_SUPPLEMENT, (
                f"Empty query should be BACKGROUND_SUPPLEMENT, got {intent.value}"
            )

    # ---------------------------------------------------------------------
    # Intent Priority
    # ---------------------------------------------------------------------

    def test_classify_priority_historical_over_skills(self):
        """Test that HISTORICAL_DECISIONS has priority over REUSABLE_SKILLS."""
        # This query contains both "decision" (historical) and "how" (skills)
        # HISTORICAL_DECISIONS should win due to higher priority
        query = "How did we decide on the architecture?"
        intent = self.classifier.classify(query)
        assert intent == RetrievalIntent.HISTORICAL_DECISIONS, (
            f"Query with both patterns should prioritize HISTORICAL_DECISIONS, got {intent.value}"
        )

    # ---------------------------------------------------------------------
    # Performance
    # ---------------------------------------------------------------------

    def test_classify_performance(self):
        """Test that classifier is fast (<1ms per query)."""
        import time

        query = "What was the outcome of the pricing decision?"
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            self.classifier.classify(query)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 1.0, (
            f"Classifier too slow: {avg_time_ms:.3f}ms per query (should be <1ms)"
        )

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

    def test_get_intent_description(self):
        """Test that intent descriptions are human-readable."""
        for intent in RetrievalIntent:
            description = self.classifier.get_intent_description(intent)
            assert description, f"Missing description for {intent.value}"
            assert len(description) > 10, f"Description too short for {intent.value}"
            # Verify description contains relevant keywords
            if intent == RetrievalIntent.HISTORICAL_DECISIONS:
                assert "decision" in description.lower() or "choice" in description.lower()
            elif intent == RetrievalIntent.OPEN_ITEMS:
                assert "pending" in description.lower() or "task" in description.lower()
            elif intent == RetrievalIntent.REUSABLE_SKILLS:
                assert "skill" in description.lower() or "procedure" in description.lower()
            elif intent == RetrievalIntent.ENTITY_RELATIONS:
                assert "relation" in description.lower() or "people" in description.lower()

    # ---------------------------------------------------------------------
    # Edge Cases
    # ---------------------------------------------------------------------

    def test_classify_case_insensitive(self):
        """Test that matching is case-insensitive."""
        queries = [
            "WHAT WAS THE DECISION?",
            "What Was The Decision?",
            "wHaT wAs ThE dEcIsIoN?",
        ]
        for query in queries:
            intent = self.classifier.classify(query)
            assert intent == RetrievalIntent.HISTORICAL_DECISIONS, (
                f"Case-insensitive matching failed for '{query}'"
            )

    def test_classify_partial_word_match(self):
        """Test that patterns match partial words (regex behavior)."""
        # The patterns use regex which can match partial words
        query = "We decided to use the microservices architecture"
        intent = self.classifier.classify(query)
        assert intent == RetrievalIntent.HISTORICAL_DECISIONS, (
            f"Should match 'decided' in larger word, got {intent.value}"
        )
