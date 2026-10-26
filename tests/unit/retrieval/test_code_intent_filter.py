from __future__ import annotations

import pytest

from retrieval.code_intent_filter import classify_code_retrieval_intent

POSITIVE_SAMPLES = [
    "Find where add is implemented, fix the failing test, run tests.",
    "Where is AuthService.login defined?",
    "Open src/auth/service.py and inspect parse_token.",
    "The stack trace points to ValueError in config.parse_config_line.",
    "Locate slugify implementation, make it use dashes, run tests.",
    "Search references to PaymentService.charge and report the callers.",
    "Go to definition for UserRepository.create_user.",
    "Show all files under src/parser that mention token_stream.",
    "What does parse_token return?",
    "How is apply_discount computed in billing.py?",
    "Where is token_stream used in production code?",
    "How many callers does add have?",
]

NEGATIVE_SAMPLES = [
    "Remember that project alpha uses PostgreSQL. What database does it use?",
    "Remember that project beta stores feature flags in LaunchDarkly. Where are they stored?",
    "Remember that release policy requires two approvals. How many approvals are required?",
    "Remember that the frontend uses React Query. What library does it use?",
    "When did Caroline go to the LGBTQ support group?",
    "What is Caroline's identity?",
    "Where did Caroline move from 4 years ago?",
    "What books has Melanie read?",
    "Would Caroline likely have Dr. Seuss books on her bookshelf?",
    "Who supports Caroline when she has a negative experience?",
]


@pytest.mark.parametrize("query", POSITIVE_SAMPLES)
def test_positive_samples_trigger_code_intent(query):
    decision = classify_code_retrieval_intent(query)

    assert decision.is_code is True
    assert decision.reasons
    assert decision.blocked_by == ()


@pytest.mark.parametrize("query", NEGATIVE_SAMPLES)
def test_negative_samples_do_not_trigger_code_intent(query):
    decision = classify_code_retrieval_intent(query)

    assert decision.is_code is False
    assert decision.blocked_by
