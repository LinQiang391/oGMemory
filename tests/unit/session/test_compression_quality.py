from session.compression_quality import CompressionQualityEvaluator


def test_quality_metrics_expose_prd_level_fields_by_default():
    messages = [
        {"role": "user", "content": "Use PostgreSQL and AGFS for Project Falcon."},
        {"role": "assistant", "content": "We will keep ContextFS compatibility."},
    ]

    metrics = CompressionQualityEvaluator().evaluate(
        messages,
        overview="Project Falcon uses PostgreSQL, AGFS, and ContextFS.",
        abstract="Project Falcon storage plan",
    )

    assert set(metrics) == {
        "information_retention_ratio",
        "entity_retention_ratio",
        "compression_ratio",
    }
    assert metrics["information_retention_ratio"] == 1.0
    assert metrics["entity_retention_ratio"] == 1.0
    assert 0 < metrics["compression_ratio"] < 1


def test_quality_metrics_include_diagnostics_when_requested():
    metrics = CompressionQualityEvaluator().evaluate(
        [{"role": "user", "content": "Use PostgreSQL for Project Falcon."}],
        overview="Project Falcon",
        abstract="Project Falcon",
        include_details=True,
    )

    assert "details" in metrics
    assert "source_chars" in metrics["details"]
    assert "missing_entities" in metrics["details"]


def test_quality_metrics_handles_empty_source():
    metrics = CompressionQualityEvaluator().evaluate([], overview="", abstract="")

    assert metrics["information_retention_ratio"] == 1.0
    assert metrics["entity_retention_ratio"] == 1.0
    assert metrics["compression_ratio"] == 1.0


def test_quality_metrics_ignore_common_title_case_words_as_entities():
    metrics = CompressionQualityEvaluator().evaluate(
        [{"role": "user", "content": "The API uses Project Falcon. Yes, keep PostgreSQL."}],
        overview="The API uses Project Falcon and PostgreSQL.",
        abstract="API Project Falcon PostgreSQL",
        include_details=True,
    )

    assert "The" not in metrics["details"]["missing_entities"]
    assert "Yes" not in metrics["details"]["missing_entities"]
    assert metrics["entity_retention_ratio"] == 1.0
