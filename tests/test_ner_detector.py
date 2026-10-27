
import pytest
from unittest.mock import Mock

from app.services.ner_detector import NERDetector


def test_empty_text_returns_no_entities():
    detector = NERDetector()

    assert detector.detect("") == []
    assert detector.detect("   ") == []


def test_detect_people_organizations_and_locations():
    detector = NERDetector()

    text = (
        "Rahul Sharma works at Google "
        "in Bengaluru."
    )

    entities = detector.detect(text)

    print("\nDetected entities:")

    for entity in entities:
        print(
            entity.text,
            entity.entity_type,
            round(entity.confidence, 3),
            entity.start,
            entity.end,
        )

    types = {entity.entity_type for entity in entities}

    assert "PER" in types
    assert any(
        entity.entity_type == "PER"
        and entity.text == "Rahul Sharma"
        for entity in entities
    )


def test_inference_error_propagates():
    detector = NERDetector()

    detector.__dict__["_pipeline"] = Mock(
        side_effect=RuntimeError("model inference failed")
    )

    with pytest.raises(
        RuntimeError,
        match="model inference failed",
    ):
        detector.detect("Rahul Sharma works at Google.")


def test_invalid_offsets_are_rejected():
    detector = NERDetector()

    detector.__dict__["_pipeline"] = Mock(
        return_value=[
            {
                "score": 0.99,
                "start": -1,
                "end": 5,
                "entity_group": "PER",
            },
            {
                "score": 0.99,
                "start": 2,
                "end": 100,
                "entity_group": "PER",
            },
            {
                "score": 0.99,
                "start": 5,
                "end": 5,
                "entity_group": "PER",
            },
        ]
    )

    assert detector.detect("Rahul Sharma") == []
