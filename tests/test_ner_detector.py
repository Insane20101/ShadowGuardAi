
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







def test_low_confidence_entities_are_filtered():
    detector = NERDetector(min_confidence=0.80)
    detector.__dict__["_pipeline"] = Mock(return_value=[
        {"score": 0.79, "start": 0, "end": 5, "entity_group": "PER"},
        {"score": 0.95, "start": 6, "end": 12, "entity_group": "ORG"},
    ])

    entities = detector.detect("Rahul Google")

    assert len(entities) == 1
    assert entities[0].text == "Google"
    assert entities[0].entity_type == "ORG"


def test_invalid_confidence_values_are_rejected():
    detector = NERDetector()
    detector.__dict__["_pipeline"] = Mock(return_value=[
        {"score": float("nan"), "start": 0, "end": 5, "entity_group": "PER"},
        {"score": float("inf"), "start": 0, "end": 5, "entity_group": "PER"},
        {"score": float("-inf"), "start": 0, "end": 5, "entity_group": "PER"},
        {"score": 1.5, "start": 6, "end": 12, "entity_group": "ORG"},
    ])

    assert detector.detect("Rahul Google") == []

def test_detected_entity_offsets_match_original_text():
    detector = NERDetector()
    text = "Contact Rahul Sharma today."
    detector.__dict__["_pipeline"] = Mock(return_value=[
        {"score": 0.99, "start": 8, "end": 20, "entity_group": "PER"},
    ])

    entities = detector.detect(text)

    assert len(entities) == 1
    assert entities[0].text == "Rahul Sharma"
    assert text[entities[0].start:entities[0].end] == entities[0].text
