
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
