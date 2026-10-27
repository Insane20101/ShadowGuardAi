
from typing import cast

from app.security.sanitizer import Sanitizer, restore
from app.services.ner_detector import NERDetector


class FakeEntity:
    def __init__(
        self,
        text: str,
        entity_type: str,
        start: int,
        end: int,
        confidence: float = 0.99,
    ):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.confidence = confidence


class FakeNERDetector:
    """Deterministic test double; doesn't download or load BERT."""

    def detect(self, text: str) -> list[FakeEntity]:
        entities = []

        for name, label in [
            ("Rahul Sharma", "PER"),
            ("Google", "ORG"),
            ("Bengaluru", "LOC"),
        ]:
            start = text.find(name)

            if start != -1:
                entities.append(
                    FakeEntity(
                        name,
                        label,
                        start,
                        start + len(name),
                    )
                )

        return entities


def test_combines_regex_and_ner():
    text = (
        "Rahul Sharma works at Google in Bengaluru. "
        "Email him at rahul@example.com."
    )

    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )
    result = sanitizer.sanitize(text)

    assert "[PERSON_1]" in result.sanitized_text
    assert "[ORG_1]" in result.sanitized_text
    assert "[LOCATION_1]" in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text

    assert "rahul@example.com" not in result.sanitized_text
    assert "Rahul Sharma" not in result.sanitized_text

    assert result.mapping["[PERSON_1]"] == "Rahul Sharma"
    assert result.mapping["[EMAIL_1]"] == "rahul@example.com"


def test_restore_recovers_original_text():
    text = "Rahul Sharma uses rahul@example.com."

    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )
    result = sanitizer.sanitize(text)

    restored = restore(result.sanitized_text, result.mapping)

    assert restored == text


def test_regex_pii_takes_priority_over_ner_overlap():
    class OverlappingNER:
        def detect(self, text: str) -> list[FakeEntity]:
            start = text.find("rahul@example.com")

            if start == -1:
                return []

            return [
                FakeEntity(
                    "rahul@example.com",
                    "PER",
                    start,
                    start + len("rahul@example.com"),
                )
            ]

    text = "Contact rahul@example.com"

    result = Sanitizer(
        ner_detector=cast(NERDetector, OverlappingNER())
    ).sanitize(text)

    assert result.sanitized_text == "Contact [EMAIL_1]"
    assert "[PERSON_1]" not in result.sanitized_text


def test_repeated_email_uses_same_token():
    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )

    result = sanitizer.sanitize(
        "Contact alice@example.com or alice@example.com"
    )

    assert result.sanitized_text.count("[EMAIL_1]") == 2
    assert result.entity_counts["EMAIL"] == 1
    assert result.mapping["[EMAIL_1]"] == "alice@example.com"


def test_invalid_credit_card_is_not_masked():
    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )

    text = "Reference number 1234567890123"
    result = sanitizer.sanitize(text)

    assert result.sanitized_text == text
    assert "CREDIT_CARD" not in result.entity_counts


def test_normal_text_remains_unchanged():
    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )

    text = "The meeting is at 10 AM in room 204."
    result = sanitizer.sanitize(text)

    assert result.sanitized_text == text
    assert result.entity_counts == {}


def test_mapping_is_not_in_sanitized_text():
    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )

    result = sanitizer.sanitize("Email alice@example.com")

    assert "[EMAIL_1]" in result.sanitized_text
    assert "alice@example.com" not in result.sanitized_text
    assert result.mapping["[EMAIL_1]"] == "alice@example.com"


def test_existing_placeholder_does_not_collide_with_generated_token():
    sanitizer = Sanitizer(
        ner_detector=cast(NERDetector, FakeNERDetector())
    )

    text = "Literal [EMAIL_1] is written here; contact alice@example.com."
    result = sanitizer.sanitize(text)

    # The placeholder already present in the input must remain literal.
    assert "[EMAIL_1] is written here" in result.sanitized_text

    # The actual email must receive a different placeholder.
    assert "[EMAIL_2]" in result.sanitized_text
    assert result.mapping["[EMAIL_2]"] == "alice@example.com"
    assert "[EMAIL_1]" not in result.mapping

    # Restoring the email must not alter the original literal placeholder.
    restored = restore(result.sanitized_text, result.mapping)

    assert restored == text






def test_restore_does_not_replace_tokens_inside_restored_values():
    mapping = {
        "[EMAIL_1]": "user@example.com",
        "[PERSON_1]": "Alice [EMAIL_1]",
    }

    restored = restore(
        "Contact [PERSON_1] and [EMAIL_1]",
        mapping,
    )

    assert restored == (
        "Contact Alice [EMAIL_1] and user@example.com"
    )
