
from app.security.sanitizer import Sanitizer, restore
from unittest.mock import Mock



def sanitize_without_ner(text: str):
    detector = Mock()
    detector.detect.return_value = []

    sanitizer = Sanitizer(ner_detector=detector)
    return sanitizer.sanitize(text)



def test_email_masking():
    result = sanitize_without_ner("Contact rahul@example.com")

    assert "rahul@example.com" not in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text


def test_repeated_email():
    result = sanitize_without_ner(
        "Email rahul@example.com or rahul@example.com"
    )

    assert result.sanitized_text.count("[EMAIL_1]") == 2
    assert len(result.mapping) == 1


def test_restore():
    result = sanitize_without_ner("Contact rahul@example.com")

    answer = restore(
        f"Hello, {result.sanitized_text}",
        result.mapping,
    )

    assert "rahul@example.com" in answer


def test_invalid_card_is_not_masked():
    result = sanitize_without_ner("Reference: 1234567890123456")

    assert "1234567890123456" in result.sanitized_text


def test_normal_text_is_unchanged():
    text = "Explain Python dictionaries."

    result = sanitize_without_ner(text)

    assert result.sanitized_text == text
    assert result.mapping == {}







def test_multiple_emails_get_different_placeholders():
    result = sanitize_without_ner(
        "Contact rahul@example.com or priya@example.com"
    )

    assert "rahul@example.com" not in result.sanitized_text
    assert "priya@example.com" not in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text
    assert "[EMAIL_2]" in result.sanitized_text
    assert len(result.mapping) == 2


def test_email_with_uppercase_characters():
    result = sanitize_without_ner("Contact Rahul@Example.com")

    assert "Rahul@Example.com" not in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text


def test_email_at_end_of_sentence():
    result = sanitize_without_ner("Contact rahul@example.com.")

    assert "rahul@example.com" not in result.sanitized_text
    assert result.sanitized_text.endswith(".")


def test_restore_does_not_change_unrelated_text():
    result = sanitize_without_ner("Contact rahul@example.com")

    answer = restore(
        f"Please email [EMAIL_1].",
        result.mapping,
    )

    assert answer == "Please email rahul@example.com."


def test_empty_string():
    result = sanitize_without_ner("")

    assert result.sanitized_text == ""
    assert result.mapping == {}



def test_existing_placeholder_does_not_collide():
    result = sanitize_without_ner(
        "Existing [EMAIL_1]; contact rahul@example.com"
    )

    assert "[EMAIL_1]" in result.sanitized_text
    assert "[EMAIL_2]" in result.sanitized_text
    assert len(result.mapping) == 1


def test_invalid_card_number_remains_unchanged():
    text = "Reference: 1234567890123456"

    result = sanitize_without_ner(text)

    assert result.sanitized_text == text