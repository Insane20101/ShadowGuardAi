
from app.security.sanitizer import sanitize, restore


def test_email_masking():
    result = sanitize("Contact rahul@example.com")

    assert "rahul@example.com" not in result.sanitized_text
    assert "[EMAIL_1]" in result.sanitized_text


def test_repeated_email():
    result = sanitize(
        "Email rahul@example.com or rahul@example.com"
    )

    assert result.sanitized_text.count("[EMAIL_1]") == 2
    assert len(result.mapping) == 1


def test_restore():
    result = sanitize("Contact rahul@example.com")

    answer = restore(
        f"Hello, {result.sanitized_text}",
        result.mapping,
    )

    assert "rahul@example.com" in answer


def test_invalid_card_is_not_masked():
    result = sanitize("Reference: 1234567890123456")

    assert "1234567890123456" in result.sanitized_text


def test_normal_text_is_unchanged():
    text = "Explain Python dictionaries."

    result = sanitize(text)

    assert result.sanitized_text == text
    assert result.mapping == {}
