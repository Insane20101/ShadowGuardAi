
import re
from dataclasses import dataclass, field

from app.services.ner_detector import NERDetector


@dataclass
class SanitizationResult:
    sanitized_text: str
    entity_counts: dict[str, int]
    # Hidden from repr so original PII isn't printed accidentally.
    mapping: dict[str, str] = field(repr=False)


PATTERNS = [
    (
        "API_KEY",
        r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b",
    ),
    (
        "EMAIL",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ),
    (
        "CREDIT_CARD",
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
    ),
    (
        "PHONE",
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
    ),
]

COMBINED_PATTERN = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in PATTERNS)
)

# Map NER model labels to readable token labels.
NER_LABELS = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
}


def looks_like_credit_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)

    if not 13 <= len(digits) <= 19:
        return False

    total = 0

    for index, char in enumerate(reversed(digits)):
        digit = int(char)

        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9

        total += digit

    return total % 10 == 0


class Sanitizer:
    def __init__(self, ner_detector: NERDetector | None = None):
        # Lazy model loading inside NERDetector means importing this
        # module does not immediately download or load the model.
        self.ner_detector = ner_detector or NERDetector()

    def sanitize(self, text: str) -> SanitizationResult:
        mapping: dict[str, str] = {}
        counts: dict[str, int] = {}
        replacements: dict[tuple[str, str], str] = {}

        # First detect regex-based PII and record protected spans.
        regex_matches = []

        for match in COMBINED_PATTERN.finditer(text):
            entity_type = match.lastgroup
            if entity_type is None:
                continue

            original = match.group()

            if entity_type == "CREDIT_CARD":
                if not looks_like_credit_card(original):
                    continue

            elif entity_type == "PHONE":
                digits = re.sub(r"\D", "", original)
                if not 8 <= len(digits) <= 15:
                    continue

            regex_matches.append(
                (match.start(), match.end(), entity_type, original)
            )

        # Generate stable placeholders for repeated values.
        def make_token(entity_type: str, original: str) -> str:
            cache_key = (entity_type, original)

            if cache_key not in replacements:
                counts[entity_type] = counts.get(entity_type, 0) + 1
                token = f"[{entity_type}_{counts[entity_type]}]"
                replacements[cache_key] = token
                mapping[token] = original

            return replacements[cache_key]

        # Mask regex matches first. NER should not overwrite these spans.
        protected_spans = []
        for start, end, entity_type, original in regex_matches:
            token = make_token(entity_type, original)
            protected_spans.append((start, end, token))

        # Detect names, organizations, and locations.
        ner_matches = []

        if text.strip():
            for entity in self.ner_detector.detect(text):
                label = NER_LABELS.get(entity.entity_type)

                if label is None:
                    continue

                start, end = entity.start, entity.end

                # Ignore malformed offsets.
                if start < 0 or end > len(text) or start >= end:
                    continue

                # Regex PII has priority: skip any overlapping NER span.
                overlaps_regex = any(
                    start < protected_end and end > protected_start
                    for protected_start, protected_end, _ in protected_spans
                )

                if overlaps_regex:
                    continue

                ner_matches.append(
                    (start, end, label, text[start:end])
                )

        # Resolve any overlapping NER spans, preferring the longer span.
        ner_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted_ner = []

        for candidate in ner_matches:
            start, end, _, _ = candidate

            if any(
                start < accepted_end and end > accepted_start
                for accepted_start, accepted_end, _, _ in accepted_ner
            ):
                continue

            accepted_ner.append(candidate)

        # Replace spans from right to left so offsets remain valid.
        all_matches = []

        for start, end, token in protected_spans:
            all_matches.append((start, end, token))

        for start, end, entity_type, original in accepted_ner:
            token = make_token(entity_type, original)
            all_matches.append((start, end, token))

        sanitized_text = text

        for start, end, token in sorted(
            all_matches, key=lambda item: item[0], reverse=True
        ):
            sanitized_text = (
                sanitized_text[:start] + token + sanitized_text[end:]
            )

        return SanitizationResult(
            sanitized_text=sanitized_text,
            entity_counts=counts,
            mapping=mapping,
        )


def sanitize(text: str) -> SanitizationResult:
    """Backward-compatible helper for existing callers."""
    return Sanitizer().sanitize(text)


def restore(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)

    return text
