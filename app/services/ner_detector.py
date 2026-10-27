
import os
from dataclasses import dataclass
from functools import cached_property

from transformers import pipeline


@dataclass(frozen=True)
class DetectedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float


class NERDetector:
    """
    Detect named entities using a local Hugging Face NER model.

    Model inference runs locally after the model has been downloaded.
    Raw prompts are not sent to an inference API by this detector.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: int = -1,
        min_confidence: float = 0.80,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")

        self.model_id = (
            model_id or os.getenv("HF_MODEL_ID", "dslim/bert-base-NER")
        )
        self.device = device
        self.min_confidence = min_confidence

    @cached_property
    def _pipeline(self):
        """Load the model once, on first use."""
        token = os.getenv("HF_TOKEN") or None

        return pipeline(
            task="token-classification",
            model=self.model_id,
            aggregation_strategy="simple",
            device=self.device,
            token=token,
        )

    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect entities and merge contiguous same-type predictions."""
        if not text or not text.strip():
            return []

        predictions = self._pipeline(text)
        entities: list[DetectedEntity] = []

        for item in predictions:
            try:
                score = float(item["score"])
                start = int(item["start"])
                end = int(item["end"])
                entity_type = str(item["entity_group"])
            except (KeyError, TypeError, ValueError, OverflowError):
                # Ignore malformed predictions rather than creating
                # entities from incomplete model output.
                continue

            if not 0.0 <= score <= 1.0:
                continue

            if score < self.min_confidence:
                continue

            # Reject invalid or empty spans before slicing the input.
            if start < 0 or end > len(text) or start >= end:
                continue

            if not entity_type or entity_type == "None":
                continue

            entities.append(
                DetectedEntity(
                    text=text[start:end],
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    confidence=score,
                )
            )

        # Merge subword fragments whose spans touch exactly.
        merged: list[DetectedEntity] = []

        for entity in entities:
            if (
                merged
                and merged[-1].entity_type == entity.entity_type
                and merged[-1].end == entity.start
            ):
                previous = merged[-1]

                merged[-1] = DetectedEntity(
                    text=text[previous.start:entity.end],
                    entity_type=previous.entity_type,
                    start=previous.start,
                    end=entity.end,
                    confidence=min(
                        previous.confidence,
                        entity.confidence,
                    ),
                )
            else:
                merged.append(entity)

        return merged
