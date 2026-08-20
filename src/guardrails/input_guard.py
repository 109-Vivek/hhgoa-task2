import re
import time
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class GuardrailCheckResult:
    is_safe: bool
    reason: str
    action: str  # "allow", "block", "reprompt"
    latency_ms: float


class InputGuardrail:
    """
    Ultra-low latency Input Guardrail that validates queries before retrieval:
    1. Toxicity / Abusive content patterns.
    2. Prompt injection / jailbreak patterns.
    3. Minimum acoustic confidence check.
    4. Off-topic gibberish or empty audio transcripts.
    """

    PROMPT_INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard (all )?prior prompts",
        r"you are now (dan|unfiltered|jailbroken)",
        r"system prompt:",
        r"<\|im_start\|>",
        r"override safety",
    ]

    UNSAFE_PATTERNS = [
        r"\b(how to build a bomb|make explosives|steal credit card)\b",
        r"\b(hack into|ddos attack|sql injection)\b",
    ]

    def __init__(self, min_transcript_len: int = 2, min_confidence: float = 0.35):
        self.min_transcript_len = min_transcript_len
        self.min_confidence = min_confidence

    def evaluate(self, query: str, stt_confidence: float = 1.0) -> GuardrailCheckResult:
        start_time = time.perf_counter()

        if not query or len(query.strip()) < self.min_transcript_len:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GuardrailCheckResult(
                is_safe=False,
                reason="Query is empty or audio transcript is too short.",
                action="reprompt",
                latency_ms=elapsed_ms,
            )

        if stt_confidence < self.min_confidence:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GuardrailCheckResult(
                is_safe=False,
                reason=f"Acoustic confidence ({stt_confidence:.2f}) below threshold ({self.min_confidence}).",
                action="reprompt",
                latency_ms=elapsed_ms,
            )

        # Check prompt injections
        lower_query = query.lower()
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lower_query):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return GuardrailCheckResult(
                    is_safe=False,
                    reason="Potential prompt injection or jailbreak detected.",
                    action="block",
                    latency_ms=elapsed_ms,
                )

        # Check safety/harmful content
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, lower_query):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return GuardrailCheckResult(
                    is_safe=False,
                    reason="Query violated safety policy.",
                    action="block",
                    latency_ms=elapsed_ms,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return GuardrailCheckResult(
            is_safe=True,
            reason="Query passed all input guardrails.",
            action="allow",
            latency_ms=elapsed_ms,
        )
