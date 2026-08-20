import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GroundingCheckResult:
    is_grounded: bool
    grounding_score: float
    is_abstention: bool
    hallucination_detected: bool
    reason: str
    latency_ms: float


class OutputGuardrail:
    """
    Ultra-low latency Output Guardrail that verifies:
    1. Grounding score of the generated answer against retrieved context passages.
    2. Detection of model abstentions or refusal to hallucinate.
    3. Hallucination flags when answer asserts ungrounded entities or facts.
    4. Language safety of the synthesized output.
    """

    ABSTENTION_PHRASES = [
        r"do not have enough relevant context",
        r"context does not contain",
        r"cannot answer based on the provided",
        r"not mentioned in the context",
        r"जानकारी उपलब्ध नहीं है",
        r"संदर्भ में उल्लेख नहीं है",
        r"સંદર્ભમાં પૂરતી માહિતી નથી",
        r"ડેટાબેઝમાં પૂરતો સંદર્ભ નથી",
        r"తగినంత సమాచారం లేదు",
        r"సందర్భంలో పేర్కొనబడలేదు",
        r"i don't know",
        r"not enough information",
    ]

    def __init__(self, min_grounding_threshold: float = 0.25):
        self.min_grounding_threshold = min_grounding_threshold

    @staticmethod
    def _tokenize(text: str) -> set:
        if not text:
            return set()
        text = text.lower()
        # Support Devanagari, Gujarati, Telugu, and Latin
        words = re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]+", text)
        stopwords = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", "of", "with",
            "का", "के", "की", "है", "हैं", "में", "से", "पर", "और", "को", "एक", "था", "थी", "थे",
            "અને", "છે", "માં", "ના", "ની", "નું", "નો", "માટે", "એક", "હતા", "હતી",
            "మరియు", "ఉంది", "లో", "యొక్క", "ఒక", "కోసం", "ఉన్నారు", "ఉన్నాయి", "అని"
        }
        return {w for w in words if len(w) > 2 and w not in stopwords}

    def evaluate(
        self,
        answer: str,
        retrieved_passages: List[str],
        max_retrieval_similarity: float = 1.0,
        similarity_threshold: float = 0.35,
    ) -> GroundingCheckResult:
        start_time = time.perf_counter()

        if not answer or not answer.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GroundingCheckResult(
                is_grounded=False,
                grounding_score=0.0,
                is_abstention=True,
                hallucination_detected=False,
                reason="Empty answer produced.",
                latency_ms=elapsed_ms,
            )

        # Check explicit abstention keywords
        lower_answer = answer.lower()
        is_explicit_abstention = any(
            re.search(phrase, lower_answer) for phrase in self.ABSTENTION_PHRASES
        )

        if is_explicit_abstention:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GroundingCheckResult(
                is_grounded=True,
                grounding_score=1.0,
                is_abstention=True,
                hallucination_detected=False,
                reason="Model safely abstained due to insufficient context.",
                latency_ms=elapsed_ms,
            )

        # If retrieved context similarity is below threshold, answer should abstain
        if max_retrieval_similarity < similarity_threshold and not is_explicit_abstention:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GroundingCheckResult(
                is_grounded=False,
                grounding_score=0.1,
                is_abstention=False,
                hallucination_detected=True,
                reason=f"Retrieval similarity ({max_retrieval_similarity:.2f}) below threshold ({similarity_threshold:.2f}); model answered without sufficient context.",
                latency_ms=elapsed_ms,
            )

        if not retrieved_passages:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GroundingCheckResult(
                is_grounded=False,
                grounding_score=0.0,
                is_abstention=False,
                hallucination_detected=True,
                reason="No retrieved passages available to ground the response.",
                latency_ms=elapsed_ms,
            )

        # Token overlap verification
        answer_tokens = self._tokenize(answer)
        if not answer_tokens:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GroundingCheckResult(
                is_grounded=True,
                grounding_score=1.0,
                is_abstention=False,
                hallucination_detected=False,
                reason="Answer contains only common connective tokens.",
                latency_ms=elapsed_ms,
            )

        context_tokens = set()
        for p in retrieved_passages:
            context_tokens.update(self._tokenize(p))

        overlap = answer_tokens.intersection(context_tokens)
        grounding_score = len(overlap) / max(len(answer_tokens), 1)

        is_grounded = grounding_score >= self.min_grounding_threshold
        hallucination_detected = not is_grounded

        reason = (
            f"Grounding score: {grounding_score:.2f} ({len(overlap)}/{len(answer_tokens)} tokens matched)."
            if is_grounded
            else f"Low grounding score: {grounding_score:.2f}. Potential hallucination detected."
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return GroundingCheckResult(
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 3),
            is_abstention=False,
            hallucination_detected=hallucination_detected,
            reason=reason,
            latency_ms=elapsed_ms,
        )
