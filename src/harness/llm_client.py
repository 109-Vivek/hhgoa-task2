import os
import time
import json
from typing import Dict, Any, Optional, List
import requests
from src.config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    SARVAM_API_KEY,
    PRIMARY_LLM_PROVIDER,
    PRIMARY_LLM_MODEL,
    FALLBACK_LLM_MODEL,
    FORCE_LLM_MOCK,
    ALLOW_LLM_FALLBACK,
)


class ResilientLLMClient:
    """
    High-performance, multi-provider LLM client with configurable fallback,
    retry mechanism, and deterministic offline synthesis.
    Supports Groq, xAI (Grok), OpenAI, Gemini, Sarvam, and Mock mode.
    """

    def __init__(self):
        self.groq_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.sarvam_key = SARVAM_API_KEY or os.getenv("SARVAM_API_KEY", "")
        self.disabled_providers = set()

    def generate_answer(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        lang: str = "en",
        is_abstention: bool = False,
    ) -> Dict[str, Any]:
        """
        Synthesizes a grounded answer from retrieved context.
        Returns: { 'answer': str, 'provider': str, 'latency_ms': float }
        """
        start_time = time.perf_counter()

        if is_abstention or not retrieved_contexts:
            refusal_msg = self._get_abstention_text(lang)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "answer": refusal_msg,
                "provider": "guardrail_abstention",
                "latency_ms": elapsed_ms,
            }

        # If FORCE_LLM_MOCK is set, directly use local grounded synthesizer
        if FORCE_LLM_MOCK:
            fallback_answer = self._generate_extractive_grounded_answer(query, retrieved_contexts, lang)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "answer": fallback_answer,
                "provider": "local_mock_harness",
                "latency_ms": elapsed_ms,
            }

        # Build context prompt
        context_str = "\n\n".join(
            [
                f"[{i+1}] (Doc ID: {doc.get('passage_id', doc.get('chunk_id', 'unknown'))}): {doc.get('raw_text', doc.get('text', ''))}"
                for i, doc in enumerate(retrieved_contexts)
            ]
        )

        lang_names = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
        target_lang = lang_names.get(lang, "the same language as the query")

        system_prompt = (
            f"You are an ultra-fast, accurate Indic Voice-RAG assistant for HH Goa 2026. "
            f"Answer the user's question concisely in {target_lang} based strictly on the retrieved context below. "
            f"Cite the source passages using [1], [2] when appropriate. "
            f"If the context does not contain sufficient information, state clearly that you do not have enough context."
        )

        user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

        # Try Providers in order, respecting PRIMARY_LLM_PROVIDER first
        provider_attempts = []
        if PRIMARY_LLM_PROVIDER == "gemini":
            provider_attempts = ["gemini", "groq", "openai"]
        elif PRIMARY_LLM_PROVIDER == "openai":
            provider_attempts = ["openai", "gemini", "groq"]
        else:
            provider_attempts = ["groq", "gemini", "openai"]

        # If xAI key is provided, prioritize it
        if "xai" not in self.disabled_providers and (self.groq_key.startswith("xai-") or self.gemini_key.startswith("xai-")):
            xai_key = self.groq_key if self.groq_key.startswith("xai-") else self.gemini_key
            res = self._call_xai(xai_key, system_prompt, user_prompt)
            if res:
                res["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
                return res

        for provider in provider_attempts:
            if provider == "gemini" and "gemini" not in self.disabled_providers:
                if self.gemini_key and not self.gemini_key.startswith("xai-") and not self.gemini_key.startswith("your_"):
                    res = self._call_gemini(system_prompt, user_prompt)
                    if res:
                        res["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
                        return res

            elif provider == "groq" and "groq" not in self.disabled_providers:
                if self.groq_key and self.groq_key.startswith("gsk_"):
                    res = self._call_groq(system_prompt, user_prompt)
                    if res:
                        res["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
                        return res

            elif provider == "openai" and "openai" not in self.disabled_providers:
                if self.openai_key and not self.openai_key.startswith("your_"):
                    res = self._call_openai(system_prompt, user_prompt)
                    if res:
                        res["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
                        return res

        # If live LLM calls failed and fallback is disabled, raise error
        if not ALLOW_LLM_FALLBACK:
            raise RuntimeError("[LLM Client] Configured live LLM providers failed and ALLOW_LLM_FALLBACK=False.")

        # Fast local grounded fallback (<2ms latency)
        fallback_answer = self._generate_extractive_grounded_answer(query, retrieved_contexts, lang)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "answer": fallback_answer,
            "provider": "local_indic_harness",
            "latency_ms": elapsed_ms,
        }

    def _call_groq(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            completion = client.chat.completions.create(
                model=PRIMARY_LLM_MODEL or "llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return {
                "answer": completion.choices[0].message.content.strip(),
                "provider": "groq",
            }
        except Exception as e:
            print(f"[LLM Client] Groq attempt failed: {e}")
            return None

    def _call_xai(self, api_key: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            for model_name in ["grok-2", "grok-2-1212", "grok-beta"]:
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "model": model_name,
                    "temperature": 0.2,
                    "max_tokens": 300,
                }
                resp = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "answer": data["choices"][0]["message"]["content"].strip(),
                        "provider": "xai-grok",
                    }
            print(f"[LLM Client] xAI HTTP {resp.status_code}: {resp.text}")
            if resp.status_code in [400, 401, 403]:
                self.disabled_providers.add("xai")
            return None
        except Exception as e:
            print(f"[LLM Client] xAI attempt failed: {e}")
            self.disabled_providers.add("xai")
            return None

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            models_to_try = []
            if PRIMARY_LLM_PROVIDER == "gemini" and PRIMARY_LLM_MODEL.startswith("gemini"):
                models_to_try.append(PRIMARY_LLM_MODEL)
            models_to_try.extend(["gemini-3.6-flash", "gemini-3-flash"])
            # Remove duplicates while preserving order
            models_to_try = list(dict.fromkeys(models_to_try))

            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                    )
                    if response and response.text:
                        return {
                            "answer": response.text.strip(),
                            "provider": f"gemini ({model_name})",
                        }
                except Exception as model_err:
                    print(f"[LLM Client] Gemini model {model_name} error: {model_err}")
                    continue

            return None
        except Exception as e:
            print(f"[LLM Client] Gemini attempt failed: {e}")
            return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return {
                "answer": completion.choices[0].message.content.strip(),
                "provider": "openai",
            }
        except Exception as e:
            print(f"[LLM Client] OpenAI attempt failed: {e}")
            return None

    @staticmethod
    def _get_abstention_text(lang: str) -> str:
        if "hi" in lang:
            return "मुझे इस प्रश्न का उत्तर देने के लिए डेटाबेस में पर्याप्त प्रासंगिक संदर्भ नहीं मिला।"
        elif "ta" in lang:
            return "இந்தக் கேள்விக்கு துல்லியமாக பதிலளிக்க தரவுத்தளத்தில் போதிய சூழல் கிடைக்கவில்லை."
        return "I do not have enough relevant context in the database to answer this question accurately."

    @staticmethod
    def _generate_extractive_grounded_answer(
        query: str, contexts: List[Dict[str, Any]], lang: str
    ) -> str:
        """
        Deterministic, grounded response synthesizer from the most relevant retrieved passages.
        Ensures 100% grounding without unverified hallucinations even when offline.
        """
        if not contexts:
            return ResilientLLMClient._get_abstention_text(lang)

        top_doc = contexts[0]
        text = top_doc.get("raw_text") or top_doc.get("text") or ""
        
        # Split into key sentences
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if sentences:
            selected = ". ".join(sentences[:2])
            return f"{selected}. [1]"
        return f"{text[:200]}... [1]"
