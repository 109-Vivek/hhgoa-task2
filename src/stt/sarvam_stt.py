import os
import time
import requests
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from src.config import SARVAM_API_KEY, FORCE_STT_MOCK, ALLOW_STT_FALLBACK


@dataclass
class TranscriptionResult:
    text: str
    language_code: str
    confidence: float
    latency_ms: float
    is_mock: bool = False
    raw_response: Optional[Dict[str, Any]] = None


class SarvamSTT:
    """
    Speech-to-Text transcriber using Sarvam AI API (saaras:v4 model)
    with support for Hindi (hi-IN), Tamil (ta-IN), Indian English (en-IN), and configurable mock fallback.
    """

    SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str = SARVAM_API_KEY):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "hi-IN",
        model: str = "saaras:v4",
    ) -> TranscriptionResult:
        """
        Transcribes raw audio bytes using Sarvam STT API.
        Respects FORCE_STT_MOCK and ALLOW_STT_FALLBACK environment toggles.
        """
        start_time = time.perf_counter()

        if FORCE_STT_MOCK:
            print("\n" + "=" * 65)
            print("⚠️  [Sarvam STT] MOCK MODE ACTIVE (FORCE_STT_MOCK=True)")
            print(f"• Language: {language_code}")
            res = self._mock_transcription(start_time, language_code, reason="FORCE_STT_MOCK=True")
            print(f"• Mock STT Text: \"{res.text}\"")
            print("• Status: 🔶 SIMULATED MOCK (Not a live API call)")
            print("=" * 65 + "\n")
            return res

        if not self.api_key or self.api_key.startswith("your_"):
            if not ALLOW_STT_FALLBACK:
                raise RuntimeError("[Sarvam STT] SARVAM_API_KEY is missing and ALLOW_STT_FALLBACK=False.")
            print("\n" + "=" * 65)
            print("⚠️  [Sarvam STT] API KEY MISSING -> FALLING BACK TO MOCK")
            print(f"• Language: {language_code}")
            res = self._mock_transcription(start_time, language_code, reason="SARVAM_API_KEY missing/placeholder")
            print(f"• Mock STT Text: \"{res.text}\"")
            print("• Status: 🔶 SIMULATED MOCK (Provide SARVAM_API_KEY in .env for live STT)")
            print("=" * 65 + "\n")
            return res

        headers = {
            "api-subscription-key": self.api_key
        }
        
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        
        data = {
            "model": model,
            "language_code": language_code,
            "with_timestamps": "false",
        }

        try:
            print(f"\n[Sarvam STT] 🚀 Sending live audio request to Sarvam API ({model}, {language_code})...")
            response = requests.post(
                self.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=10,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 200:
                res_data = response.json()
                transcript = res_data.get("transcript", "").strip()
                detected_lang = res_data.get("language_code", language_code)

                print("=" * 65)
                print("🎙️  [Sarvam STT] ✅ LIVE TRANSCRIPTION SUCCESS")
                print(f"• Model          : {model}")
                print(f"• Target Lang    : {language_code} -> Detected: {detected_lang}")
                print(f"• Latency        : {elapsed_ms:.2f} ms")
                print(f"• REAL STT TEXT  : \"{transcript}\"")
                print("• Mode           : ✅ REAL LIVE SARVAM API CALL")
                print("=" * 65 + "\n")

                return TranscriptionResult(
                    text=transcript,
                    language_code=detected_lang,
                    confidence=0.95,
                    latency_ms=elapsed_ms,
                    is_mock=False,
                    raw_response=res_data,
                )
            else:
                err_msg = f"Sarvam STT HTTP {response.status_code}: {response.text}"
                print(f"[Sarvam STT] ❌ Live API call failed: {err_msg}")
                if not ALLOW_STT_FALLBACK:
                    raise RuntimeError(f"[Sarvam STT] {err_msg}")

                print("\n" + "=" * 65)
                print(f"⚠️  [Sarvam STT] FALLING BACK TO MOCK (API Error {response.status_code})")
                res = self._mock_transcription(start_time, language_code, reason=f"HTTP {response.status_code}")
                print(f"• Mock STT Text: \"{res.text}\"")
                print("• Mode: 🔶 SIMULATED MOCK")
                print("=" * 65 + "\n")
                return res

        except Exception as e:
            if not ALLOW_STT_FALLBACK and not isinstance(e, RuntimeError):
                raise RuntimeError(f"[Sarvam STT] Live transcription failed: {e}")
            elif not ALLOW_STT_FALLBACK and isinstance(e, RuntimeError):
                raise e

            print(f"[Sarvam STT] ❌ Exception during live transcription: {e}")
            print("\n" + "=" * 65)
            print("⚠️  [Sarvam STT] FALLING BACK TO MOCK (Exception caught)")
            res = self._mock_transcription(start_time, language_code, reason=str(e))
            print(f"• Mock STT Text: \"{res.text}\"")
            print("• Mode: 🔶 SIMULATED MOCK")
            print("=" * 65 + "\n")
            return res

    def transcribe_file(self, file_path: str, language_code: str = "hi-IN") -> TranscriptionResult:
        """Transcribes audio from a local file path."""
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe_audio_bytes(
            audio_bytes,
            filename=os.path.basename(file_path),
            language_code=language_code,
        )

    @staticmethod
    def _mock_transcription(start_time: float, language_code: str, reason: str = "") -> TranscriptionResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # Return realistic sample transcript for local testing
        if "hi" in language_code:
            text = "भारत की राजधानी क्या है?"
        elif "ta" in language_code:
            text = "இந்தியாவின் தலைநகரம் எது?"
        else:
            text = "What is the capital of India?"

        return TranscriptionResult(
            text=text,
            language_code=language_code,
            confidence=0.92,
            latency_ms=elapsed_ms or 45.0,
            is_mock=True,
            raw_response={"mock": True, "reason": reason},
        )
