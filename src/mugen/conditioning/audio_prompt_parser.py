"""Rule-based audio/music prompt parsing for showcase generation.

The parser intentionally avoids LLM/API dependencies. It extracts music, sound,
and rhythm descriptions from a natural prompt so audio style can participate in
MUGen's condition fusion even when no audio file is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AudioPromptParseResult:
    """Parsed video-content and audio-style prompt components."""

    original_prompt: str
    content_prompt: str
    audio_prompt: str
    source: str
    descriptors: List[str] = field(default_factory=list)
    motion_phrase: str = "neutral natural motion"
    mood_phrase: str = "neutral ambient sound"

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_prompt": self.original_prompt,
            "content_prompt": self.content_prompt,
            "audio_prompt": self.audio_prompt,
            "source": self.source,
            "descriptors": self.descriptors,
            "motion_phrase": self.motion_phrase,
            "mood_phrase": self.mood_phrase,
        }


class AudioPromptParser:
    """Extract audio/music style from English or Chinese prompts."""

    DEFAULT_AUDIO_PROMPT = "neutral ambient background sound"

    EN_CONNECTORS = (" with ", " featuring ", " background ")
    AUDIO_KEYWORDS = (
        "music", "soundtrack", "audio", "sound", "beat", "beats", "rhythm", "rhythmic",
        "drum", "drums", "ambient", "song", "bgm", "melody", "energetic", "quiet", "soft",
        "upbeat", "fast", "slow",
        "\u97f3\u4e50", "\u80cc\u666f\u97f3\u4e50", "\u914d\u4e50", "\u58f0\u97f3", "\u97f3\u6548", "\u9f13\u70b9", "\u8282\u594f", "\u73af\u5883\u58f0", "\u6c1b\u56f4", "\u8f7b\u5feb", "\u5b89\u9759", "\u5f3a\u70c8",
    )

    def parse(self, prompt: str) -> AudioPromptParseResult:
        original = " ".join((prompt or "").strip().split())
        if not original:
            return self._neutral("")

        audio_prompt = self._extract_audio_prompt(original)
        if not audio_prompt:
            return self._neutral(original)

        content_prompt = self._remove_audio_text(original, audio_prompt)
        if not content_prompt:
            content_prompt = original
        descriptors = self._descriptors(audio_prompt)
        return AudioPromptParseResult(
            original_prompt=original,
            content_prompt=content_prompt,
            audio_prompt=audio_prompt,
            source="prompt",
            descriptors=descriptors,
            motion_phrase=self._motion_phrase(descriptors),
            mood_phrase=self._mood_phrase(descriptors),
        )

    def _neutral(self, content_prompt: str) -> AudioPromptParseResult:
        return AudioPromptParseResult(
            original_prompt=content_prompt,
            content_prompt=content_prompt,
            audio_prompt=self.DEFAULT_AUDIO_PROMPT,
            source="default",
            descriptors=["ambient"],
            motion_phrase="neutral natural motion",
            mood_phrase="neutral ambient sound",
        )

    def _extract_audio_prompt(self, prompt: str) -> str:
        lower = prompt.lower()
        for connector in self.EN_CONNECTORS:
            if connector in lower:
                idx = lower.index(connector)
                tail = prompt[idx + len(connector):].strip(" ,;???")
                if tail and self._contains_audio_keyword(tail):
                    return tail
        for sep in [",", ";", "?", "?", "?"]:
            parts = [part.strip() for part in prompt.split(sep)]
            audio_parts = [part for part in parts if self._contains_audio_keyword(part)]
            if audio_parts:
                return "; ".join(dict.fromkeys(audio_parts))
        if self._contains_audio_keyword(prompt):
            return prompt
        return ""

    def _remove_audio_text(self, prompt: str, audio_prompt: str) -> str:
        if not audio_prompt:
            return prompt
        result = prompt.replace(audio_prompt, "")
        for connector in self.EN_CONNECTORS:
            result = result.replace(connector + audio_prompt, "")
        result = result.replace("with ", "")
        result = result.replace("featuring ", "")
        result = result.replace("background ", "")
        return self._cleanup(result)

    def _contains_audio_keyword(self, text: str) -> bool:
        lower = text.lower()
        return any(keyword.lower() in lower for keyword in self.AUDIO_KEYWORDS)

    def _cleanup(self, text: str) -> str:
        text = " ".join(text.replace("\n", " ").split())
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip(" ,;???")

    def _descriptors(self, audio_prompt: str) -> List[str]:
        lower = audio_prompt.lower()
        descriptors = []
        if any(keyword in lower for keyword in ("rhythm", "rhythmic", "beat", "drum", "\u8282\u594f", "\u9f13\u70b9")):
            descriptors.append("rhythmic")
        if any(keyword in lower for keyword in ("cinematic", "soundtrack", "\u914d\u4e50", "\u7535\u5f71\u611f")):
            descriptors.append("cinematic")
        if any(keyword in lower for keyword in ("ambient", "\u73af\u5883\u58f0", "ambiently")):
            descriptors.append("ambient")
        if any(keyword in lower for keyword in ("fast", "upbeat", "energetic", "\u8f7b\u5feb", "\u5f3a\u70c8")):
            descriptors.append("energetic")
        if any(keyword in lower for keyword in ("soft", "quiet", "slow", "\u5b89\u9759", "\u8212\u7f13")):
            descriptors.append("soft")
        return descriptors or ["ambient"]

    def _motion_phrase(self, descriptors: List[str]) -> str:
        if "energetic" in descriptors and "rhythmic" in descriptors:
            return "fast rhythmic motion"
        if "rhythmic" in descriptors:
            return "steady rhythmic motion"
        if "soft" in descriptors:
            return "soft slow pacing"
        if "ambient" in descriptors:
            return "smooth ambient pacing"
        return "neutral natural motion"

    def _mood_phrase(self, descriptors: List[str]) -> str:
        if "cinematic" in descriptors and "ambient" in descriptors:
            return "cinematic ambient soundtrack mood"
        if "cinematic" in descriptors:
            return "cinematic soundtrack mood"
        if "energetic" in descriptors:
            return "energetic music mood"
        if "soft" in descriptors:
            return "soft quiet sound mood"
        return "neutral ambient sound"
