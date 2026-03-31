from pydantic import BaseModel
from enum import Enum
from typing import Optional


class EmotionState(str, Enum):
    NEUTRAL  = "neutral"
    FRIENDLY = "friendly"
    HOSTILE  = "hostile"
    FEARFUL  = "fearful"
    EXCITED  = "excited"


EMOTION_EMOJI = {
    EmotionState.NEUTRAL:  "😐",
    EmotionState.FRIENDLY: "😊",
    EmotionState.HOSTILE:  "😠",
    EmotionState.FEARFUL:  "😰",
    EmotionState.EXCITED:  "🤩",
}

# Maps each emotion to how the NPC speech style should shift
EMOTION_STYLE = {
    EmotionState.NEUTRAL:  "Speak in a measured, matter-of-fact tone. Neither warm nor cold.",
    EmotionState.FRIENDLY: "Speak warmly and openly. Use lighter language and be more forthcoming.",
    EmotionState.HOSTILE:  "Speak with suspicion and edge. Keep responses short and guarded. Show distrust.",
    EmotionState.FEARFUL:  "Speak nervously. Use shorter sentences. React anxiously to the topic.",
    EmotionState.EXCITED:  "Speak with enthusiasm and energy. Be expressive and eager to share.",
}


class NPCProfile(BaseModel):
    id: str
    name: str
    role: str
    backstory: str
    personality_traits: list[str]
    speech_style: str
    knowledge_domains: list[str]
    default_emotion: EmotionState
    emotion_sensitivity: float  # 0.0 (hard to shift) – 1.0 (very reactive)
    avatar_emoji: str
    lore_collection: str        # ChromaDB collection name to query
