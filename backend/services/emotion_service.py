"""
Emotion State Machine
---------------------
Tracks NPC emotional state across a conversation and shifts it based on
detected sentiment of player messages. Uses a keyword heuristic so there
are no extra API calls on the hot path.

Emotion transition rules:
  - Positive sentiment  → nudges toward FRIENDLY / EXCITED
  - Negative sentiment  → nudges toward HOSTILE
  - Threatening words   → immediate spike toward HOSTILE or FEARFUL
  - Excitement keywords → nudge toward EXCITED
Each NPC has an `emotion_sensitivity` scalar (0–1) that scales how quickly
their state changes.
"""

import re
from backend.models.npc import EmotionState, NPCProfile, EMOTION_STYLE


# ── Sentiment keyword lists ────────────────────────────────────────────────

POSITIVE_WORDS = {
    "thank", "thanks", "please", "help", "friend", "kind", "good", "great",
    "wonderful", "love", "appreciate", "grateful", "honor", "respect",
    "agree", "yes", "absolutely", "certainly", "happy", "glad",
}

NEGATIVE_WORDS = {
    "hate", "stupid", "idiot", "useless", "liar", "wrong", "fool",
    "incompetent", "coward", "pathetic", "disgrace", "never", "refuse",
    "betray", "enemy",
}

THREAT_WORDS = {
    "kill", "attack", "destroy", "hurt", "die", "death", "blood",
    "weapon", "fight", "war", "stab", "threaten", "curse",
}

EXCITEMENT_WORDS = {
    "incredible", "amazing", "unbelievable", "discovery", "secret", "found",
    "ancient", "treasure", "legend", "prophecy", "reveal", "power",
}


def _score_message(text: str) -> dict:
    """Return a dict of sentiment scores for a player message."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return {
        "positive":   len(words & POSITIVE_WORDS),
        "negative":   len(words & NEGATIVE_WORDS),
        "threat":     len(words & THREAT_WORDS),
        "excitement": len(words & EXCITEMENT_WORDS),
    }


# ── Transition table ───────────────────────────────────────────────────────
# For each (current_state, dominant_signal) → new_state
_TRANSITIONS: dict[tuple[EmotionState, str], EmotionState] = {
    # From NEUTRAL
    (EmotionState.NEUTRAL,  "positive"):   EmotionState.FRIENDLY,
    (EmotionState.NEUTRAL,  "negative"):   EmotionState.HOSTILE,
    (EmotionState.NEUTRAL,  "threat"):     EmotionState.HOSTILE,
    (EmotionState.NEUTRAL,  "excitement"): EmotionState.EXCITED,
    (EmotionState.NEUTRAL,  "none"):       EmotionState.NEUTRAL,
    # From FRIENDLY
    (EmotionState.FRIENDLY, "positive"):   EmotionState.EXCITED,
    (EmotionState.FRIENDLY, "negative"):   EmotionState.NEUTRAL,
    (EmotionState.FRIENDLY, "threat"):     EmotionState.HOSTILE,
    (EmotionState.FRIENDLY, "excitement"): EmotionState.EXCITED,
    (EmotionState.FRIENDLY, "none"):       EmotionState.FRIENDLY,
    # From HOSTILE
    (EmotionState.HOSTILE,  "positive"):   EmotionState.NEUTRAL,
    (EmotionState.HOSTILE,  "negative"):   EmotionState.HOSTILE,
    (EmotionState.HOSTILE,  "threat"):     EmotionState.HOSTILE,
    (EmotionState.HOSTILE,  "excitement"): EmotionState.HOSTILE,
    (EmotionState.HOSTILE,  "none"):       EmotionState.HOSTILE,
    # From FEARFUL
    (EmotionState.FEARFUL,  "positive"):   EmotionState.NEUTRAL,
    (EmotionState.FEARFUL,  "negative"):   EmotionState.FEARFUL,
    (EmotionState.FEARFUL,  "threat"):     EmotionState.FEARFUL,
    (EmotionState.FEARFUL,  "excitement"): EmotionState.FEARFUL,
    (EmotionState.FEARFUL,  "none"):       EmotionState.FEARFUL,
    # From EXCITED
    (EmotionState.EXCITED,  "positive"):   EmotionState.EXCITED,
    (EmotionState.EXCITED,  "negative"):   EmotionState.NEUTRAL,
    (EmotionState.EXCITED,  "threat"):     EmotionState.FEARFUL,
    (EmotionState.EXCITED,  "excitement"): EmotionState.EXCITED,
    (EmotionState.EXCITED,  "none"):       EmotionState.FRIENDLY,
}


class EmotionService:
    """Stateless service — all state lives in the session object."""

    def update_emotion(
        self,
        current_emotion: EmotionState,
        player_message: str,
        npc: NPCProfile,
    ) -> EmotionState:
        """
        Given the NPC's current emotion and the player's latest message,
        return the updated emotion state.
        """
        scores = _score_message(player_message)

        # Determine dominant signal
        # Threat always wins if present (even a single threat word)
        if scores["threat"] > 0:
            dominant = "threat"
        elif scores["negative"] > scores["positive"] and scores["negative"] > 0:
            dominant = "negative"
        elif scores["excitement"] > scores["positive"] and scores["excitement"] > 0:
            dominant = "excitement"
        elif scores["positive"] > 0:
            dominant = "positive"
        else:
            dominant = "none"

        # Sensitivity gate: low-sensitivity NPCs resist small shifts
        if dominant != "none" and dominant != "threat":
            # Only shift if sensitivity roll passes (simplistic threshold)
            if npc.emotion_sensitivity < 0.4 and dominant in ("positive", "excitement"):
                dominant = "none"  # stoic NPC ignores mild positivity

        next_state = _TRANSITIONS.get((current_emotion, dominant), current_emotion)
        return next_state

    def get_emotion_instruction(self, emotion: EmotionState) -> str:
        """Return the system-prompt instruction for the current emotion."""
        return EMOTION_STYLE[emotion]


emotion_service = EmotionService()
