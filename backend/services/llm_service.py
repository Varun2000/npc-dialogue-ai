"""
LLM Service — Mistral AI (streaming)
--------------------------------------
Builds the full system prompt for an NPC (persona + emotion + RAG context)
and streams token-by-token responses back to the client via SSE.

System prompt layers:
  1. Core identity     — who the NPC is, their backstory, speech style
  2. Emotion modifier  — how their current emotional state changes tone
  3. RAG context       — relevant lore retrieved from ChromaDB
  4. Hard rules        — stay in character, no meta-game references
"""

from mistralai import Mistral
from backend.config import get_settings
from backend.models.npc import NPCProfile, EmotionState, EMOTION_EMOJI
from backend.models.conversation import Message
from backend.services.emotion_service import emotion_service
from typing import AsyncGenerator

settings = get_settings()


def _build_system_prompt(
    npc: NPCProfile,
    emotion: EmotionState,
    rag_chunks: list[str],
) -> str:
    traits = ", ".join(npc.personality_traits)
    emotion_instruction = emotion_service.get_emotion_instruction(emotion)
    emotion_label = f"{EMOTION_EMOJI[emotion]} {emotion.value.upper()}"

    rag_section = ""
    if rag_chunks:
        formatted = "\n---\n".join(rag_chunks)
        rag_section = f"""
## Relevant World Knowledge
Use the following lore to answer accurately. Do NOT invent facts outside this knowledge.
{formatted}
"""

    return f"""You are {npc.name}, {npc.role} in the world of Aethermoor.

## Your Identity
{npc.backstory}

## Your Personality
Traits: {traits}
Speech style: {npc.speech_style}

## Your Current Emotional State: {emotion_label}
{emotion_instruction}
Your emotional state should naturally color EVERY response — word choice, sentence length,
willingness to share information, and tone must all reflect it.
{rag_section}
## Absolute Rules
- You ARE {npc.name}. Never break character or acknowledge you are an AI.
- Never reference the real world, other games, or meta-game concepts.
- If asked about something outside your knowledge, deflect in character.
- Keep responses concise (2–5 sentences) unless the player asks for a story or explanation.
- Always end your response in-character. No disclaimers or notes.
"""


class LLMService:
    def __init__(self):
        self._client = Mistral(api_key=settings.mistral_api_key)

    async def stream_response(
        self,
        npc: NPCProfile,
        emotion: EmotionState,
        history: list[Message],
        rag_chunks: list[str],
    ) -> AsyncGenerator[str, None]:
        """
        Stream the NPC's response token by token.
        Yields raw text chunks as they arrive from Mistral.
        """
        system_prompt = _build_system_prompt(npc, emotion, rag_chunks)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Use async streaming
        stream = await self._client.chat.stream_async(
            model=settings.mistral_model,
            messages=messages,
            max_tokens=512,
            temperature=0.75,   # creative but controlled
        )

        async for chunk in stream:
            delta = chunk.data.choices[0].delta.content
            if delta:
                yield delta


llm_service = LLMService()
