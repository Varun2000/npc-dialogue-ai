from pydantic import BaseModel
from typing import Optional
from backend.models.npc import EmotionState


class Message(BaseModel):
    role: str           # "user" or "assistant"
    content: str


class ConversationSession(BaseModel):
    session_id: str
    npc_id: str
    messages: list[Message] = []
    current_emotion: EmotionState = EmotionState.NEUTRAL


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # creates new session if omitted


class ChatMetadata(BaseModel):
    session_id: str
    npc_id: str
    npc_name: str
    emotion: EmotionState
    emotion_emoji: str
    rag_context_used: bool
