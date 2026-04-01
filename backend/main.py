"""
NPC Dialogue AI — FastAPI Backend
===================================
Endpoints:
  GET  /npcs                         → list all available NPCs
  GET  /npcs/{npc_id}                → get a single NPC profile
  POST /chat/{npc_id}                → start/continue chat (SSE stream)
  GET  /sessions/{session_id}        → get conversation history + emotion
  DELETE /sessions/{session_id}      → reset a session
  GET  /health                       → health check + RAG status
"""

import json
import uuid
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.models.npc import NPCProfile, EmotionState, EMOTION_EMOJI
from backend.models.conversation import (
    ConversationSession, ChatRequest, Message, ChatMetadata
)
from backend.services.llm_service import llm_service
from backend.services.rag_service import rag_service
from backend.services.emotion_service import emotion_service

settings = get_settings()

# ── In-memory session store (swap for Redis in production) ─────────────────
_sessions: dict[str, ConversationSession] = {}

# ── NPC registry ───────────────────────────────────────────────────────────
_npcs: dict[str, NPCProfile] = {}

NPC_DIR = Path(__file__).parent / "data" / "npcs"
LORE_DIR = Path(__file__).parent / "data" / "lore"


def _load_npcs():
    for path in NPC_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        npc = NPCProfile(**data)
        _npcs[npc.id] = npc
    print(f"  Loaded {len(_npcs)} NPC profiles: {list(_npcs.keys())}")


def _ensure_rag():
    if rag_service.collection_size("world_lore") == 0:
        print("  ChromaDB empty — ingesting lore now...")
        count = rag_service.ingest_lore_directory(str(LORE_DIR), "world_lore")
        print(f"  Ingested {count} lore chunks.")
    else:
        print(f"  ChromaDB ready ({rag_service.collection_size('world_lore')} chunks).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting NPC Dialogue AI...")
    _load_npcs()
    _ensure_rag()
    print("Ready.")
    yield


app = FastAPI(
    title="NPC Dialogue AI",
    description="LLM-powered NPC dialogue with RAG, persona consistency, emotion state, and streaming.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "npcs_loaded": len(_npcs),
        "rag_chunks": rag_service.collection_size("world_lore"),
        "model": settings.mistral_model,
    }


@app.get("/npcs")
def list_npcs():
    return [
        {
            "id": npc.id,
            "name": npc.name,
            "role": npc.role,
            "avatar_emoji": npc.avatar_emoji,
            "default_emotion": npc.default_emotion,
        }
        for npc in _npcs.values()
    ]


@app.get("/npcs/{npc_id}")
def get_npc(npc_id: str):
    npc = _npcs.get(npc_id)
    if not npc:
        raise HTTPException(404, f"NPC '{npc_id}' not found")
    return npc


@app.post("/chat/{npc_id}")
async def chat(npc_id: str, request: ChatRequest):
    """
    Stream the NPC's response using Server-Sent Events (SSE).

    The stream sends two types of events:
      - data: <token>            plain text token chunks
      - data: [METADATA] {...}   final JSON metadata (emotion, session_id, etc.)
      - data: [DONE]             signals end of stream
    """
    npc = _npcs.get(npc_id)
    if not npc:
        raise HTTPException(404, f"NPC '{npc_id}' not found")

    # Resolve or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSession(
            session_id=session_id,
            npc_id=npc_id,
            current_emotion=npc.default_emotion,
        )
    session = _sessions[session_id]

    # Guard: don't let sessions drift to a different NPC
    if session.npc_id != npc_id:
        raise HTTPException(400, "Session belongs to a different NPC")

    player_message = request.message.strip()
    if not player_message:
        raise HTTPException(400, "Message cannot be empty")

    # 1. Update emotion based on player message
    new_emotion = emotion_service.update_emotion(
        session.current_emotion, player_message, npc
    )
    session.current_emotion = new_emotion

    # 2. Retrieve relevant lore via RAG
    rag_chunks = rag_service.query(
        query_text=player_message,
        collection_name=npc.lore_collection,
    )

    # 3. Add player message to history
    session.messages.append(Message(role="user", content=player_message))

    # Trim history to avoid runaway token counts
    max_msgs = settings.max_history_messages
    if len(session.messages) > max_msgs:
        session.messages = session.messages[-max_msgs:]

    async def event_stream():
        full_response = []

        async for token in llm_service.stream_response(
            npc=npc,
            emotion=new_emotion,
            history=session.messages,
            rag_chunks=rag_chunks,
        ):
            full_response.append(token)
            yield f"data: {token}\n\n"

        # Store assistant response in history
        assistant_text = "".join(full_response)
        session.messages.append(Message(role="assistant", content=assistant_text))

        # Send final metadata event
        metadata = ChatMetadata(
            session_id=session_id,
            npc_id=npc_id,
            npc_name=npc.name,
            emotion=new_emotion,
            emotion_emoji=EMOTION_EMOJI[new_emotion],
            rag_context_used=len(rag_chunks) > 0,
        )
        yield f"data: [METADATA] {metadata.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {
        "session_id": session.session_id,
        "npc_id": session.npc_id,
        "current_emotion": session.current_emotion,
        "emotion_emoji": EMOTION_EMOJI[session.current_emotion],
        "message_count": len(session.messages),
        "messages": session.messages,
    }


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found")
    npc_id = _sessions[session_id].npc_id
    npc = _npcs[npc_id]
    _sessions[session_id] = ConversationSession(
        session_id=session_id,
        npc_id=npc_id,
        current_emotion=npc.default_emotion,
    )
    return {"status": "reset", "session_id": session_id}


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
