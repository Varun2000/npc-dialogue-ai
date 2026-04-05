# 🎮 NPC Dialogue AI — Aethermoor

> **LLM-powered NPC dialogue system** with RAG grounding, persona consistency, dynamic emotion state, and real-time streaming. Built with Mistral AI, ChromaDB, and FastAPI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-mistral--small-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Demo

[Demo](https://github.com/Varun2000/npc-dialogue-ai/blob/main/assets/demo.mov)

---

## ✨ Features

| Feature | What it does |
|---|---|
| **Persona Consistency** | Each NPC has a detailed identity — backstory, speech style, personality traits. The LLM never breaks character. |
| **RAG (Retrieval-Augmented Generation)** | Game lore is stored in ChromaDB. Relevant world knowledge is retrieved per query and injected into the system prompt, preventing hallucination. |
| **Emotion State Machine** | Each NPC tracks an emotional state (Neutral / Friendly / Hostile / Fearful / Excited) that shifts based on conversation sentiment. Emotion colors the NPC's tone and willingness to share information. |
| **Real-time Streaming** | Responses stream token-by-token via Server-Sent Events (SSE) for instant, low-latency feel. |
| **Session Memory** | Full conversation history per session with configurable memory window. |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (SSE Client)                 │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /chat/{npc_id}
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                       │
│                                                         │
│  1. EmotionService  →  update NPC emotional state       │
│  2. RAGService      →  retrieve relevant lore chunks    │
│  3. LLMService      →  build system prompt + stream     │
│                         ┌─────────────────────┐         │
│                         │   Mistral AI API    │         │
│                         │  (mistral-small)    │         │
│                         └─────────────────────┘         │
│                                                         │
│  ChromaDB (local) ←── sentence-transformers embeddings  │
└─────────────────────────────────────────────────────────┘
```

### System Prompt Layers

```
[1] NPC Identity       →  name, backstory, role, speech style, traits
[2] Emotion Modifier   →  current emotional state colors tone & behavior
[3] RAG Context        →  top-K lore chunks most relevant to the query
[4] Hard Rules         →  stay in character, no meta-game references
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Varun2000/npc-dialogue-ai.git
cd npc-dialogue-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize the lore database

```bash
python -m backend.init_db
# Output: Ingested 18 lore chunks. ChromaDB is ready.
```

### 3. Run the server

```bash
uvicorn backend.main:app --reload
```

### 4. Open the UI

Visit **http://localhost:8000** in your browser.

Or use the API directly:

```bash
curl -X POST http://localhost:8000/chat/eldrin \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the Sundering"}' \
  --no-buffer
```

---

## 🧙 NPCs

| Character | Role | Default Emotion | Sensitivity |
|---|---|---|---|
| **Eldrin Ashveil** 🧙 | Ancient Court Wizard | Neutral | Low (stoic) |
| **Kira Stonefang** ⚔️ | Mercenary Guard Captain | Neutral | High (reactive) |
| **Mira Coldwell** 🍺 | Tavern Keeper & Informant | Friendly | Medium |

### Emotion Transitions

```
Player message sentiment →  Dominant signal  →  New NPC emotion
─────────────────────────────────────────────────────────────────
Positive words           →  positive         →  Friendly / Excited
Rude / negative words    →  negative         →  Neutral / Hostile
Threat keywords          →  threat           →  Hostile / Fearful
Excitement keywords      →  excitement       →  Excited
No clear signal          →  none             →  State persists
```

Each NPC's `emotion_sensitivity` (0–1) controls how easily their state shifts.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/npcs` | List all NPCs |
| `GET` | `/npcs/{npc_id}` | Get NPC profile |
| `POST` | `/chat/{npc_id}` | Chat (SSE stream) |
| `GET` | `/sessions/{session_id}` | Get session history |
| `DELETE` | `/sessions/{session_id}` | Reset session |
| `GET` | `/health` | Health + RAG status |
| `GET` | `/app` | Frontend UI |

### Chat request body

```json
{
  "message": "What do you know about the Veilborn?",
  "session_id": "optional-uuid-to-continue-session"
}
```

### SSE stream format

```
data: Once, long ago,          ← token chunks
data:  the Veil was whole…
data: [METADATA] {"session_id": "...", "emotion": "fearful", ...}
data: [DONE]
```

---

## 📁 Project Structure

```
npc-dialogue-ai/
├── backend/
│   ├── main.py                 # FastAPI app + routes
│   ├── config.py               # Settings (pydantic-settings)
│   ├── init_db.py              # Lore ingestion script
│   ├── models/
│   │   ├── npc.py              # NPCProfile, EmotionState
│   │   └── conversation.py     # Session, Message, ChatRequest
│   ├── services/
│   │   ├── llm_service.py      # Mistral streaming + prompt builder
│   │   ├── rag_service.py      # ChromaDB ingestion + query
│   │   └── emotion_service.py  # Emotion state machine
│   └── data/
│       ├── npcs/               # NPC JSON profiles
│       └── lore/               # World lore .txt files
├── frontend/
│   └── index.html              # Single-file React-free UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🛠 Adding Your Own NPCs

1. Create `backend/data/npcs/your_npc.json`:

```json
{
  "id": "your_npc",
  "name": "Character Name",
  "role": "Their Role",
  "backstory": "Their history...",
  "personality_traits": ["trait1", "trait2"],
  "speech_style": "How they speak...",
  "knowledge_domains": ["topic1", "topic2"],
  "default_emotion": "neutral",
  "emotion_sensitivity": 0.5,
  "avatar_emoji": "🎭",
  "lore_collection": "world_lore"
}
```

2. Restart the server — the NPC auto-loads from the JSON directory.

---

## 🔮 Extending the Project

- **Fine-tuning**: Fine-tune Mistral on in-character NPC dialogue samples for even stronger persona lock-in
- **Persistent sessions**: Swap the in-memory dict for Redis to survive server restarts
- **Multi-NPC conversations**: Route messages through multiple NPC agents with shared memory
- **Unity/Unreal plugin**: Expose the SSE endpoint as a game engine plugin

---

## 🧰 Tech Stack

- **LLM**: Mistral AI (`mistral-small-latest`) — free tier available
- **Vector DB**: ChromaDB (local persistent) + sentence-transformers (`all-MiniLM-L6-v2`)
- **Backend**: FastAPI + uvicorn + SSE streaming
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Config**: pydantic-settings

---

## 📄 License

MIT — free to use, modify, and extend.

---

*Built by [Sai Varun D](https://linkedin.com/in/sai-varun-d-371581160) as an AI/ML portfolio project demonstrating LLM integration, RAG pipelines, and real-time streaming in a gaming context.*
