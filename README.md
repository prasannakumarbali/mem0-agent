# Customer Support AI Agent with Persistent Memory

An AI-powered customer support agent that remembers customers across sessions using [mem0](https://mem0.ai) for long-term memory, Google Gemini as the LLM, and Qdrant as the vector database.

## Demo

```
Session 1:
  You: I ordered a laptop, order #x22, it hasn't arrived
  Agent: I'm sorry to hear that! Let me look into order #x22...

  [exit]

Session 2 (new session, same customer):
  You: hey do you remember me?
  Agent: Of course! Welcome back Raam. Any update on your laptop order #x22?
```

The agent remembers your name, order details, and past issues — without you repeating yourself.

## How It Works

```
User message
     ↓
Gemini Embeddings  →  converts text to vector
     ↓
Qdrant (Docker)    →  searches similar past memories
     ↓
Gemini LLM         →  generates response with memories injected into prompt
     ↓
Gemini LLM         →  extracts key facts from conversation
     ↓
Qdrant             →  stores new facts persistently
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 3.1 Flash Lite |
| Embeddings | Google Gemini Embedding 001 |
| Memory Layer | mem0 (OSS) |
| Vector Database | Qdrant (Docker) |
| Language | Python 3.11+ |

## Project Structure

```
mem0/
├── oss/
│   └── support_agent.py   # Main agent (extended & production-ready)
├── docker/
│   └── docker-compose.yml # Qdrant vector DB
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and navigate
```bash
git clone <your-repo-url>
cd mem0
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your Gemini API key (get it free at [aistudio.google.com](https://aistudio.google.com)):
```
GEMINI_API_KEY=your_key_here
```

### 5. Start Qdrant (vector database)
```bash
cd docker && docker compose up -d
```

### 6. Run the agent
```bash
python oss/support_agent.py
```

## Usage

```
=== Customer Support AI Agent ===
Enter your customer ID: 123

--- Support Session Started for customer: 123 ---
Type 'memories' to see what I remember about you.
Type 'exit' to end the session.

You: I ordered a laptop and it hasn't arrived
Agent: I'm sorry to hear that! Could you share your order number?

You: memories        ← see all stored facts
You: exit            ← end session
```

Run again with the same customer ID — the agent will remember everything from previous sessions.

## Key Features

- **Persistent memory across sessions** — Qdrant stores facts permanently
- **Intelligent fact extraction** — mem0 extracts only key facts, not raw conversation
- **Multi-user support** — each customer ID has isolated memory
- **In-session context** — remembers conversation history within a session
- **Memory inspection** — type `memories` to see stored facts anytime

## What mem0 Does

Instead of storing entire conversations (expensive, slow), mem0 extracts only the key facts:

```
Full conversation  →  mem0  →  "User ordered laptop #x22, undelivered"
                               "User's name is Raam"
                               "User requested escalation"
```

This gives **90% token savings** and **91% lower latency** vs full-context approaches.
