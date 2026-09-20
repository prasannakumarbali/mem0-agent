"""
REAL Comparison: mem0 approach vs Naive full-history approach
Uses actual Gemini API token counts from usage_metadata — no estimates.
"""

import os
import time
from pathlib import Path
from google import genai
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GEMINI_MODEL = "gemini-3.1-flash-lite"

# same conversation used for both approaches
CONVERSATION = [
    "Hi, my name is Prasanna. I ordered a laptop, order #x22, it hasn't arrived yet.",
    "My email is prasanna@gmail.com and I am a premium member.",
    "The order was placed 10 days ago. Expected delivery was 5 days ago.",
    "I also ordered a mouse with order #x23, that hasn't arrived either.",
    "I prefer to be contacted via email, not phone.",
    "Can you give me a refund if they don't arrive by tomorrow?",
    "What is the status of my orders right now?",
    "Ok thanks. I also want to change my delivery address.",
]


# ─────────────────────────────────────────────
# APPROACH 1: Naive — full history every turn
# ─────────────────────────────────────────────
def run_naive(client):
    print("\n" + "="*60)
    print("  APPROACH 1: Naive (Full Conversation History Every Turn)")
    print("="*60)

    history = []
    system_prompt = "You are a helpful customer support AI agent."
    total_input_tokens = 0
    total_output_tokens = 0
    turn_times = []

    for i, user_msg in enumerate(CONVERSATION):
        history.append({"role": "user", "content": user_msg})

        # build messages with full history every single turn
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]

        full_prompt = system_prompt + "\n\n"
        for m in history:
            role = "Customer" if m["role"] == "user" else "Agent"
            full_prompt += f"{role}: {m['content']}\n"
        full_prompt += "Agent:"

        start = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        elapsed = round(time.time() - start, 2)
        turn_times.append(elapsed)

        input_tokens  = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        total_input_tokens  += input_tokens
        total_output_tokens += output_tokens

        assistant_msg = response.text.strip()
        history.append({"role": "assistant", "content": assistant_msg})

        print(f"  Turn {i+1:2d} | input: {input_tokens:4d} | output: {output_tokens:3d} | time: {elapsed}s")

    avg_time = round(sum(turn_times) / len(turn_times), 2)
    print(f"\n  Total input tokens   : {total_input_tokens:,}")
    print(f"  Total output tokens  : {total_output_tokens:,}")
    print(f"  Avg response time    : {avg_time}s")

    return total_input_tokens, total_output_tokens, avg_time


# ─────────────────────────────────────────────
# APPROACH 2: mem0 — only relevant memories
# ─────────────────────────────────────────────
def run_mem0(client, memory):
    print("\n" + "="*60)
    print("  APPROACH 2: mem0 (Persistent Memory — Top 3 Facts Only)")
    print("="*60)

    user_id = "comparison_test_user"
    system_prompt = "You are a helpful customer support AI agent."
    total_input_tokens = 0
    total_output_tokens = 0
    turn_times = []

    for i, user_msg in enumerate(CONVERSATION):
        # retrieve only top 3 relevant memories
        relevant = memory.search(query=user_msg, filters={"user_id": user_id}, limit=3)
        memories_str = "\n".join(f"- {e['memory']}" for e in relevant["results"])

        # prompt = system + 3 memory bullets + current message only (no history)
        full_prompt = (
            f"{system_prompt}\n"
            f"Customer Memory:\n{memories_str if memories_str else 'No past history.'}\n\n"
            f"Customer: {user_msg}\nAgent:"
        )

        start = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        elapsed = round(time.time() - start, 2)
        turn_times.append(elapsed)

        input_tokens  = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        total_input_tokens  += input_tokens
        total_output_tokens += output_tokens

        assistant_msg = response.text.strip()

        # store exchange in mem0
        memory.add(
            [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            user_id=user_id,
        )

        memories_injected = len(relevant["results"])
        print(f"  Turn {i+1:2d} | input: {input_tokens:4d} | output: {output_tokens:3d} | memories: {memories_injected} | time: {elapsed}s")

    avg_time = round(sum(turn_times) / len(turn_times), 2)
    print(f"\n  Total input tokens   : {total_input_tokens:,}")
    print(f"  Total output tokens  : {total_output_tokens:,}")
    print(f"  Avg response time    : {avg_time}s")

    return total_input_tokens, total_output_tokens, avg_time


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
def print_results(naive_in, naive_out, naive_time, mem0_in, mem0_out, mem0_time):
    naive_total = naive_in + naive_out
    mem0_total  = mem0_in  + mem0_out
    saved       = naive_total - mem0_total
    saving_pct  = round(saved / naive_total * 100, 1)
    time_diff   = round(naive_time - mem0_time, 2)

    # Gemini flash lite pricing: $0.075 input / $0.30 output per 1M tokens
    input_price  = 0.075  / 1_000_000
    output_price = 0.30   / 1_000_000

    naive_cost = (naive_in * input_price) + (naive_out * output_price)
    mem0_cost  = (mem0_in  * input_price) + (mem0_out  * output_price)
    cost_saved = naive_cost - mem0_cost

    print("\n" + "="*60)
    print("  REAL RESULTS (actual Gemini token counts)")
    print("="*60)
    print(f"  {'Metric':<30} {'Naive':>10} {'mem0':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Input tokens':<30} {naive_in:>10,} {mem0_in:>10,}")
    print(f"  {'Output tokens':<30} {naive_out:>10,} {mem0_out:>10,}")
    print(f"  {'Total tokens':<30} {naive_total:>10,} {mem0_total:>10,}")
    print(f"  {'Avg response time':<30} {naive_time:>9}s {mem0_time:>9}s")
    print(f"  {'Cost per session':<30} ${naive_cost:>9.6f} ${mem0_cost:>9.6f}")
    print(f"  {'-'*50}")
    print(f"  {'Tokens saved':<30} {saved:>10,}")
    print(f"  {'Token saving %':<30} {saving_pct:>9}%")
    print(f"  {'Time saved per turn':<30} {time_diff:>9}s")
    print(f"  {'Cost saved per session':<30} ${cost_saved:>9.6f}")

    print(f"\n  At 10,000 sessions/day:")
    print(f"  {'Naive daily cost':<30} ${naive_cost * 10_000:>9.2f}")
    print(f"  {'mem0 daily cost':<30} ${mem0_cost  * 10_000:>9.2f}")
    print(f"  {'Daily savings':<30} ${cost_saved * 10_000:>9.2f}")
    print(f"  {'Monthly savings':<30} ${cost_saved * 10_000 * 30:>9.2f}")
    print("="*60)


if __name__ == "__main__":
    print("\n  Initializing clients...")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-3.1-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY"),
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "embedding_dims": 3072,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": 3072,
            },
        },
    }
    memory = Memory.from_config(config)

    print("  Running 8-turn conversation through both approaches...\n")

    naive_in, naive_out, naive_time = run_naive(client)

    # small pause between approaches to avoid rate limits
    print("\n  Waiting 10s before mem0 run to avoid rate limits...")
    time.sleep(10)

    mem0_in, mem0_out, mem0_time = run_mem0(client, memory)

    print_results(naive_in, naive_out, naive_time, mem0_in, mem0_out, mem0_time)
