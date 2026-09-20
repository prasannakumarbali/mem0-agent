import os
import time
from pathlib import Path
from google import genai
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_MODEL = "gemini-3.1-flash-lite"


def count_tokens(text):
    # rough estimate: 1 token ≈ 4 characters (standard approximation)
    return len(text) // 4


class CustomerSupportAIAgent:
    def __init__(self):
        # ! Make sure qdrant is running (see docker-compose.yml)
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
        self.memory = Memory.from_config(config)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.app_id = "customer-support"

        # session metrics
        self.session_tokens_used = 0
        self.session_turns = 0
        self.session_start = time.time()

    def handle_query(self, query, user_id):
        turn_start = time.time()

        # Step 1: search relevant memories
        relevant_memories = self.memory.search(query=query, filters={"user_id": user_id}, limit=3)
        memories_str = "\n".join(
            f"- {entry['memory']}" for entry in relevant_memories["results"]
        )

        # Step 2: build system prompt with injected memories
        system_prompt = (
            "You are a helpful customer support AI agent.\n"
            "Use the customer's past interaction history below to give personalized responses.\n"
            f"Customer Memory:\n{memories_str if memories_str else 'No past history found.'}"
        )

        # Step 3: build prompt — memories replace history (no raw history sent)
        full_prompt = f"{system_prompt}\n\nCustomer: {query}\nAgent:"

        # --- token tracking ---
        tokens_used = count_tokens(full_prompt)
        self.session_tokens_used += tokens_used
        self.session_turns += 1

        # Step 4: get response from Gemini
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        assistant_response = response.text.strip()

        turn_time = round(time.time() - turn_start, 2)

        print(f"  [tokens used: {tokens_used} | memories: {len(relevant_memories['results'])} | time: {turn_time}s]")

        # Step 5: store full exchange in memory
        self.memory.add(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": assistant_response},
            ],
            user_id=user_id,
            metadata={"app_id": self.app_id},
        )

        return assistant_response

    def get_memories(self, user_id):
        memories = self.memory.get_all(filters={"user_id": user_id})
        return memories.get("results", [])

    def print_session_stats(self):
        session_time = round(time.time() - self.session_start, 2)
        print("\n========== Session Stats ==========")
        print(f"  Turns                : {self.session_turns}")
        print(f"  Session duration     : {session_time}s")
        print(f"  Tokens used (mem0)   : {self.session_tokens_used}")
        print(f"  See compare_approaches.py for token savings benchmark")
        print("===================================\n")

    def chat(self, user_id):
        print(f"\n--- Support Session Started for customer: {user_id} ---")
        print("Type 'memories' to see what I remember about you.")
        print("Type 'stats'    to see token savings so far.")
        print("Type 'exit'     to end the session.\n")

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Agent: Thank you for contacting support. Goodbye!")
                self.print_session_stats()
                break

            if user_input.lower() == "memories":
                memories = self.get_memories(user_id)
                if memories:
                    print("\n--- What I remember about you ---")
                    for i, m in enumerate(memories, 1):
                        print(f"{i}. {m['memory']}")
                    print()
                else:
                    print("Agent: I don't have any memories about you yet.\n")
                continue

            if user_input.lower() == "stats":
                self.print_session_stats()
                continue

            response = self.handle_query(user_input, user_id)
            print(f"Agent: {response}\n")


if __name__ == "__main__":
    agent = CustomerSupportAIAgent()

    print("=== Customer Support AI Agent ===")
    customer_id = input("Enter your customer ID: ").strip()

    if not customer_id:
        customer_id = "default_user"

    agent.chat(customer_id)
