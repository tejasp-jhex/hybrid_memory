"""Sonnet chatbot with hybrid graph+vector memory.   python chatbot.py [user_id]

Commands: /memory  show graph (incl. history)   /debug  toggle router+context trace
          /wipe    erase this user's memory      /quit
"""
import json
import sys

from hybrid_memory import HybridMemory, config, llm  # noqa: F401  (config re-exported via package import)
from hybrid_memory import config as cfg


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else cfg.USER_ID
    memory = HybridMemory(user_id)
    history: list[dict] = []
    debug = False

    print(f"Memory chatbot ready (user={user_id}). Router={cfg.HAIKU_MODEL}  Chat={cfg.SONNET_MODEL}")
    print("Commands: /memory  /debug  /wipe  /quit")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue

        if user == "/quit":
            break
        if user == "/debug":
            debug = not debug
            print(f"[debug {'on' if debug else 'off'}]")
            continue
        if user == "/memory":
            rows = memory.graph.dump(user_id)
            if not rows:
                print("[graph is empty]")
            for r in rows:
                state = "active " if r["active"] else f"ENDED  {str(r['ended'])[:10]}"
                print(f"  {state}  {r['subject']} -[{r['relation']}]-> {r['object']}")
            print(f"  episodes stored: {memory.vectors.episodes.count()}")
            continue
        if user == "/wipe":
            if input("Erase all memory for this user? (yes/no): ").strip().lower() == "yes":
                memory.wipe()
                history.clear()
                print("[memory wiped]")
            continue

        # ---- memory: route + write + retrieve (never let it break the chat) ----
        try:
            context, info = memory.process_turn(user, history[-5:])
        except Exception as e:
            print(f"[memory error, continuing without memory: {e}]")
            context, info = "", {"plan": {}, "applied": []}

        if debug:
            print("\n--- router plan ---")
            print(json.dumps(info["plan"], indent=2))
            print("--- applied ---")
            for a in info["applied"]:
                print("  " + a)
            print("--- context sent to Sonnet ---")
            print(context or "(none)")
            print("------------------------------")

        history.append({"role": "user", "content": user})
        trimmed = history[-19:] if len(history) > 19 else history  # always starts on a user turn

        print("\nClaude: ", end="", flush=True)
        reply = ""
        for chunk in llm.chat_stream(trimmed, context):
            print(chunk, end="", flush=True)
            reply += chunk
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
