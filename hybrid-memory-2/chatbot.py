"""Sonnet chatbot with hybrid graph+vector memory.   python chatbot.py [user_id]

The bot greets you first, using everything it knows about you (core memory).

Commands: /memory  graph incl. ended history    /core    show core memory sent every turn
          /reflect rebuild profile summary now  /debug   toggle router + context trace
          /wipe    erase this user's memory     /quit
"""
import json
import sys

from hybrid_memory import HybridMemory, llm
from hybrid_memory import config as cfg


def stream_reply(history: list[dict], context: str) -> str:
    reply = ""
    for chunk in llm.chat_stream(history, context):
        print(chunk, end="", flush=True)
        reply += chunk
    print()
    return reply


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else cfg.USER_ID
    memory = HybridMemory(user_id)
    debug = False

    print(f"Memory chatbot (user={user_id}). Router={cfg.HAIKU_MODEL}  Chat={cfg.SONNET_MODEL}")
    print("Commands: /memory /core /reflect /debug /wipe /quit\n")

    # ---- Claude speaks first, with all known info as core memory ----
    session = memory.start_session()
    history: list[dict] = [{"role": "user", "content": llm.greeting_prompt(session)}]
    print("Claude: ", end="", flush=True)
    try:
        history.append({"role": "assistant", "content": stream_reply(history, memory.core_text())})
    except Exception as e:
        print(f"[greeting failed: {e}]")
        history = []

    try:
        while True:
            user = input("\nYou: ").strip()
            if not user:
                continue

            if user == "/quit":
                break
            if user == "/debug":
                debug = not debug
                print(f"[debug {'on' if debug else 'off'}]")
                continue
            if user == "/core":
                print(memory.core_text() or "[core memory is empty]")
                continue
            if user == "/reflect":
                print(memory.consolidate() or "[nothing to summarise yet]")
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

            # ---- memory: route + write + core + retrieve (never let it break the chat) ----
            recent = history[1:][-5:]  # skip the synthetic greeting prompt
            try:
                context, info = memory.process_turn(user, recent)
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
            history.append({"role": "assistant", "content": stream_reply(trimmed, context)})
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        if memory.dirty:
            print("[updating your profile summary...]")
            try:
                memory.consolidate()
            except Exception as e:
                print(f"[reflection skipped: {e}]")


if __name__ == "__main__":
    main()
