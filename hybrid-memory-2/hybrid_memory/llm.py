from datetime import datetime, timezone
from typing import Iterator

import anthropic

from . import config
from .prompts import ROUTER_SYSTEM, ROUTER_TOOL, SONNET_SYSTEM, SUMMARY_SYSTEM

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

EMPTY_PLAN = {
    "graph_mutations": [], "narrative": "", "needs_retrieval": False,
    "entities": [], "relations": [], "semantic_query": "",
}


def route(user_msg: str, recent: list[dict]) -> dict:
    """Small-LLM router (Haiku): one call returns both the write plan and the retrieval plan."""
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent[-5:]) or "(none)"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = (
        f"Current date: {today}\n\n"
        f"<recent_messages>\n{transcript}\n</recent_messages>\n\n"
        f"<new_user_message>\n{user_msg}\n</new_user_message>"
    )
    resp = _client.messages.create(
        model=config.HAIKU_MODEL,
        max_tokens=1024,
        system=ROUTER_SYSTEM,
        tools=[ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "route_memory"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return {**EMPTY_PLAN, **block.input}
    return dict(EMPTY_PLAN)


def summarize_profile(facts: str, notes: str) -> str:
    """Reflection step (Haiku): condense facts + notes into a short profile."""
    resp = _client.messages.create(
        model=config.HAIKU_MODEL,
        max_tokens=1024,
        system=SUMMARY_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"<current_facts>\n{facts or '(none)'}\n</current_facts>\n\n<notes>\n{notes or '(none)'}\n</notes>",
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def greeting_prompt(session: dict) -> str:
    """Synthetic first user turn that makes Sonnet greet the user first."""
    now = datetime.now()
    return (
        f"(The user just opened the chat. Local time: {now:%A, %H:%M}. "
        f"Last conversation: {session['gap']}.\n"
        "Greet them warmly in 2-3 sentences. Use their name if you know it, and weave in at most one or two "
        "relevant things from core memory (e.g. something they were working on) as a natural hook. "
        "Don't recite their profile. If you know nothing about them yet, introduce yourself briefly and ask their name.)"
    )


def chat_stream(history: list[dict], memory_context: str) -> Iterator[str]:
    """Big LLM (Sonnet) with the fused memory blocks appended to the system prompt."""
    system = SONNET_SYSTEM + (f"\n\n{memory_context}" if memory_context else "")
    with _client.messages.stream(
        model=config.SONNET_MODEL, max_tokens=1500, system=system, messages=history
    ) as stream:
        for text in stream.text_stream:
            yield text
