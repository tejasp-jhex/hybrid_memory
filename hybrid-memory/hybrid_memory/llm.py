from datetime import datetime, timezone
from typing import Iterator

import anthropic

from . import config
from .prompts import ROUTER_SYSTEM, ROUTER_TOOL, SONNET_SYSTEM

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


def chat_stream(history: list[dict], memory_context: str) -> Iterator[str]:
    """Big LLM (Sonnet) with the fused memory block appended to the system prompt."""
    system = SONNET_SYSTEM + (f"\n\n{memory_context}" if memory_context else "")
    with _client.messages.stream(
        model=config.SONNET_MODEL, max_tokens=1500, system=system, messages=history
    ) as stream:
        for text in stream.text_stream:
            yield text
