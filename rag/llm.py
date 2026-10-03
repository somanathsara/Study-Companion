import anthropic
import config

_client = None


def complete(system: str, user: str, max_tokens: int = 600, model: str | None = None) -> str:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = _client.messages.create(
        model=model or config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
