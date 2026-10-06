"""Parse artifact-producing responses: full bot or SEARCH/REPLACE diff, full conventions or a delta."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.S)


class ParseError(ValueError):
    pass


def tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip("\n") if m else None


def strip_fence(block: str) -> str:
    m = _FENCE.search(block)
    return (m.group(1) if m else block).strip("\n") + "\n"


def apply_search_replace(code: str, diff: str) -> str:
    blocks = _BLOCK.findall(diff)
    if not blocks:
        raise ParseError("bot_diff contains no SEARCH/REPLACE blocks")
    for old, new in blocks:
        if old not in code:
            raise ParseError(f"SEARCH text not found in current bot.py: {old[:80]!r}")
        code = code.replace(old, new, 1)
    return code


def parse_artifact(text: str, current_code: str | None, current_conventions: str | None) -> tuple[str, str, str]:
    """Return (code, conventions, edit_mode) where edit_mode is 'full' or 'diff'."""
    bot = tag(text, "bot")
    diff = tag(text, "bot_diff")
    if bot is not None:
        code, mode = strip_fence(bot), "full"
    elif diff is not None:
        if current_code is None:
            raise ParseError("bot_diff given but there is no current bot.py")
        code, mode = apply_search_replace(current_code, diff), "diff"
    else:
        raise ParseError("response has neither <bot> nor <bot_diff>")
    conv = tag(text, "conventions")
    delta = tag(text, "conventions_delta")
    if conv is not None:
        conventions = conv.strip() + "\n"
    elif delta is not None:
        conventions = (current_conventions or "").rstrip() + "\n\n" + delta.strip() + "\n"
    else:
        conventions = current_conventions or ""
    return code, conventions, mode
