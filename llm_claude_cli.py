"""
Custom LightRAG LLM function using the Claude Code CLI (claude -p).
No Anthropic API key needed — uses your existing Claude Code session.
"""
import asyncio
import shutil
from typing import Any

# Limit concurrent Claude CLI subprocesses to avoid overloading the session
_SEMAPHORE = asyncio.Semaphore(4)

# Resolve full path to claude executable once at import time.
# On Windows, `claude` is a .cmd wrapper — create_subprocess_exec cannot find
# .cmd files via PATH, so we need the absolute path.
_CLAUDE_BIN = shutil.which("claude")
if _CLAUDE_BIN is None:
    raise FileNotFoundError(
        "Could not find 'claude' on PATH. "
        "Install Claude Code CLI: npm install -g @anthropic-ai/claude-code"
    )


async def claude_cli_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    """
    Drop-in LLM function for LightRAG that calls `claude -p` via subprocess.
    Prompt is passed via stdin to support long entity-extraction prompts.

    Usage in LightRAG:
        from llm_claude_cli import claude_cli_complete
        rag = LightRAG(llm_model_func=claude_cli_complete, ...)
    """
    # Strip LightRAG-internal kwargs that should not be forwarded
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)

    model: str | None = kwargs.pop("model", None)

    cmd = [
        _CLAUDE_BIN,
        "--print",
        "--no-session-persistence",
        "--output-format", "text",
        "--tools", "",           # disable all tools (Bash, Edit, etc.) for clean LLM-only calls
        "--dangerously-skip-permissions",  # skip interactive trust dialog in automated mode
    ]

    if model:
        cmd += ["--model", model]

    # Both system_prompt and prompt are merged into stdin to avoid
    # Windows 32KB command-line limit (entity extraction prompts are huge).
    if system_prompt:
        stdin_text = f"[System Instructions]\n{system_prompt}\n\n[User Input]\n{prompt}"
    else:
        stdin_text = prompt

    async with _SEMAPHORE:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=stdin_text.encode("utf-8"))

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {err}")

    return stdout.decode("utf-8", errors="replace").strip()
