"""Conversation compaction helpers for sequential mini-swe-agent runs.

This module owns the agent compaction mechanism: deciding which messages are
summarized, rendering compactor prompts, validating summaries, and building
deterministic fallback summaries. The orchestration that calls the model still
lives in ``driver.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

MINI_SUMMARY_MARKER = "<mini_sequential_compaction_summary>"
MINI_SUMMARY_END = "</mini_sequential_compaction_summary>"
POST_TASK_COMPACTION_TRANSITION = "Ready for the next task."
DEFAULT_COMPACTION_RESERVE_TOKENS = 20_000
DEFAULT_KEEP_RECENT_TOKENS = 20_000

COMPACTION_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions "
    "in the conversation. ONLY output the structured summary."
)
COMPACTION_SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured
context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session
covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error
messages. Do not include markdown fenced code blocks or transcript role markers.
"""
COMPACTION_UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the
existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error
messages. Do not include markdown fenced code blocks or transcript role markers.
"""
COMPACTION_TURN_PREFIX_SUMMARIZATION_PROMPT = """\
This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent
work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the kept suffix]

Be concise. Do not include markdown fenced code blocks or transcript role
markers.
"""
COMPACTION_REPAIR_PROMPT = """\
The previous compaction summary failed validation:
{validation_error}

Rewrite the candidate below into a valid structured summary. Preserve only
factual information from the candidate. Do not continue the coding task. Do not
include markdown fenced code blocks, [user]/[assistant] transcript markers,
THOUGHT sections, or raw <returncode> tool-output dumps.

Use this EXACT format:

## Goal
[goal]

## Constraints & Preferences
- [constraints or "(none)"]

## Progress
### Done
- [x] [completed work]

### In Progress
- [ ] [current work]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- **[decision]**: [rationale]

## Next Steps
1. [next step]

## Critical Context
- [critical context or "(none)"]

<candidate-summary>
{candidate_summary}
</candidate-summary>
"""
COMPACTION_REQUIRED_HEADERS = (
    "## Goal",
    "## Constraints & Preferences",
    "## Progress",
    "### Done",
    "### In Progress",
    "### Blocked",
    "## Key Decisions",
    "## Next Steps",
    "## Critical Context",
)
TURN_PREFIX_REQUIRED_HEADERS = (
    "## Original Request",
    "## Early Progress",
    "## Context for Suffix",
)


def compaction_system_prompt() -> str:
    return COMPACTION_SUMMARIZATION_SYSTEM_PROMPT


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Conservative dependency-free token estimate for pre-call compaction."""

    chars = 0
    for message in messages:
        chars += len(str(message.get("role", ""))) + 8
        content = message.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        else:
            chars += len(json.dumps(content, ensure_ascii=False, default=str))
    return max(1, chars // 4)


def recent_keep_start_index(
    messages: list[dict[str, Any]],
    *,
    start: int,
    max_messages: int,
    max_tokens: int,
    min_messages: int = 1,
) -> int:
    """Index where the recent-keep window begins.

    Walks back from the end keeping the most recent messages, bounded by BOTH a
    message count (``max_messages``) and an estimated-token budget
    (``max_tokens``), while keeping at least ``min_messages``.
    """

    index = len(messages)
    kept = 0
    tokens = 0
    min_messages = max(0, min_messages)
    max_messages = max(min_messages, max(0, max_messages))
    max_tokens = max(0, max_tokens)
    while index > start:
        candidate = messages[index - 1]
        candidate_tokens = estimate_message_tokens([candidate])
        if kept >= min_messages and (
            kept >= max_messages or tokens + candidate_tokens > max_tokens
        ):
            break
        index -= 1
        kept += 1
        tokens += candidate_tokens
    return index


def adjust_recent_start_for_mini_turn(
    messages: list[dict[str, Any]],
    *,
    start: int,
    recent_start: int,
) -> int:
    if not (start < recent_start < len(messages)):
        return recent_start
    if (
        is_mini_observation_message(messages[recent_start])
        and messages[recent_start - 1].get("role") == "assistant"
    ):
        return recent_start - 1
    return recent_start


def split_messages_for_compaction(
    messages: list[dict[str, Any]],
    *,
    current_task_start_index: int | None,
    keep_recent_messages: int,
    keep_recent_tokens: int,
    min_recent_messages: int = 1,
) -> dict[str, Any]:
    if not messages:
        return {
            "all": messages,
            "system": None,
            "compact": [],
            "keep": [],
            "current_task_message": None,
        }
    system = messages[0] if messages[0].get("role") == "system" else None
    start = 1 if system is not None else 0
    keep_indices: set[int] = set()
    if (
        current_task_start_index is not None
        and 0 <= current_task_start_index < len(messages)
    ):
        keep_indices.add(current_task_start_index)
    recent_start = recent_keep_start_index(
        messages,
        start=start,
        max_messages=keep_recent_messages,
        max_tokens=keep_recent_tokens,
        min_messages=min_recent_messages,
    )
    recent_start = adjust_recent_start_for_mini_turn(
        messages,
        start=start,
        recent_start=recent_start,
    )
    keep_indices.update(range(recent_start, len(messages)))
    turn_prefix_indices: set[int] = set()
    if (
        current_task_start_index is not None
        and current_task_start_index in keep_indices
        and current_task_start_index < recent_start
    ):
        turn_prefix_indices.update(range(current_task_start_index + 1, recent_start))
    compact: list[dict[str, Any]] = []
    history_compact: list[dict[str, Any]] = []
    turn_prefix: list[dict[str, Any]] = []
    keep: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index < start:
            continue
        if index in keep_indices:
            keep.append(message)
        else:
            compact.append(message)
            if index in turn_prefix_indices:
                turn_prefix.append(message)
            else:
                history_compact.append(message)
    current_task_message = (
        messages[current_task_start_index]
        if current_task_start_index is not None
        and 0 <= current_task_start_index < len(messages)
        else None
    )
    return {
        "all": messages,
        "system": system,
        "compact": compact,
        "history_compact": history_compact,
        "turn_prefix": turn_prefix,
        "keep": keep,
        "current_task_message": current_task_message,
        "keep_recent_messages": keep_recent_messages,
        "keep_recent_tokens": keep_recent_tokens,
        "min_recent_messages": min_recent_messages,
    }


def messages_to_transcript(
    messages: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    chunks: list[str] = []
    remaining = max_chars
    for message in messages:
        if remaining <= 0:
            break
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        chunk = f"\n\n[{role}]\n{content}"
        if len(chunk) > remaining:
            chunk = chunk[:remaining] + "\n[truncated for compaction]"
        chunks.append(chunk)
        remaining -= len(chunk)
    return "".join(chunks).strip()


def history_compaction_messages(
    transcript: str,
    *,
    previous_summary: str | None,
) -> list[dict[str, str]]:
    prompt = f"<conversation>\n{transcript}\n</conversation>\n\n"
    if previous_summary:
        prompt += f"<previous-summary>\n{previous_summary.strip()}\n</previous-summary>\n\n"
        prompt += COMPACTION_UPDATE_SUMMARIZATION_PROMPT
    else:
        prompt += COMPACTION_SUMMARIZATION_PROMPT
    return [
        {"role": "system", "content": compaction_system_prompt()},
        {"role": "user", "content": prompt},
    ]


def turn_prefix_compaction_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": compaction_system_prompt()},
        {
            "role": "user",
            "content": (
                f"<conversation>\n{transcript}\n</conversation>\n\n"
                f"{COMPACTION_TURN_PREFIX_SUMMARIZATION_PROMPT}"
            ),
        },
    ]


def validate_compaction_summary(
    summary: str,
    *,
    required_headers: tuple[str, ...] | None = None,
) -> str | None:
    text = summary.strip()
    if not text:
        return "empty summary"
    if "```" in text:
        return "contains markdown fenced code block"
    if re.search(r"(?im)^\s*\[(?:user|assistant|system|tool)(?:\]|\]:)", text):
        return "contains transcript role markers"
    if re.search(r"(?im)^\s*THOUGHT\s*:", text):
        return "contains assistant THOUGHT section"
    if "<returncode>" in text.lower() or "</returncode>" in text.lower():
        return "contains raw returncode tool output"
    if required_headers is None:
        required_headers = COMPACTION_REQUIRED_HEADERS
    missing = [header for header in required_headers if header not in text]
    if missing:
        return f"missing required headers: {', '.join(missing)}"
    return None


def compaction_repair_prompt(
    *,
    candidate_summary: str,
    validation_error: str,
) -> str:
    return COMPACTION_REPAIR_PROMPT.format(
        validation_error=validation_error,
        candidate_summary=candidate_summary,
    )


def bound_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def is_compaction_summary_message(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and MINI_SUMMARY_MARKER in content
    )


def extract_compaction_summary(message: dict[str, Any]) -> str | None:
    if not is_compaction_summary_message(message):
        return None
    content = str(message.get("content", ""))
    start = content.find(MINI_SUMMARY_MARKER)
    if start < 0:
        return None
    start += len(MINI_SUMMARY_MARKER)
    end = content.find(MINI_SUMMARY_END, start)
    if end < 0:
        return None
    summary = content[start:end].strip()
    return summary or None


def latest_compaction_summary(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        summary = extract_compaction_summary(message)
        if summary:
            return summary
    return None


def without_compaction_summary_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [m for m in messages if not is_compaction_summary_message(m)]


def is_mini_observation_message(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and "<returncode>" in content
    )


def sanitize_compaction_snippet(text: str, *, max_chars: int = 300) -> str:
    text = text.replace("```", "'''")
    text = re.sub(r"</?(?:returncode|output)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?im)^\s*\[(?:user|assistant|system|tool)(?:\]|\]:)\s*",
        "",
        text,
    )
    text = re.sub(r"(?im)^\s*THOUGHT\s*:\s*", "", text)
    text = " ".join(text.split())
    return bound_text(text, max_chars=max_chars)


def fallback_compaction_summary(
    messages: list[dict[str, Any]],
    *,
    previous_summary: str | None = None,
) -> str:
    previous_context = ""
    if previous_summary:
        previous_context = sanitize_compaction_snippet(
            previous_summary,
            max_chars=3_000,
        )
    snippets = []
    for message in messages[-6:]:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        snippet = sanitize_compaction_snippet(content)
        if snippet:
            snippets.append(f"- {message.get('role', 'unknown')}: {snippet}")
    compacted_note = f"Compacted messages: {len(messages)}."
    snippet_text = "\n".join(snippets) if snippets else "- (none)"
    previous_text = (
        f"- Previous compacted summary, sanitized and abbreviated: {previous_context}"
        if previous_context
        else "- (none)"
    )
    if not messages:
        compacted_note = "No older conversation content was available to summarize."
    return f"""\
## Goal
- Continue the mini-swe-agent SWE-bench task after context compaction.

## Constraints & Preferences
- Preserve the active task requirements, repository constraints, and the mini command protocol.

## Progress
### Done
- [x] Created a deterministic fallback summary because model compaction was unavailable or invalid.
- [x] {compacted_note}

### In Progress
- [ ] Continue from the retained recent context and current task message.

### Blocked
- (none known from deterministic fallback)

## Key Decisions
- **Fallback compaction**: Used a structured deterministic summary to avoid
  injecting transcript-shaped content into the live context.

## Next Steps
1. Read the retained recent messages and continue the task from there.
2. Re-run focused tests or inspections already suggested by the retained context.

## Critical Context
{previous_text}
- Recent compacted snippets:
{snippet_text}
"""


def fallback_turn_prefix_summary(messages: list[dict[str, Any]]) -> str:
    snippets = []
    for message in messages[-4:]:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        snippet = sanitize_compaction_snippet(content, max_chars=220)
        if snippet:
            snippets.append(f"- {message.get('role', 'unknown')}: {snippet}")
    snippet_text = "\n".join(snippets) if snippets else "- (none)"
    return f"""\
## Original Request
- The current task prefix was compacted; use the retained task message for the exact request.

## Early Progress
{snippet_text}

## Context for Suffix
- The recent suffix is still present in the conversation and contains the exact continuation state.
"""


def extract_file_ops(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    read: set[str] = set()
    modified: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for command in extract_bash_commands(content):
            paths = extract_command_paths(command)
            if not paths:
                continue
            if looks_like_modifying_command(command):
                modified.update(paths)
            elif looks_like_reading_command(command):
                read.update(paths)
    return sorted(read - modified), sorted(modified)


def format_file_operations(
    read_files: list[str],
    modified_files: list[str],
) -> str:
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(
            f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>"
        )
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def extract_bash_commands(content: str) -> list[str]:
    commands = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:bash|sh|shell)?\s*\n(.*?)```",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    return [command for command in commands if command]


def looks_like_modifying_command(command: str) -> bool:
    return bool(
        re.search(
            r"\b(apply_patch|tee|mv|cp|rm|touch)\b|>\s*[\w./-]+|\bsed\s+-i\b|"
            r"\bwrite_text\s*\(|\bopen\s*\([^)]*['\"]w",
            command,
        )
    )


def looks_like_reading_command(command: str) -> bool:
    return bool(
        re.search(
            r"\b(cat|sed|grep|rg|head|tail|less|find|ls|wc|go\s+test|pytest|"
            r"python\s+-m\s+pytest)\b",
            command,
        )
    )


def extract_command_paths(command: str) -> set[str]:
    path_pattern = re.compile(
        r"(?:(?:/app|\.{1,2})/[A-Za-z0-9_./@+-]+|"
        r"[A-Za-z0-9_./@+-]+\."
        r"(?:py|go|js|ts|tsx|jsx|java|rs|rb|php|c|cc|cpp|h|hpp|md|"
        r"yaml|yml|toml|json|ini|cfg|txt|proto|sql))"
    )
    paths = set()
    for raw in path_pattern.findall(command):
        path = raw.strip("'\"`.,:;()[]{}")
        if path and path not in {".", ".."}:
            paths.add(path)
    return paths


def coalesce_consecutive_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent same-role messages (string content) into one.

    Anthropic requires strictly alternating user/assistant turns after the
    system message. Compaction can place the summary (a user turn) next to a
    kept user turn, or leave two user turns adjacent across a compacted gap;
    merging keeps the transcript valid for every provider.
    """

    out: list[dict[str, Any]] = []
    for message in messages:
        if (
            out
            and out[-1].get("role") == message.get("role")
            and isinstance(out[-1].get("content"), str)
            and isinstance(message.get("content"), str)
        ):
            merged = dict(out[-1])
            merged["content"] = out[-1]["content"] + "\n\n" + message["content"]
            out[-1] = merged
        else:
            out.append(message)
    return out


def locate_current_task_index(
    messages: list[dict[str, Any]],
    target: dict[str, Any] | None,
) -> int | None:
    """Locate the current task message after compaction/coalescing."""

    if target is None:
        return None
    for index, message in enumerate(messages):
        if message is target:
            return index
    target_content = target.get("content")
    if isinstance(target_content, str) and target_content:
        for index, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, str) and target_content in content:
                return index
    return None
