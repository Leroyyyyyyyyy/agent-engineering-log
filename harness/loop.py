"""
The agent loop, with the model and the approval policy injected.

Same behaviour as agent-loop/my_agent.py - the only changes are that the
provider and the approval policy come in as arguments instead of being a
module-level client and a hard-coded input() call.

Tools and guardrails still live here. Splitting them out is deferred until
they stop fitting.

The loop is a GENERATOR (`run`). It yields one event per thing that happens
instead of printing, because a print cannot be sent over HTTP and cannot be
tested without capturing stdout. `agent()` is kept as a thin wrapper that
drains the generator and returns the final string, so every existing caller
and all 13 tests keep working unchanged.
"""

from __future__ import annotations

import shlex
import subprocess

TOOLS = [
    {
        "name": "bash",
        "description": "Run a read-only shell command such as ls, cat, pwd, date.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]

# Allowlist: only these executables may run. Anything else is rejected.
ALLOWED_COMMANDS = {
    "ls", "pwd", "echo", "cat", "date", "whoami", "uname",
    "wc", "head", "tail", "grep", "find", "which", "df",
}

# Shell operators enable chaining/redirection, which defeats any allowlist.
SHELL_OPERATORS = ["&&", "||", ";", "|", ">", "<", "`", "$(", "\n"]


class CancellationToken:
    """
    One flag, checked at the top of every step and before every tool runs.

    Stopping an agent is the harness's job, not the HTTP layer's: closing a
    connection does not stop a subprocess that is already running. The token is
    the thing an HTTP handler flips; the loop is the thing that honours it.
    """

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def __bool__(self) -> bool:
        return self.cancelled


def check_command(command: str) -> str | None:
    """Return a rejection reason, or None if the command is allowed."""
    for operator in SHELL_OPERATORS:
        if operator in command:
            return f"Rejected: shell operator '{operator}' is not allowed"

    parts = shlex.split(command)
    if not parts:
        return "Rejected: empty command"

    executable = parts[0]
    if executable not in ALLOWED_COMMANDS:
        return f"Rejected: '{executable}' is not in the allowlist"

    return None


def run_command(command: str) -> str:
    """Run an already-validated command without a shell."""
    try:
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, timeout=30
        )
        return result.stdout or result.stderr or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30s"
    except Exception as error:
        return f"Error: {error}"


def approved(policy: str) -> bool:
    """Decide whether one tool call may run. 'ask' is the only interactive path."""
    if policy == "auto":
        return True
    if policy == "never":
        return False
    return input("  Approve? (y/n): ").strip().lower() == "y"


def collect_text(content: list) -> str:
    """Join every text block. content is a typed array - never index it."""
    texts = []
    for block in content:
        if block.type == "text":
            texts.append(block.text)
    return "\n".join(texts)


def handle_tool_use(block, approve: str) -> dict:
    """Approve, validate and run one tool_use block. Always returns a tool_result."""
    if not approved(approve):
        # Every tool_use needs a matching tool_result, even a refusal
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": "User denied execution of this command.",
            "is_error": True,
        }

    command = block.input["command"]
    rejection = check_command(command)
    if rejection:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": rejection,
            "is_error": True,
        }

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": run_command(command),
    }


def run(
    goal: str,
    provider,
    approve: str = "auto",
    max_steps: int = 10,
    cancel: CancellationToken | None = None,
):
    """
    Run the agent loop, yielding one event per thing that happens.

    Every event is a plain dict with a "type" key, so it can go straight into
    an SSE frame, a JSONL log or a test assertion without a translation layer.

    Event types:
        text         model said something
        tool_use     model asked for a tool, with the arguments it chose
        tool_result  the tool ran (or was refused), is_error says which
        usage        per-step token counts, when the provider reports them
        done         finished normally, result is the final text
        stopped      finished abnormally, reason says why
        cancelled    the caller flipped the token
    """
    messages = [{"role": "user", "content": goal}]

    for step in range(max_steps):
        if cancel:
            yield {"type": "cancelled", "step": step}
            return

        response = provider.create(messages, TOOLS)
        # Append the raw content blocks unchanged - thinking blocks must survive
        messages.append({"role": "assistant", "content": response.content})

        usage = getattr(response, "usage", None)
        if usage is not None:
            # Cache tokens are separate fields, not a subset of input_tokens.
            # Billing needs them apart: writing cache costs MORE than plain
            # input, reading it costs an order of magnitude LESS. Folding them
            # into one number is wrong in both directions at once.
            yield {
                "type": "usage",
                "step": step,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            }

        if response.stop_reason == "end_turn":
            yield {
                "type": "done",
                "result": collect_text(response.content) or "(model returned no text)",
            }
            return

        if response.stop_reason == "max_tokens":
            partial = collect_text(response.content)
            yield {
                "type": "stopped",
                "reason": "max_tokens",
                "result": (
                    f"Stopped: hit max_tokens on step {step + 1}.\n"
                    f"Partial output:\n{partial}"
                ),
            }
            return

        if response.stop_reason != "tool_use":
            # refusal, pause_turn, or anything added to the enum later
            yield {
                "type": "stopped",
                "reason": str(response.stop_reason),
                "result": f"Stopped: unhandled stop_reason '{response.stop_reason}'",
            }
            return

        # One tool_result per tool_use, all in a single user message.
        # The pairing is not optional: a tool_use with no matching tool_result
        # is a 400 from the API on the next request.
        tool_results = []
        for block in response.content:
            if block.type == "text":
                yield {"type": "text", "text": block.text}
            elif block.type == "tool_use":
                yield {"type": "tool_use", "name": block.name, "input": block.input}

                if cancel:
                    # Still emit a tool_result, or the history is unsendable.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Cancelled by user before execution.",
                            "is_error": True,
                        }
                    )
                    messages.append({"role": "user", "content": tool_results})
                    yield {"type": "cancelled", "step": step}
                    return

                result = handle_tool_use(block, approve)
                tool_results.append(result)
                yield {
                    "type": "tool_result",
                    "content": result["content"],
                    "is_error": result.get("is_error", False),
                }

        messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "stopped",
        "reason": "max_steps",
        "result": f"Stopped: reached the {max_steps}-step limit without finishing",
    }


def agent(goal: str, provider, approve: str = "auto", max_steps: int = 10) -> str:
    """Drain run() and return only the final string. The pre-SSE interface."""
    final = "(loop produced no result)"
    for event in run(goal, provider, approve=approve, max_steps=max_steps):
        if event["type"] == "text":
            print(f"\n{event['text']}")
        elif event["type"] == "tool_use":
            print(f"\n  -> Tool: {event['name']}({event['input']})")
        elif event["type"] == "tool_result" and event["is_error"]:
            print(f"  {event['content']}")
        elif event["type"] in ("done", "stopped"):
            final = event["result"]
    return final


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    from provider import AnthropicProvider

    load_dotenv(find_dotenv())
    model = AnthropicProvider(model="claude-haiku-4-5-20251001")

    print("Harness (type 'quit' to exit)")
    while True:
        task = input("\nYou: ").strip()
        if task.lower() in ("exit", "quit", "q", ""):
            break
        print(f"\nAgent: {agent(task, model, approve='ask')}")
