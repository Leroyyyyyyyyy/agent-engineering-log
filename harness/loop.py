"""
The agent loop, with the model and the approval policy injected.

Same behaviour as agent-loop/my_agent.py - the only changes are that the
provider and the approval policy come in as arguments instead of being a
module-level client and a hard-coded input() call.

Tools and guardrails still live here. Splitting them out is deferred until
they stop fitting.
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
    print(f"\n  -> Tool: {block.name}({block.input})")

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
        print(f"  {rejection}")
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


def agent(goal: str, provider, approve: str = "auto", max_steps: int = 10) -> str:
    """Run the agent loop until the model stops asking for tools."""
    messages = [{"role": "user", "content": goal}]

    for step in range(max_steps):
        response = provider.create(messages, TOOLS)
        # Append the raw content blocks unchanged - thinking blocks must survive
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return collect_text(response.content) or "(model returned no text)"

        if response.stop_reason == "max_tokens":
            partial = collect_text(response.content)
            return f"Stopped: hit max_tokens on step {step + 1}.\nPartial output:\n{partial}"

        if response.stop_reason != "tool_use":
            # refusal, pause_turn, or anything added to the enum later
            return f"Stopped: unhandled stop_reason '{response.stop_reason}'"

        # One tool_result per tool_use, all in a single user message
        tool_results = []
        for block in response.content:
            if block.type == "text":
                print(f"\n{block.text}")
            elif block.type == "tool_use":
                tool_results.append(handle_tool_use(block, approve))

        messages.append({"role": "user", "content": tool_results})

    return f"Stopped: reached the {max_steps}-step limit without finishing"


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
