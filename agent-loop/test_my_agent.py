"""
Characterization test for my_agent.py.

Not a correctness test - it pins down what the loop does RIGHT NOW, so the
stage-04 refactor into a harness can be proven behaviour-preserving.

Every assertion below is a behaviour that must survive the refactor. If one
starts failing, either you changed something on purpose (update the test and
say why) or you broke it by accident (that is the whole point of this file).

Run:
    uv run --directory /Users/dld/AIeatwld/agentic-ai-engineering/01-foundations/05-agent-loop \
        python /Users/dld/AIeatwld/agent-engineering-log/agent-loop/test_my_agent.py

No API key needed - the client is replaced by a fake.
"""

import builtins
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import my_agent


# --- fakes -------------------------------------------------------------


class TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input


class FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class FakeMessages:
    """Hands back queued responses and records the messages it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # Shallow copy: the list is what gets appended to, so this freezes
        # the conversation as it looked at this call.
        self.calls.append(list(kwargs["messages"]))
        if self.responses:
            return self.responses.pop(0)
        return self.last_response

    @property
    def last_response(self):
        return self._repeat

    def repeat_forever(self, response):
        self._repeat = response


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def run_agent(goal, responses, approvals, repeat=None):
    """Run my_agent.agent() against fakes. Returns (result, fake_messages)."""
    fake = FakeClient(responses)
    if repeat is not None:
        fake.messages.repeat_forever(repeat)

    answers = list(approvals)

    def fake_input(prompt=""):
        return answers.pop(0) if answers else "y"

    real_client = my_agent.client
    real_input = builtins.input
    my_agent.client = fake
    builtins.input = fake_input
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = my_agent.agent(goal)
    finally:
        my_agent.client = real_client
        builtins.input = real_input

    return result, fake.messages


# --- tests -------------------------------------------------------------


def test_end_turn_returns_joined_text():
    result, _ = run_agent(
        "hi",
        [FakeResponse("end_turn", [TextBlock("part one"), TextBlock("part two")])],
        [],
    )
    assert result == "part one\npart two", result


def test_end_turn_without_text_returns_placeholder():
    result, _ = run_agent("hi", [FakeResponse("end_turn", [])], [])
    assert result == "(model returned no text)", result


def test_max_tokens_returns_partial_output():
    result, _ = run_agent(
        "hi", [FakeResponse("max_tokens", [TextBlock("half a sen")])], []
    )
    assert result.startswith("Stopped: hit max_tokens on step 1."), result
    assert "half a sen" in result, result


def test_unknown_stop_reason_is_not_swallowed():
    result, _ = run_agent("hi", [FakeResponse("refusal", [])], [])
    assert result == "Stopped: unhandled stop_reason 'refusal'", result


def test_every_tool_use_gets_exactly_one_tool_result():
    responses = [
        FakeResponse(
            "tool_use",
            [
                TextBlock("let me check"),
                ToolUseBlock("id_a", "bash", {"command": "echo a"}),
                ToolUseBlock("id_b", "bash", {"command": "echo b"}),
            ],
        ),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    result, messages = run_agent("hi", responses, ["y", "y"])
    assert result == "done", result

    # Second API call saw: user goal, assistant blocks, one user message of results
    second_call = messages.calls[1]
    assert len(second_call) == 3, second_call
    tool_results = second_call[2]["content"]
    assert second_call[2]["role"] == "user", second_call[2]["role"]
    assert len(tool_results) == 2, tool_results
    assert [r["tool_use_id"] for r in tool_results] == ["id_a", "id_b"], tool_results


def test_assistant_content_is_appended_unchanged():
    blocks = [TextBlock("thinking out loud"), ToolUseBlock("id_a", "bash", {"command": "echo a"})]
    responses = [
        FakeResponse("tool_use", blocks),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    _, messages = run_agent("hi", responses, ["y"])
    appended = messages.calls[1][1]["content"]
    assert appended is blocks, "assistant content was copied or filtered, not passed through"


def test_denied_approval_still_returns_a_tool_result():
    responses = [
        FakeResponse("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})]),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    _, messages = run_agent("hi", responses, ["n"])
    result_block = messages.calls[1][2]["content"][0]
    assert result_block["is_error"] is True, result_block
    assert result_block["tool_use_id"] == "id_a", result_block


def test_shell_operator_is_rejected():
    responses = [
        FakeResponse("tool_use", [ToolUseBlock("id_a", "bash", {"command": "ls && rm -rf /"})]),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    _, messages = run_agent("hi", responses, ["y"])
    result_block = messages.calls[1][2]["content"][0]
    assert result_block["is_error"] is True, result_block
    assert "shell operator" in result_block["content"], result_block


def test_executable_outside_allowlist_is_rejected():
    responses = [
        FakeResponse("tool_use", [ToolUseBlock("id_a", "bash", {"command": "rm -rf /tmp/x"})]),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    _, messages = run_agent("hi", responses, ["y"])
    result_block = messages.calls[1][2]["content"][0]
    assert "not in the allowlist" in result_block["content"], result_block


def test_allowed_command_actually_runs():
    responses = [
        FakeResponse("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo hello"})]),
        FakeResponse("end_turn", [TextBlock("done")]),
    ]
    _, messages = run_agent("hi", responses, ["y"])
    result_block = messages.calls[1][2]["content"][0]
    assert result_block["content"].strip() == "hello", result_block
    assert "is_error" not in result_block, result_block


def test_max_steps_caps_the_loop():
    forever = FakeResponse("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})])
    result, messages = run_agent("hi", [], ["y"] * 50, repeat=forever)
    assert result == f"Stopped: reached the {my_agent.MAX_STEPS}-step limit without finishing", result
    assert len(messages.calls) == my_agent.MAX_STEPS, len(messages.calls)


# --- runner ------------------------------------------------------------


def main():
    tests = []
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            tests.append((name, value))

    failures = []
    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failures.append((name, error))
            print(f"  FAIL  {name}\n        {error}")
        except Exception as error:
            failures.append((name, error))
            print(f"  ERROR {name}\n        {type(error).__name__}: {error}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
