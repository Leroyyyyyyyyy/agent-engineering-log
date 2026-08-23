"""
Characterization test for harness/loop.py.

Same 11 assertions as agent-loop/test_my_agent.py, ported to the injected
interface. If these pass, the provider/approval refactor did not change
behaviour.

Note what disappeared: no monkeypatching of a module-level client, and no
patching of builtins.input except in the one test that exercises the 'ask'
policy on purpose. That shrinkage IS the result of the refactor.

Run (no API key, no anthropic package needed - AnthropicProvider imports
the SDK lazily and these tests never construct one):
    python3 /Users/dld/AIeatwld/agent-engineering-log/harness/test_loop.py
"""

import builtins
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loop import agent
from provider import FakeProvider, Response, TextBlock, ToolUseBlock


def run(goal, responses=None, approve="auto", repeat=None, max_steps=10):
    """Run the loop against a fake provider with stdout muted."""
    provider = FakeProvider(responses, repeat=repeat)
    with contextlib.redirect_stdout(io.StringIO()):
        result = agent(goal, provider, approve=approve, max_steps=max_steps)
    return result, provider


# --- tests -------------------------------------------------------------


def test_end_turn_returns_joined_text():
    result, _ = run("hi", [Response("end_turn", [TextBlock("part one"), TextBlock("part two")])])
    assert result == "part one\npart two", result


def test_end_turn_without_text_returns_placeholder():
    result, _ = run("hi", [Response("end_turn", [])])
    assert result == "(model returned no text)", result


def test_max_tokens_returns_partial_output():
    result, _ = run("hi", [Response("max_tokens", [TextBlock("half a sen")])])
    assert result.startswith("Stopped: hit max_tokens on step 1."), result
    assert "half a sen" in result, result


def test_unknown_stop_reason_is_not_swallowed():
    result, _ = run("hi", [Response("refusal", [])])
    assert result == "Stopped: unhandled stop_reason 'refusal'", result


def test_every_tool_use_gets_exactly_one_tool_result():
    responses = [
        Response(
            "tool_use",
            [
                TextBlock("let me check"),
                ToolUseBlock("id_a", "bash", {"command": "echo a"}),
                ToolUseBlock("id_b", "bash", {"command": "echo b"}),
            ],
        ),
        Response("end_turn", [TextBlock("done")]),
    ]
    result, provider = run("hi", responses)
    assert result == "done", result

    # Second request saw: user goal, assistant blocks, one user message of results
    second_call = provider.calls[1]
    assert len(second_call) == 3, second_call
    assert second_call[2]["role"] == "user", second_call[2]["role"]
    tool_results = second_call[2]["content"]
    assert len(tool_results) == 2, tool_results
    assert [r["tool_use_id"] for r in tool_results] == ["id_a", "id_b"], tool_results


def test_assistant_content_is_appended_unchanged():
    blocks = [TextBlock("thinking out loud"), ToolUseBlock("id_a", "bash", {"command": "echo a"})]
    responses = [Response("tool_use", blocks), Response("end_turn", [TextBlock("done")])]
    _, provider = run("hi", responses)
    appended = provider.calls[1][1]["content"]
    assert appended is blocks, "assistant content was copied or filtered, not passed through"


def test_denied_approval_still_returns_a_tool_result():
    responses = [
        Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})]),
        Response("end_turn", [TextBlock("done")]),
    ]
    _, provider = run("hi", responses, approve="never")
    result_block = provider.calls[1][2]["content"][0]
    assert result_block["is_error"] is True, result_block
    assert result_block["tool_use_id"] == "id_a", result_block


def test_ask_policy_still_reads_from_stdin():
    """The one remaining interactive path. Everything else is now non-interactive."""
    responses = [
        Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})]),
        Response("end_turn", [TextBlock("done")]),
    ]
    real_input = builtins.input
    builtins.input = lambda prompt="": "n"
    try:
        _, provider = run("hi", responses, approve="ask")
    finally:
        builtins.input = real_input
    result_block = provider.calls[1][2]["content"][0]
    assert result_block["is_error"] is True, result_block


def test_shell_operator_is_rejected():
    responses = [
        Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "ls && rm -rf /"})]),
        Response("end_turn", [TextBlock("done")]),
    ]
    _, provider = run("hi", responses)
    result_block = provider.calls[1][2]["content"][0]
    assert result_block["is_error"] is True, result_block
    assert "shell operator" in result_block["content"], result_block


def test_executable_outside_allowlist_is_rejected():
    responses = [
        Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "rm -rf /tmp/x"})]),
        Response("end_turn", [TextBlock("done")]),
    ]
    _, provider = run("hi", responses)
    result_block = provider.calls[1][2]["content"][0]
    assert "not in the allowlist" in result_block["content"], result_block


def test_allowed_command_actually_runs():
    responses = [
        Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo hello"})]),
        Response("end_turn", [TextBlock("done")]),
    ]
    _, provider = run("hi", responses)
    result_block = provider.calls[1][2]["content"][0]
    assert result_block["content"].strip() == "hello", result_block
    assert "is_error" not in result_block, result_block


def test_max_steps_caps_the_loop():
    forever = Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})])
    result, provider = run("hi", repeat=forever, max_steps=10)
    assert result == "Stopped: reached the 10-step limit without finishing", result
    assert len(provider.calls) == 10, len(provider.calls)


def test_max_steps_is_a_parameter_now():
    """New: max_steps was a module constant, so it could not be varied per run."""
    forever = Response("tool_use", [ToolUseBlock("id_a", "bash", {"command": "echo a"})])
    result, provider = run("hi", repeat=forever, max_steps=3)
    assert result == "Stopped: reached the 3-step limit without finishing", result
    assert len(provider.calls) == 3, len(provider.calls)


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
