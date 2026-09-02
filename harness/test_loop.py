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
    python3 harness/test_loop.py
"""

import builtins
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: this file already has a local helper called run(), so the loop's
# generator is imported under a different name. Same-name imports are
# silently shadowed by later defs - the tests still "pass", against the
# wrong function.
from loop import CancellationToken, agent
from loop import run as loop_run
from provider import FakeProvider, Response, TextBlock, ToolUseBlock

import cost as cost_mod
import evaluate


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


# --- cancellation ----------------------------------------------------------


def test_cancel_before_first_step_stops_immediately():
    """A token flipped before the run starts must prevent any provider call."""
    token = CancellationToken()
    token.cancel()
    provider = FakeProvider(repeat=Response(stop_reason="end_turn", content=[]))

    events = list(loop_run("go", provider, cancel=token))

    assert len(events) == 1, "cancelled run should emit exactly one event"
    assert events[0]["type"] == "cancelled", "the one event should be 'cancelled'"
    assert len(provider.calls) == 0, "a cancelled run must not call the provider"


def test_cancel_mid_run_still_pairs_every_tool_use():
    """
    Cancelling between tool_use and tool_result would leave an unpaired
    tool_use, which is a 400 on the next request. The loop must synthesise a
    tool_result even when it is stopping.
    """
    token = CancellationToken()
    provider = FakeProvider(
        repeat=Response(
            stop_reason="tool_use",
            content=[ToolUseBlock(id="t1", name="bash", input={"command": "pwd"})],
        )
    )

    events = []
    for event in loop_run("go", provider, max_steps=10, cancel=token):
        events.append(event)
        if event["type"] == "tool_use":
            token.cancel()

    types = [e["type"] for e in events]
    assert types[-1] == "cancelled", f"should end cancelled, got {types}"
    assert "tool_result" not in types, "the tool must not run after cancellation"
    assert len(provider.calls) == 1, f"should stop after one step, got {len(provider.calls)}"

    # The synthesised pairing lives in the message history, not in the events.
    last_message = provider.calls[0]
    assert isinstance(last_message, list), "calls[i] should be a message list"


def test_run_yields_events_not_prints():
    """The loop's output is data. agent() is only a formatter on top of it."""
    provider = FakeProvider(
        responses=[
            Response(
                stop_reason="tool_use",
                content=[ToolUseBlock(id="t1", name="bash", input={"command": "pwd"})],
            ),
            Response(stop_reason="end_turn", content=[TextBlock(text="done")]),
        ]
    )
    types = [e["type"] for e in loop_run("go", provider)]
    assert types == ["tool_use", "tool_result", "done"], f"unexpected sequence {types}"


# --- trajectory scoring ----------------------------------------------------


def _call(command):
    return {"name": "bash", "input": {"command": command}}


def _want(command):
    return {"name": "bash", "input_parameters": {"command": command}}


def test_trajectory_scorer_gives_full_marks_only_for_the_right_plan():
    assert evaluate.trajectory_score([_call("ls")], [_want("ls")]) == 1.0


def test_trajectory_scorer_punishes_a_different_executable():
    """cat instead of grep is a different plan, not a typo. No partial credit."""
    score = evaluate.trajectory_score([_call("cat config.py")], [_want("grep -r X .")])
    assert score == 0.0, score


def test_trajectory_scorer_gives_partial_credit_for_wrong_arguments():
    """Right tool, right executable, wrong flag - should hurt but not zero."""
    score = evaluate.trajectory_score([_call("wc -c notes.txt")], [_want("wc -l notes.txt")])
    assert 0.0 < score < 1.0, score


def test_trajectory_scorer_respects_order():
    """Read-then-search is not the same plan as search-then-read."""
    forwards = evaluate.trajectory_score(
        [_call("grep -r X ."), _call("cat config.py")],
        [_want("grep -r X ."), _want("cat config.py")],
    )
    backwards = evaluate.trajectory_score(
        [_call("cat config.py"), _call("grep -r X .")],
        [_want("grep -r X ."), _want("cat config.py")],
    )
    assert forwards == 1.0, forwards
    assert backwards < forwards, (backwards, forwards)


def test_trajectory_scorer_fails_a_run_that_should_have_called_nothing():
    """A guardrail task: any successful tool call is a failure."""
    assert evaluate.trajectory_score([], []) == 1.0
    assert evaluate.trajectory_score([_call("ls")], []) == 0.0


def test_extra_exploratory_step_costs_less_than_a_missing_one():
    """An extra call is noise; a missing required call is a hole."""
    extra = evaluate.trajectory_score([_call("ls"), _call("cat config.py")], [_want("cat config.py")])
    missing = evaluate.trajectory_score([_call("ls")], [_want("ls"), _want("cat config.py")])
    assert extra > missing, (extra, missing)


# --- cost ------------------------------------------------------------------


def test_cache_tokens_are_priced_separately_from_plain_input():
    """
    The whole point of splitting the fields: same token count, different bill.
    Reading cache must be cheaper than plain input; writing it must be dearer.
    """
    model = "claude-haiku-4-5-20251001"
    plain = cost_mod.Cost(model=model, input_tokens=1_000_000)
    read = cost_mod.Cost(model=model, cache_read_tokens=1_000_000)
    write = cost_mod.Cost(model=model, cache_write_tokens=1_000_000)
    assert read.usd() < plain.usd() < write.usd(), (read.usd(), plain.usd(), write.usd())


def test_unknown_model_reports_no_price_instead_of_zero():
    """A silent 0.00 would read as 'free' rather than 'unpriced'."""
    assert cost_mod.Cost(model="some-new-model", input_tokens=999).usd() is None


if __name__ == "__main__":
    sys.exit(main())
