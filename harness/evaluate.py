"""
Trajectory evaluation: score what the agent DID, not just what it said.

Two numbers per task, deliberately separate:

    trajectory  did it call the right tools, with the right arguments, in order
    answer      does the final text contain the expected answer

They are separate because they fail independently, and the interesting failures
are the ones where they disagree: a right answer off a wrong trajectory is luck
(or a guess from the model's prior, not from the tool output), and a right
trajectory with a wrong answer is a reading-comprehension failure, not a
planning failure. A single blended score hides both.

Scoring uses a weighted longest-common-subsequence rather than exact match, so
an extra exploratory step costs a little and a missing required step costs a
lot - the same shape deepeval's ToolCorrectnessMetric uses.

    python3 evaluate.py                # FakeProvider, deterministic, free
    python3 evaluate.py --real         # real model, costs money
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost import Cost  # noqa: E402
from loop import run  # noqa: E402

HERE = Path(__file__).parent
TASKS_PATH = HERE / "tasks.json"


def normalise(command: str) -> list[str]:
    """
    A command as a comparable token list.

    Compared token-wise, not as a raw string: `ls -la` and `ls  -la` are the
    same call, and `cat config.py` vs `cat ./config.py` differ in a way that
    should cost something but not everything.
    """
    return command.replace("./", "").split()


def call_similarity(actual: dict, expected: dict) -> float:
    """How well one actual tool call matches one expected call, 0..1."""
    if actual["name"] != expected["name"]:
        return 0.0

    want = normalise(expected.get("input_parameters", {}).get("command", ""))
    got = normalise(actual.get("input", {}).get("command", ""))
    if not want:
        return 1.0

    # The executable has to match; the arguments are partial credit. Calling
    # `cat` when `grep` was expected is a different plan, not a typo.
    if not got or got[0] != want[0]:
        return 0.0
    matched = 0
    for token in want[1:]:
        if token in got[1:]:
            matched += 1
    if len(want) == 1:
        return 1.0
    return 0.5 + 0.5 * (matched / (len(want) - 1))


def trajectory_score(actual: list[dict], expected: list[dict]) -> float:
    """
    Weighted LCS over the two call sequences, normalised by what was expected.

    LCS and not set overlap, because ORDER carries meaning: finding the file
    and then reading it is a plan; reading it and then searching for it is not.
    """
    if not expected:
        # Nothing should have been called. Any call is a failure.
        return 1.0 if not actual else 0.0
    if not actual:
        return 0.0

    rows = len(actual) + 1
    cols = len(expected) + 1
    table = [[0.0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            match = table[i - 1][j - 1] + call_similarity(actual[i - 1], expected[j - 1])
            table[i][j] = max(match, table[i - 1][j], table[i][j - 1])

    return min(1.0, table[-1][-1] / len(expected))


def answer_score(final_text: str, expected: str) -> float:
    """Substring containment, case-insensitive. Deliberately crude."""
    if not expected:
        return 1.0
    return 1.0 if expected.lower() in (final_text or "").lower() else 0.0


def collect(events, model: str) -> dict:
    """Drain a run, keeping the trajectory, the final text and the cost."""
    calls = []
    final = ""
    cost = Cost(model=model)
    for event in events:
        if event["type"] == "usage":
            cost.add(event)
        elif event["type"] == "tool_use":
            calls.append(event)
        elif event["type"] in ("done", "stopped", "cancelled"):
            final = event.get("result", "")
    return {"calls": calls, "final": final, "cost": cost}


def scripted_provider(task: dict):
    """
    A FakeProvider that replays the task's OWN expected trajectory.

    This does not measure the model - it measures the harness and the scorer.
    A perfect score here means the plumbing is right; a real model is what
    produces an interesting number.
    """
    from provider import FakeProvider, Response, TextBlock, ToolUseBlock

    class Usage:
        def __init__(self, i, o):
            self.input_tokens = i
            self.output_tokens = o
            self.cache_creation_input_tokens = 0
            self.cache_read_input_tokens = 0

    responses = []
    for index, call in enumerate(task["expected_tools"]):
        responses.append(
            Response(
                "tool_use",
                [
                    ToolUseBlock(
                        f"t{index}", call["name"], dict(call["input_parameters"])
                    )
                ],
                usage=Usage(1000 + index * 200, 40),
            )
        )
    responses.append(
        Response("end_turn", [TextBlock(task["expected_output"])], usage=Usage(1500, 60))
    )
    return FakeProvider(responses=responses)


def real_provider():
    from provider import AnthropicProvider

    return AnthropicProvider(model="claude-haiku-4-5-20251001")


def main() -> int:
    use_real = "--real" in sys.argv
    model = "claude-haiku-4-5-20251001"
    spec = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    fixture = HERE / spec["fixture"]

    # Tasks are phrased relative to the fixture, so the loop's subprocess calls
    # have to resolve there. Anything else and "ls" measures the wrong directory.
    original = os.getcwd()
    os.chdir(fixture)
    print(f"任务集: {len(spec['tasks'])} 条   工作目录: {fixture}")
    print(f"provider: {'real ' + model if use_real else 'scripted (FakeProvider)'}\n")

    rows = []
    try:
        for task in spec["tasks"]:
            provider = real_provider() if use_real else scripted_provider(task)
            result = collect(run(task["input"], provider, max_steps=8), model)

            traj = trajectory_score(result["calls"], task["expected_tools"])
            ans = answer_score(result["final"], task["expected_output"])
            rows.append(
                {
                    "name": task["name"],
                    "trajectory": traj,
                    "answer": ans,
                    "steps": len(result["calls"]),
                    "expected_steps": len(task["expected_tools"]),
                    "cost": result["cost"],
                }
            )
            print(
                f"  {task['name']:<18} trajectory={traj:>5.0%}  answer={ans:>4.0%}  "
                f"steps={len(result['calls'])}/{len(task['expected_tools'])}"
            )
    finally:
        os.chdir(original)

    n = len(rows)
    mean_traj = sum(r["trajectory"] for r in rows) / n
    mean_ans = sum(r["answer"] for r in rows) / n

    print("\n" + "=" * 62)
    print(f"  trajectory 均分   {mean_traj:>6.0%}")
    print(f"  answer 均分       {mean_ans:>6.0%}")

    # The disagreements are the finding, not the averages.
    lucky = [r for r in rows if r["answer"] == 1.0 and r["trajectory"] < 0.6]
    wasted = [r for r in rows if r["answer"] == 0.0 and r["trajectory"] >= 0.6]
    print(f"\n  答对但轨迹错 (蒙对的): {len(lucky)}  {[r['name'] for r in lucky]}")
    print(f"  轨迹对但答错 (读错的): {len(wasted)}  {[r['name'] for r in wasted]}")

    total = Cost(model=model)
    for row in rows:
        c = row["cost"]
        total.steps += c.steps
        total.input_tokens += c.input_tokens
        total.output_tokens += c.output_tokens
        total.cache_write_tokens += c.cache_write_tokens
        total.cache_read_tokens += c.cache_read_tokens

    usd = total.usd()
    print("\n" + "=" * 62)
    print(f"  总步数 {total.steps}   in {total.input_tokens}   out {total.output_tokens}")
    print(f"  缓存 写 {total.cache_write_tokens}  读 {total.cache_read_tokens}")
    print(f"  总成本 {'$%.6f' % usd if usd is not None else '(该模型无定价)'}")
    print(f"  单条平均 {'$%.6f' % (usd / n) if usd is not None else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
