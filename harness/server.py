"""
SSE over the agent loop.

The interesting part is not FastAPI - it is that stopping works. Closing an
HTTP connection does NOT stop a subprocess the agent already launched, so
/stop does not touch the connection at all: it flips a CancellationToken that
the loop itself checks. The HTTP layer owns the socket; the harness owns the run.

Why SSE and not WebSocket: this stream is one-directional (server -> client)
and rides on plain HTTP, so proxies, curl and EventSource all work with no
handshake. Control messages go over ordinary POSTs instead. WebSocket would
buy bidirectionality we do not need and cost a second protocol to operate.

Run:
    .venv/bin/uvicorn server:app --port 8000
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cost import Cost
from loop import CancellationToken, run
from provider import FakeProvider, TextBlock, ToolUseBlock, Response
from tracing import instrument

app = FastAPI(title="harness")

# run_id -> token. In-process only: a second worker would not see these, which
# is exactly the problem stage 04 layer 4 (durable state) has to solve.
RUNS: dict[str, CancellationToken] = {}


class RunRequest(BaseModel):
    goal: str
    approve: str = "auto"
    max_steps: int = 10
    fake: bool = False
    model: str = "claude-haiku-4-5-20251001"
    # "short" finishes in two steps; "loop" keeps asking for a slow tool so a
    # /stop call has something to actually interrupt.
    fake_mode: str = "short"


def build_provider(request: RunRequest):
    """A real provider, or a scripted one so the endpoint is testable offline."""
    if request.fake and request.fake_mode == "loop":
        return FakeProvider(
            repeat=Response(
                stop_reason="tool_use",
                content=[
                    TextBlock(text="Still searching."),
                    ToolUseBlock(
                        id="t1",
                        name="bash",
                        input={"command": "find /usr/share -name *.txt"},
                    ),
                ],
            )
        )
    if request.fake:
        return FakeProvider(
            responses=[
                Response(
                    stop_reason="tool_use",
                    content=[
                        TextBlock(text="Let me look at the directory."),
                        ToolUseBlock(id="t1", name="bash", input={"command": "pwd"}),
                    ],
                ),
                Response(stop_reason="end_turn", content=[TextBlock(text="Done.")]),
            ]
        )
    from provider import AnthropicProvider

    return AnthropicProvider(model="claude-haiku-4-5-20251001")


def sse(event: dict) -> str:
    """
    One SSE frame.

    The blank line terminates the frame - without it the client buffers
    forever and the stream looks hung rather than broken.
    """
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/run")
def start_run(request: RunRequest) -> StreamingResponse:
    """Stream one agent run. The run_id comes first so the client can stop it."""
    run_id = uuid.uuid4().hex[:12]
    token = CancellationToken()
    RUNS[run_id] = token
    provider = build_provider(request)

    def stream():
        yield sse({"type": "run_started", "run_id": run_id})
        cost = Cost(model=request.model)
        # instrument() is a pass-through generator, so tracing costs the stream
        # nothing structurally: back-pressure and event order are unchanged.
        events = instrument(
            run(
                request.goal,
                provider,
                approve=request.approve,
                max_steps=request.max_steps,
                cancel=token,
            ),
            goal=request.goal,
            model=request.model,
        )
        try:
            for event in events:
                if event["type"] == "usage":
                    cost.add(event)
                yield sse(event)
        except Exception as error:
            # A crash must still reach the client as a frame, not as a silently
            # truncated stream that looks identical to a network failure.
            yield sse({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            yield sse({"type": "run_finished", "run_id": run_id, **cost.as_dict()})
            RUNS.pop(run_id, None)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Without this an nginx in front would buffer the whole response and
        # deliver it at the end, which defeats the point of streaming.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/stop/{run_id}")
def stop_run(run_id: str) -> dict:
    """Flip the token. The loop stops itself; we never kill the connection."""
    token = RUNS.get(run_id)
    if token is None:
        return {"stopped": False, "reason": "unknown or already finished run_id"}
    token.cancel()
    return {"stopped": True, "run_id": run_id}


@app.get("/runs")
def list_runs() -> dict:
    """Which runs are still in flight."""
    return {"running": list(RUNS)}
