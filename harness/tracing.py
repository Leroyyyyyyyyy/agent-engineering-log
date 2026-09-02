"""
OTel GenAI instrumentation over the loop's event stream.

Nothing here invents a format. Every attribute name comes from the GenAI
semantic conventions (opentelemetry.semconv gen_ai_attributes), so the traces
land in any OTel backend without a custom adapter - the answer becomes "I
instrumented to the GenAI semconv", not "I wrote a trace format".

Three nested spans, matching the three things that actually happen:

    invoke_agent          one run
      chat                one model request
        execute_tool      one tool call

`instrument(run(...))` wraps the generator and passes every event through
unchanged, so the caller (agent(), server.py) does not know it is being traced.

Point it at a collector - agentevals has one built in:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

from __future__ import annotations

import json
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "harness")

_provider: TracerProvider | None = None


def setup(console: bool = False) -> trace.Tracer:
    """
    Install a tracer provider once.

    OTEL_EXPORTER_OTLP_ENDPOINT unset means no exporter is wired: the spans are
    still created, so the instrumentation stays exercised by the tests without
    needing a collector on the other end.
    """
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        if console:
            _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_provider)
    return trace.get_tracer("harness.loop")


def shutdown() -> None:
    """Flush pending spans. A short-lived script exits before the batch fires."""
    if _provider is not None:
        _provider.force_flush()


def _as_parts(event: dict) -> dict:
    """
    One loop event as a GenAI parts-based message.

    The parts schema is the newer of the two agentevals accepts, and it is the
    one that can express a tool call without stuffing JSON into a content string.
    """
    if event["type"] == "text":
        return {"role": "assistant", "parts": [{"type": "text", "content": event["text"]}]}
    return {
        "role": "assistant",
        "parts": [
            {"type": "tool_call", "name": event["name"], "arguments": event["input"]}
        ],
    }


def instrument(events, goal: str, model: str = "unknown", provider_name: str = "anthropic"):
    """
    Wrap a run() generator in spans, yielding every event through untouched.

    Written as a generator rather than a callback so back-pressure is preserved:
    the caller still controls when the next step happens.
    """
    tracer = setup()
    run_span = tracer.start_span(
        "invoke_agent",
        attributes={
            G.GEN_AI_OPERATION_NAME: G.GenAiOperationNameValues.INVOKE_AGENT.value,
            G.GEN_AI_PROVIDER_NAME: provider_name,
            G.GEN_AI_REQUEST_MODEL: model,
            G.GEN_AI_INPUT_MESSAGES: json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": goal}]}],
                ensure_ascii=False,
            ),
        },
    )

    step_span = None
    tool_span = None
    outputs: list[dict] = []
    total_in = 0
    total_out = 0

    def end_step():
        nonlocal step_span, outputs
        if step_span is None:
            return
        step_span.set_attribute(
            G.GEN_AI_OUTPUT_MESSAGES, json.dumps(outputs, ensure_ascii=False)
        )
        step_span.end()
        step_span = None
        outputs = []

    try:
        with trace.use_span(run_span, end_on_exit=False):
            for event in events:
                kind = event["type"]

                if kind == "usage":
                    # A usage event marks the start of a step: the model just
                    # answered, so open the chat span it belongs to.
                    end_step()
                    step_span = tracer.start_span(
                        "chat",
                        attributes={
                            G.GEN_AI_OPERATION_NAME: G.GenAiOperationNameValues.CHAT.value,
                            G.GEN_AI_PROVIDER_NAME: provider_name,
                            G.GEN_AI_REQUEST_MODEL: model,
                            G.GEN_AI_USAGE_INPUT_TOKENS: event["input_tokens"] or 0,
                            G.GEN_AI_USAGE_OUTPUT_TOKENS: event["output_tokens"] or 0,
                            G.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: (
                                event.get("cache_creation_input_tokens") or 0
                            ),
                            G.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: (
                                event.get("cache_read_input_tokens") or 0
                            ),
                        },
                    )
                    total_in += event["input_tokens"] or 0
                    total_out += event["output_tokens"] or 0

                elif kind in ("text", "tool_use"):
                    # Only text goes into output.messages. The tool call is
                    # already fully described by its own execute_tool span, and
                    # a consumer that reads both counts every call twice -
                    # measured: 3 calls showed up as 6 in agentevals.
                    if kind == "text":
                        outputs.append(_as_parts(event))
                    else:
                        parent = step_span or run_span
                        with trace.use_span(parent, end_on_exit=False):
                            tool_span = tracer.start_span(
                                f"execute_tool {event['name']}",
                                attributes={
                                    G.GEN_AI_OPERATION_NAME: (
                                        G.GenAiOperationNameValues.EXECUTE_TOOL.value
                                    ),
                                    G.GEN_AI_TOOL_NAME: event["name"],
                                    G.GEN_AI_TOOL_TYPE: "function",
                                    G.GEN_AI_TOOL_CALL_ARGUMENTS: json.dumps(
                                        event["input"], ensure_ascii=False
                                    ),
                                },
                            )

                elif kind == "tool_result":
                    if tool_span is not None:
                        tool_span.set_attribute(
                            G.GEN_AI_TOOL_CALL_RESULT, str(event["content"])[:4000]
                        )
                        # is_error is what separates "the tool ran and said no"
                        # from "the tool ran". Both are tool_results.
                        tool_span.set_attribute("harness.tool.is_error", event["is_error"])
                        if event["is_error"]:
                            tool_span.set_status(trace.Status(trace.StatusCode.ERROR))
                        tool_span.end()
                        tool_span = None

                elif kind in ("done", "stopped", "cancelled"):
                    # The loop reports the final answer on the done event, not
                    # as a text event, so without this the last assistant
                    # message never reaches the trace and agentText is empty.
                    if event.get("result"):
                        outputs.append(
                            {
                                "role": "assistant",
                                "parts": [{"type": "text", "content": event["result"]}],
                            }
                        )
                    end_step()
                    run_span.set_attribute(
                        G.GEN_AI_RESPONSE_FINISH_REASONS, [event.get("reason", kind)]
                    )
                    if kind != "done":
                        run_span.set_status(trace.Status(trace.StatusCode.ERROR, kind))

                yield event
    finally:
        # Whatever happened - normal end, cancellation, exception - no span may
        # be left open, or the trace is unreadable in every backend.
        if tool_span is not None:
            tool_span.end()
        end_step()
        # NOT gen_ai.usage.* on the run span: backends sum that attribute over
        # every span in the trace, so a total here is added to the per-step
        # numbers it already contains - measured: 5700 tokens reported as 11400.
        run_span.set_attribute("harness.run.input_tokens", total_in)
        run_span.set_attribute("harness.run.output_tokens", total_out)
        run_span.end()
