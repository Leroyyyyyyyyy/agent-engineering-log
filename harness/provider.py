"""
Model access, behind one small interface.

The loop should not know which vendor it is talking to, and should not need a
network to be tested. Anything with a .create(messages, tools) method works.

Deliberately NOT wrapping the response: AnthropicProvider hands back the SDK
object unchanged, and FakeProvider mimics its shape. That keeps this file tiny,
at the cost of the loop staying coupled to Anthropic's block layout. Revisit
when a second real vendor shows up, not before.
"""

from dataclasses import dataclass, field
from typing import Any


class AnthropicProvider:
    """The real thing.

    The SDK import stays inside __init__ on purpose: tests never construct this
    class, so they run on a bare interpreter with no anthropic package. Move it
    to the top of the file and test_loop.py stops working outside a venv.
    """

    def __init__(self, model, max_tokens=4096):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def create(self, messages, tools):
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools,
        )


# --- fakes: same interface, no network -------------------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Response:
    stop_reason: str
    content: list = field(default_factory=list)
    usage: Any = None


class FakeProvider:
    """Hands back queued responses and records what it was sent.

    `calls[i]` is the conversation as it looked on the i-th request. Note the
    list() copy - the loop appends to `messages` in place, so storing the
    reference would make every snapshot point at the same final list.
    """

    def __init__(self, responses=None, repeat=None):
        self.responses = list(responses or [])
        self.repeat = repeat
        self.calls = []

    def create(self, messages, tools):
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        if self.repeat is not None:
            return self.repeat
        raise AssertionError("FakeProvider ran out of queued responses")
