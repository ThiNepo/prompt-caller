import shutil
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_caller import PromptCaller


@pytest.fixture
def prompt_dir() -> Path:
    root = Path(".test_runtime")
    if root.exists():
        shutil.rmtree(root)
    path = root / "prompts"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if root.exists():
            shutil.rmtree(root)


@pytest.fixture
def write_prompt(prompt_dir: Path):
    def _write(name: str, frontmatter: str, body: str) -> Path:
        content = f"---\n{frontmatter.strip()}\n---\n{body.strip()}\n"
        target = prompt_dir / f"{name}.prompt"
        target.write_text(content, encoding="utf-8")
        return target

    return _write


@pytest.fixture
def prompt_caller(prompt_dir: Path) -> PromptCaller:
    return PromptCaller(promptPath=str(prompt_dir))


class ChatStub:
    def __init__(self, response=None):
        self.response = response if response is not None else AIMessage(content="stub")
        self.config = None
        self.messages = None
        self.structured_schema = None

    def with_structured_output(self, schema):
        self.structured_schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.response


@pytest.fixture
def chat_stub_factory():
    created = []

    def _factory(response=None):
        stub = ChatStub(response=response)
        created.append(stub)
        return stub

    _factory.created = created
    return _factory


@pytest.fixture
def fake_langchain_chat():
    return GenericFakeChatModel(messages=iter([AIMessage(content="fake-ok")]))
