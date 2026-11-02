"""
Tests for the LLM-agent helpers: endpoint handling, local-model support, and
the rule that API keys are never written to disk. These need no Tesseract
and make no network calls.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agent  # noqa: E402


def test_base_urls_become_full_endpoints():
    assert agent.normalize_endpoint("http://localhost:11434/v1") == \
        "http://localhost:11434/v1/chat/completions"
    assert agent.normalize_endpoint("http://localhost:11434/v1/") == \
        "http://localhost:11434/v1/chat/completions"
    assert agent.normalize_endpoint("https://api.openai.com/v1/chat/completions") == \
        "https://api.openai.com/v1/chat/completions"
    assert agent.normalize_endpoint("https://api.anthropic.com/v1", "anthropic") == \
        "https://api.anthropic.com/v1/messages"
    assert agent.normalize_endpoint("") == ""


def test_local_detection_is_not_fooled_by_lookalike_hosts():
    assert agent.is_local_url("http://localhost:11434/v1")
    assert agent.is_local_url("http://127.0.0.1:8000/v1")
    assert agent.is_local_url("http://[::1]:8000/v1")
    assert not agent.is_local_url("https://api.openai.com/v1")
    assert not agent.is_local_url("http://localhost.evil.com/v1")
    assert not agent.is_local_url("")


def test_keys_required_for_cloud_but_not_for_local():
    assert agent.key_required("openai", "https://api.openai.com/v1")
    assert agent.key_required("anthropic", "https://api.anthropic.com/v1")
    assert not agent.key_required("ollama", "http://localhost:11434/v1")
    assert not agent.key_required("custom", "http://127.0.0.1:9000/v1")
    assert agent.key_required("custom", "https://example.com/v1")


def test_make_backend_enforces_key_and_model():
    with pytest.raises(agent.AgentError):
        agent.make_backend("openai", "", "some-model")
    with pytest.raises(agent.AgentError):
        agent.make_backend("openai", "key", "")
    b = agent.make_backend("ollama", "", "llama3.1", "http://localhost:11434/v1")
    assert b.url == "http://localhost:11434/v1/chat/completions"


def _fake_reply(seen):
    def fake_post(url, headers, body, secret):
        seen["headers"], seen["body"] = headers, body
        return {"model": "m", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    return fake_post


def test_no_authorization_header_without_a_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "_post", _fake_reply(seen))
    b = agent.make_backend("ollama", "", "llama3.1")
    assert b.ask("hello", lambda n, a: {}) == "hi"
    assert seen["headers"] == {}


def test_key_is_sent_as_a_header_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "_post", _fake_reply(seen))
    b = agent.make_backend("openai", "sk-test", "m", reasoning_effort="none")
    b.ask("hello", lambda n, a: {})
    assert seen["headers"] == {"Authorization": "Bearer sk-test"}
    assert "sk-test" not in json.dumps(seen["body"])
    assert seen["body"]["reasoning_effort"] == "none"


def test_new_chat_resets_history(monkeypatch):
    monkeypatch.setattr(agent, "_post", _fake_reply({}))
    b = agent.make_backend("ollama", "", "llama3.1")
    b.ask("hello", lambda n, a: {})
    assert len(b.msgs) > 1
    b.reset()
    assert len(b.msgs) == 1 and b.msgs[0]["role"] == "system"


def test_settings_never_store_a_key(tmp_path):
    pytest.importorskip("tkinter")
    import ghostdecode_app as ga
    path = tmp_path / "settings.json"
    ga.save_settings({"provider": "openai", "model": "m", "folder": "C:\\v",
                      "api_key": "sk-SECRET", "key": "sk-SECRET"}, str(path))
    text = path.read_text(encoding="utf-8")
    assert "sk-SECRET" not in text
    assert set(json.loads(text)) <= set(ga.SETTINGS_FIELDS)
    assert ga.load_settings(str(path))["provider"] == "openai"
