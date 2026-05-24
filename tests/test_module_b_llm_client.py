"""
文件用途：验证模块 B LLM 客户端的普通/流式响应解析行为。
核心流程：模拟 requests 响应对象，断言非流式与流式路径都能稳定提取 content。
输入输出：输入 pytest monkeypatch 与伪响应，输出断言结果。
依赖说明：依赖 pytest 与 module_b.llm_client。
维护说明：当底层 OpenAI 兼容协议解析策略变化时需同步更新本测试。
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_video_pipeline.modules.module_b.llm_client import call_module_b_llm_chat_detailed


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        json_payload: dict | None = None,
        text: str = "",
        stream_lines: list[bytes | str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._json_payload = json_payload
        self.text = text
        self._stream_lines = list(stream_lines or [])

    def json(self) -> dict:
        if self._json_payload is None:
            raise ValueError("json payload missing")
        return self._json_payload

    def iter_lines(self, decode_unicode: bool = False):  # type: ignore[no-untyped-def]
        del decode_unicode
        for line in self._stream_lines:
            yield line


def test_call_module_b_llm_chat_detailed_should_aggregate_stream_delta_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证启用 stream 时会拼接 SSE delta.content，并返回完整文本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：首个 delta 为空时不应影响后续文本拼接。
    """
    api_key_path = tmp_path / "key.txt"
    api_key_path.write_text("test-key\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["stream"] = kwargs.get("stream")
        return _FakeResponse(
            stream_lines=[
                b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                b'data: {"choices":[{"delta":{"content":" world"}}]}',
                b"data: [DONE]",
            ]
        )

    monkeypatch.setattr("music_video_pipeline.modules.module_b.llm_client.requests.post", _fake_post)

    llm_config = SimpleNamespace(
        base_url="https://example.com/v1",
        api_key_file=str(api_key_path),
        model="demo-model",
        temperature=0.3,
        top_p=0.9,
        stream=True,
        enable_thinking=False,
        retry_times=0,
        timeout_seconds=10,
    )
    response = call_module_b_llm_chat_detailed(
        logger=logging.getLogger("test_module_b_llm_client"),
        llm_config=llm_config,
        messages=[{"role": "user", "content": "hi"}],
        project_root=tmp_path,
    )

    assert captured["stream"] is True
    assert response.content == "Hello world"
    assert response.response_json["choices"][0]["message"]["content"] == "Hello world"
    assert len(response.response_json["chunks"]) == 3


def test_call_module_b_llm_chat_detailed_should_decode_utf8_stream_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证 stream 模式会按 UTF-8 显式解码字节行，避免中文内容乱码。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：服务端返回 bytes 行时，中文应保持原样。
    """
    api_key_path = tmp_path / "key.txt"
    api_key_path.write_text("test-key\n", encoding="utf-8")

    def _fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return _FakeResponse(
            stream_lines=[
                'data: {"choices":[{"delta":{"content":"少女"}}]}'.encode("utf-8"),
                b"data: [DONE]",
            ]
        )

    monkeypatch.setattr("music_video_pipeline.modules.module_b.llm_client.requests.post", _fake_post)

    llm_config = SimpleNamespace(
        base_url="https://example.com/v1",
        api_key_file=str(api_key_path),
        model="demo-model",
        temperature=0.3,
        top_p=0.9,
        stream=True,
        enable_thinking=False,
        retry_times=0,
        timeout_seconds=10,
    )

    response = call_module_b_llm_chat_detailed(
        logger=logging.getLogger("test_module_b_llm_client"),
        llm_config=llm_config,
        messages=[{"role": "user", "content": "hi"}],
        project_root=tmp_path,
    )

    assert response.content == "少女"


def test_call_module_b_llm_chat_detailed_should_skip_invalid_stream_chunk_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能说明：验证 stream 模式遇到非法 JSON chunk 时会跳过坏片段，并继续消费后续文本。
    参数说明：无。
    返回值：无。
    异常说明：断言失败时抛 AssertionError。
    边界条件：坏片段前后仍有合法文本时，应保留合法部分。
    """
    api_key_path = tmp_path / "key.txt"
    api_key_path.write_text("test-key\n", encoding="utf-8")

    def _fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return _FakeResponse(
            stream_lines=[
                b"data: {not-json}",
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                b"data: [DONE]",
            ]
        )

    monkeypatch.setattr("music_video_pipeline.modules.module_b.llm_client.requests.post", _fake_post)

    llm_config = SimpleNamespace(
        base_url="https://example.com/v1",
        api_key_file=str(api_key_path),
        model="demo-model",
        temperature=0.3,
        top_p=0.9,
        stream=True,
        enable_thinking=False,
        retry_times=0,
        timeout_seconds=10,
    )

    response = call_module_b_llm_chat_detailed(
        logger=logging.getLogger("test_module_b_llm_client"),
        llm_config=llm_config,
        messages=[{"role": "user", "content": "hi"}],
        project_root=tmp_path,
    )

    assert response.content == "Hello"
