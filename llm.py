"""Shared LLM bindings for talking to the local llama-server."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


SERVER_URL = "http://localhost:8080"


class ChatLlamaServer(ChatOpenAI):
    """ChatOpenAI that preserves llama.cpp / DeepSeek-style `reasoning_content`."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for gen, choice in zip(result.generations, raw.get("choices", [])):
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            if reasoning and isinstance(gen.message, AIMessage):
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return result
