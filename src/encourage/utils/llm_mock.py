"""Utility functions for creating mock LLM responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from encourage.prompts import PromptCollection

if TYPE_CHECKING:
    from encourage.llm.response_wrapper import ResponseWrapper


def create_mock_responses(prompt_collection: PromptCollection) -> list[ChatCompletion]:
    """Create mock ChatCompletion objects."""
    return [
        ChatCompletion(
            id=f"mock-response-{i}",
            choices=[],
            created=0,
            model="mock",
            object="chat.completion",
            system_fingerprint="",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            ),
        )
        for i in range(len(prompt_collection.prompts))
    ]


def create_mock_response_wrapper(prompt_collection: PromptCollection) -> ResponseWrapper:
    """Create a ResponseWrapper with mock responses."""
    from encourage.llm.response_wrapper import ResponseWrapper

    mock_llm_response = create_mock_responses(prompt_collection)
    return ResponseWrapper.from_prompt_collection(mock_llm_response, prompt_collection)
