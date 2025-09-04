"""
Model Client Utilities
Helper functions for creating properly configured model clients
"""
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import ModelInfo
from config.settings import settings

def create_gemini_model_client() -> OpenAIChatCompletionClient:
    """Create a properly configured Gemini model client for AutoGen"""
    
    model_info = ModelInfo(
        family="openai",  # Using openai family since we're using OpenAI-compatible API
        vision=True,
        function_calling=True,
        json_output=True
    )
    
    return OpenAIChatCompletionClient(
        model="gemini-2.0-flash-exp",
        api_key=settings.google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info=model_info
    )

def create_openai_model_client(model: str = "gpt-4") -> OpenAIChatCompletionClient:
    """Create an OpenAI model client (for comparison/fallback)"""
    
    return OpenAIChatCompletionClient(
        model=model,
        api_key=settings.openai_api_key if hasattr(settings, 'openai_api_key') else None
    )
