"""
This module is used to initialize LLMs used in GroupAI.
"""

import os

# External Packages
from pydantic import BaseModel

# Internal Packages
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import (
    Text_to_Text,
    Image_to_Text,
    Text_to_Text_SO,
    OllamaCore,
)
from llm_agent_toolkit.transcriber import TranscriptionConfig
from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber
from llm_agent_toolkit.encoder import OllamaEncoder

OllamaCore.load_csv("/files/ollama.csv")
CONNECTION_STRING = "http://host.docker.internal:11434"

t2t_model_name = os.environ["T2T_MODEL"] if "T2T_MODEL" in os.environ else "qwen2.5:7b"
i2t_model_name = os.environ["I2T_MODEL"] if "I2T_MODEL" in os.environ else "llava:7b"
t2t_lite_model_name = (
    os.environ["T2T_LITE_MODEL"] if "T2T_LITE_MODEL" in os.environ else "qwen2.5:3b"
)
a2t_model_name = os.environ["A2T_MODEL"] if "A2T_MODEL" in os.environ else "whisper-1"
emb_model_name = (
    os.environ["EMBEDDING_MODEL"]
    if "EMBEDDING_MODEL" in os.environ
    else "bge-m3:latest"
)

# RAG
rag_t2t_config = {
    "system_prompt": os.environ["RAG_PROMPT"],
    "config": ChatCompletionConfig(
        name=t2t_model_name,
        return_n=1,
        max_iteration=10,
        max_tokens=4096,
        max_output_tokens=2048,
        temperature=0.3,
    ),
}
rag_llm = Text_to_Text(connection_string=CONNECTION_STRING, **rag_t2t_config)


# Image Interpreter
class ImageInterpreterResponse(BaseModel):
    title: str
    description: str
    keywords: list[str]


image_interpreter_config = {
    "system_prompt": os.environ["II_PROMPT"],
    "config": ChatCompletionConfig(
        name=i2t_model_name,
        max_tokens=4096,
        max_output_tokens=2048,
        temperature=0.3,
    ),
}
image_interpreter_llm = Image_to_Text(
    connection_string=CONNECTION_STRING, **image_interpreter_config
)

# Summarizer
summarizer_t2t_config = {
    "system_prompt": os.environ["SUMMARY_PROMPT"],
    "config": ChatCompletionConfig(
        name=t2t_model_name,
        return_n=1,
        max_iteration=10,
        max_tokens=4096,
        max_output_tokens=2048,
        temperature=0.3,
    ),
    "tools": None,
}
summarizer_llm = Text_to_Text(
    connection_string=CONNECTION_STRING, **summarizer_t2t_config
)


# Checker
class CheckerResponse(BaseModel):
    reason: str
    details_reason: str
    grounded: bool


checker_t2t_config = {
    "system_prompt": os.environ["CHECKER_PROMPT"],
    "config": ChatCompletionConfig(
        name=t2t_lite_model_name,
        return_n=1,
        max_iteration=10,
        max_tokens=32_768,
        max_output_tokens=1024,
        temperature=0.3,
    ),
}
checker_llm = Text_to_Text_SO(connection_string=CONNECTION_STRING, **checker_t2t_config)

# Transcriber
transcriber_llm = OpenAITranscriber(config=TranscriptionConfig(name=a2t_model_name))

# Embedder
encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=emb_model_name)
