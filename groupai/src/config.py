import os
from llm_agent_toolkit import ChatCompletionConfig

master: int = int(os.environ["MASTER_TLG_ID"])
assert master != 0

CONNECTION_STRING = "http://host.docker.internal:11434"

t2t_model_name = os.environ["T2T_MODEL"] if "T2T_MODEL" in os.environ else "qwen2.5:7b"
i2t_model_name = os.environ["I2T_MODEL"] if "I2T_MODEL" in os.environ else "llava:7b"
emb_model_name = (
    os.environ["EMBEDDING_MODEL"]
    if "EMBEDDING_MODEL" in os.environ
    else "bge-m3:latest"
)
a2t_model_name = os.environ["A2T_MODEL"] if "A2T_MODEL" in os.environ else "whisper-1"

main_t2t_config = {
    "system_prompt": os.environ["RAG_PROMPT"],
    "config": ChatCompletionConfig(
        name=t2t_model_name,
        return_n=1,
        max_iteration=10,
        max_tokens=8096,
        max_output_tokens=4096,
        temperature=0.5,
    ),
    "tools": None,
}
main_i2t_config = {
    "system_prompt": os.environ["II_PROMPT"],
    "config": ChatCompletionConfig(
        name=i2t_model_name,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.3,
    ),
}
summarizer_t2t_config = {
    "system_prompt": os.environ["SUMMARY_PROMPT"],
    "config": ChatCompletionConfig(
        name=t2t_model_name,
        return_n=1,
        max_iteration=10,
        max_tokens=2048,
        temperature=0.3,
    ),
    "tools": None,
}
