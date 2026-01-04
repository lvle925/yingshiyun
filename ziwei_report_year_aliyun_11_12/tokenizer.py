import tiktoken
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- 使用 tiktoken (推荐，速度快) ---
try:
    # "cl100k_base" 是 GPT-3.5/4 的编码方式，对很多模型兼容性良好
    # 对于Qwen等模型，这提供了一个足够精确的估算值
    encoding = tiktoken.get_encoding("cl100k_base")
    logger.info("tiktoken tokenizer 'cl100k_base' loaded successfully.")
except Exception as e:
    encoding = None
    logger.error(f"Failed to load tiktoken, token counting will be disabled. Error: {e}")


def count_tokens_for_string(text: str) -> int:
    """计算单个字符串的token数。"""
    if not encoding or not text:
        return 0
    return len(encoding.encode(text))


def count_tokens_for_messages(messages: List[Dict[str, Any]]) -> int:
    """
    计算 LangChain/OpenAI 格式的 messages 列表的总token数。
    遵循 OpenAI 的计算方式，对每个消息和角色都有额外的 token 开销。
    """
    if not encoding or not messages:
        return 0

    num_tokens = 0
    for message in messages:
        # 每个消息有4个固定token: <|start|>{role}\n{content}<|end|>\n
        num_tokens += 4
        for key, value in message.items():
            if value:  # 确保值不为None或空
                num_tokens += count_tokens_for_string(str(value))
            if key == "name":  # 如果有 name 字段，会有一个额外的 token 消耗
                num_tokens += 1
    # 每个回复都会以 <|start|>assistant<|message|> 开头，有3个固定token
    num_tokens += 3
    return num_tokens

# --- 备选方案：使用 transformers (如果 tiktoken 不兼容您的模型) ---
# from transformers import AutoTokenizer
#
# # 加载您模型对应的分词器，例如 Qwen
# # HF_TOKEN = "your_huggingface_token" # 可能需要 Hugging Face Token
# # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", use_auth_token=HF_TOKEN)
#
# def count_tokens_with_transformers(text: str) -> int:
#     if not tokenizer or not text:
#         return 0
#     return len(tokenizer.encode(text))