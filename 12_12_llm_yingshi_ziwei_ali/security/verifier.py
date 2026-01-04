# security/verifier.py
import hmac
import hashlib
import json
from typing import Dict, Any
#from fastapi import HTTPException, Body
from fastapi import HTTPException, Request
from config import APP_SECRET

def verify_signature(params: Dict[str, Any], app_secret: str) -> bool:
    """使用HMAC-SHA256验证签名。"""
    if 'sign' not in params:
        return False
    client_sign = str(params['sign'])  # 确保是字符串类型
    # 排除签名字段和可选的扩展字段（如weather_info和detailed_intent）
    excluded_fields = {'sign', 'weather_info', 'detailed_intent', 'skip_intent_check', 'is_knowledge_query'}
    sorted_params = dict(sorted({k: str(v) for k, v in params.items() if k not in excluded_fields}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    calculated_sign = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256).hexdigest()
    # 确保两个签名都是字符串类型，并转换为相同格式进行比较
    return client_sign.lower() == calculated_sign.lower()


async def signature_verifier(request: Request) -> dict:
    """
    FastAPI 依赖项，通过直接读取 Request 对象来获取请求体并验证签名。
    这种方式对 Cython 编译更友好。
    """
    try:
        # --- 步骤 1: 读取原始请求体，修复换行符，然后重新解析 ---
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')

        # 核心修改：仅替换不合法的换行符
        body_str_fixed = body_str.replace('\n', r'\n').replace('\r', r'\r')

        # 使用 json.loads 尝试解析修复后的字符串
        request_body = json.loads(body_str_fixed)
    except json.JSONDecodeError as e:
        # 如果请求体不是有效的 JSON，则拒绝请求
        raise HTTPException(status_code=400, detail=f"请求数据格式不正确，无法处理。JSON解析错误: {e.msg}")
    except Exception as e:
        # 其他异常
        raise HTTPException(status_code=400, detail=f"无效的请求体: {e}")

    if not isinstance(request_body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是一个 JSON 对象。")

    if not APP_SECRET:
        raise HTTPException(status_code=500, detail="服务器端未配置签名密钥，无法验证请求。")

    if not verify_signature(request_body, APP_SECRET):
        raise HTTPException(status_code=403, detail="签名无效或错误 (Invalid signature)。")

    # 验证通过，返回请求体字典，供后续路由函数使用
    return request_body
