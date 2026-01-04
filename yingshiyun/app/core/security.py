from typing import Dict
from app.shared.utils import sign_params


def verify_signature(params: Dict[str, str], secret: str, signature: str) -> bool:
    return sign_params(params, secret) == signature
