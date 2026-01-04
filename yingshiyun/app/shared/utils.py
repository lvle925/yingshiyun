import hashlib
import hmac
from typing import Dict


def sign_params(params: Dict[str, str], secret: str) -> str:
    """Simple HMAC-SHA256 signer placeholder."""
    sorted_items = sorted((k, str(v)) for k, v in params.items() if k != "sign")
    payload = "".join(f"{k}{v}" for k, v in sorted_items)
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
