ENDPOINTS = {
    "general": "http://general-llm:8000",
    "coder": "http://coder-llm:8001",
    "llava": "http://llava-llm:8000",
}


def resolve(model: str, has_image: bool) -> str:
    m = model.lower()
    if has_image or m in {"llava", "vision"}:
        return ENDPOINTS["llava"]
    if "code" in m or "deepseek" in m:
        return ENDPOINTS["coder"]
    return ENDPOINTS["general"]
