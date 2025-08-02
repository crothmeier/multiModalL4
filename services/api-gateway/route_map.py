ENDPOINTS = {
    "general": "http://general-llm:8000",
    "specialized": "http://specialized-llm:8001",
}


def resolve(model: str, has_image: bool) -> str:
    m = model.lower()
    if has_image or m in {"llava", "vision"}:
        return ENDPOINTS["specialized"]
    if "code" in m or "deepseek" in m:
        return ENDPOINTS["specialized"]
    return ENDPOINTS["general"]
