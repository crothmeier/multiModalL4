"""JWT authentication and security utilities."""

import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "llm-platform")
JWT_ISSUER = os.getenv("JWT_ISSUER", "multimodal-stack")

if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is required")

security = HTTPBearer()


async def verify_jwt(credentials: HTTPAuthorizationCredentials) -> dict:
    """
    Verify JWT token and return decoded claims.

    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"verify_exp": True},
        )

        # Validate required claims
        if "sub" not in payload:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")

        return payload

    except JWTError as e:
        raise HTTPException(
            status_code=401, detail=f"Invalid authentication credentials: {str(e)}"
        )


async def get_current_user(request: Request) -> Optional[dict]:
    """Extract and verify JWT from request headers."""
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=auth_header.replace("Bearer ", "")
    )

    return await verify_jwt(credentials)
