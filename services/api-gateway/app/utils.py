"""Utility functions for the API Gateway."""

import re
from urllib.parse import unquote


def validate_path(path: str) -> bool:
    """
    Validate request path to prevent traversal attacks.

    Returns:
        bool: True if path is safe, False otherwise
    """
    # Decode URL encoding first
    decoded_path = unquote(path)

    # Check for path traversal patterns
    traversal_patterns = [
        r"\.\.",  # Direct ..
        r"%2e%2e",  # URL encoded ..
        r"%252e%252e",  # Double encoded ..
        r"\.\%2e",  # Mixed encoding
        r"%2e\.",  # Mixed encoding
        r"\.\./",  # ../ pattern
        r"\.\.//",  # ../
        r"/\.\./",  # /../
    ]

    for pattern in traversal_patterns:
        if re.search(pattern, decoded_path, re.IGNORECASE):
            return False

    # Check for null bytes
    if "\x00" in decoded_path or "%00" in decoded_path:
        return False

    # Check for absolute paths
    if decoded_path.startswith("/"):
        decoded_path = decoded_path[1:]

    # Ensure path only contains allowed characters
    if not re.match(r"^[a-zA-Z0-9/_\-\.]+$", decoded_path):
        return False

    return True
