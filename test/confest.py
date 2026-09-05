"""
Runs before any test file is collected. Sets dummy environment variables
so config.py's fail-fast validation (which runs at import time, by
design — you want the real app to crash loudly if a key is missing)
doesn't block unit tests that never actually touch the network.

These values are never used to make a real API call in the unit tests
below -- they only need to be non-empty strings.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")