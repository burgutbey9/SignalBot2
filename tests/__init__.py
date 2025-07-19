"""
SignalBot Tests Package
Unit tests, integration tests, fixtures
"""
import sys
import os
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_CONFIG = {
    "database_url": "sqlite+aiosqlite:///:memory:",
    "log_level": "DEBUG",
    "log_dir": "tests/logs",
    "test_mode": True
}

# Mock API keys for testing
TEST_API_KEYS = {
    "telegram": "test_bot_token",
    "binance": {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret"
    },
    "gemini": ["test_key_1", "test_key_2"],
    "huggingface": "test_hf_token",
    "newsapi": "test_news_key"
}

# Test symbols
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Fixtures path
FIXTURES_PATH = Path(__file__).parent / "fixtures"
