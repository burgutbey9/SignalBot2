"""
API Clients Module
Trading APIs, AI clients, news and social media APIs
"""

from .trading_apis import (
    TradingAPIManager,
    OneInchAPI,
    AlchemyAPI,
    CoinGeckoAPI,
    CCXTManager
)

from .ai_clients import (
    AIClientManager,
    HuggingFaceClient,
    GeminiClient,
    ClaudeClient
)

from .news_social import (
    NewsSocialAPI,
    NewsAPIClient,
    RedditClient,
    CryptoPanicClient
)

from .telegram_client import (
    TelegramClient,
    MessageType,
    KeyboardBuilder
)

__all__ = [
    # Trading APIs
    "TradingAPIManager",
    "OneInchAPI",
    "AlchemyAPI",
    "CoinGeckoAPI",
    "CCXTManager",
    
    # AI Clients
    "AIClientManager",
    "HuggingFaceClient",
    "GeminiClient",
    "ClaudeClient",
    
    # News & Social
    "NewsSocialAPI",
    "NewsAPIClient",
    "RedditClient",
    "CryptoPanicClient",
    
    # Telegram
    "TelegramClient",
    "MessageType",
    "KeyboardBuilder"
]
