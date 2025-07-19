"""
Configuration Module
Main configuration management and fallback system
"""

from .config import (
    ConfigManager,
    TradingMode,
    APIProvider,
    APIConfig,
    TradingConfig,
    TelegramConfig,
    config_manager
)

from .fallback_config import (
    FallbackManager,
    HealthStatus,
    FallbackStrategy,
    APIHealth,
    FallbackChain,
    fallback_manager
)

__all__ = [
    # Config Manager
    "ConfigManager",
    "TradingMode",
    "APIProvider",
    "APIConfig",
    "TradingConfig",
    "TelegramConfig",
    "config_manager",
    
    # Fallback System
    "FallbackManager",
    "HealthStatus",
    "FallbackStrategy",
    "APIHealth",
    "FallbackChain",
    "fallback_manager"
]

# Auto-load configuration on import
import asyncio

def _load_config():
    """Load configuration synchronously"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If loop is already running, schedule the coroutine
        asyncio.create_task(config_manager.load_config())
    else:
        # If loop is not running, run until complete
        loop.run_until_complete(config_manager.load_config())

# Load config on import
_load_config()
