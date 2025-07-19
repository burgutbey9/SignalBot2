# src/telegram/__init__.py
"""
Telegram bot interface module
"""

# Lazy import to avoid circular dependencies
_telegram_interface = None

def get_telegram_interface():
    """Get telegram interface instance"""
    global _telegram_interface
    if _telegram_interface is None:
        from .bot_interface import TelegramInterface
        _telegram_interface = TelegramInterface()
    return _telegram_interface

# For backward compatibility
@property
def telegram_interface():
    return get_telegram_interface()

__all__ = ["get_telegram_interface", "telegram_interface"]
