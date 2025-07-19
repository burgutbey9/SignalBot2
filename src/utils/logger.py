# src/utils/logger.py - Tuzatilgan versiya
"""
Advanced logging system for SignalBot
Loguru based logger with rotation, monitoring, and Telegram integration
"""
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
from functools import wraps

from loguru import logger
from config.config import config_manager
from utils.helpers import TimeUtils

# Log levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

class TelegramLogHandler:
    """Telegram orqali log yuborish"""
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=100)
        self.enabled = False
        self.min_level = "ERROR"
        
    async def send_log(self, message: str, level: str):
        """Telegram orqali log yuborish"""
        try:
            if not self.enabled:
                return
                
            if LOG_LEVELS.get(level, 0) < LOG_LEVELS.get(self.min_level, 40):
                return
                
            # Add to queue
            await self.queue.put({
                "message": message,
                "level": level,
                "timestamp": TimeUtils.now_uzb()
            })
            
        except asyncio.QueueFull:
            pass  # Skip if queue is full
            
    async def process_queue(self):
        """Queue ni ishlov berish"""
        while True:
            try:
                if not self.queue.empty():
                    log_data = await self.queue.get()
                    
                    # Format message
                    emoji = {
                        "DEBUG": "🔍",
                        "INFO": "ℹ️",
                        "SUCCESS": "✅",
                        "WARNING": "⚠️",
                        "ERROR": "❌",
                        "CRITICAL": "🚨"
                    }.get(log_data["level"], "📝")
                    
                    message = (
                        f"{emoji} <b>{log_data['level']}</b>\n"
                        f"🕐 {log_data['timestamp'].strftime('%H:%M:%S')}\n"
                        f"📝 {log_data['message']}"
                    )
                    
                    # Send via telegram (will be imported dynamically)
                    try:
                        from telegram.bot_interface import telegram_interface
                        await telegram_interface.send_admin_message(message)
                    except:
                        pass
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                await asyncio.sleep(5)

# Global Telegram handler
telegram_handler = TelegramLogHandler()

def setup_logging():
    """Logger ni sozlash"""
    # Remove default handler
    logger.remove()
    
    # Get log directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # File handler - all logs
    logger.add(
        log_dir / "signalbot_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    # Error file handler
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    # Trade logs
    logger.add(
        log_dir / "trades_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="365 days",
        level="INFO",
        filter=lambda record: "trade" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | TRADE | {message}",
        encoding="utf-8"
    )
    
    # Performance logs
    logger.add(
        log_dir / "performance_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="INFO",
        filter=lambda record: "performance" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | PERF | {message}",
        encoding="utf-8"
    )
    
    # Custom Telegram handler
    def telegram_sink(message):
        """Telegram sink"""
        try:
            record = message.record
            if telegram_handler.enabled:
                asyncio.create_task(
                    telegram_handler.send_log(
                        record["message"],
                        record["level"].name
                    )
                )
        except:
            pass
            
    logger.add(
        telegram_sink,
        level="ERROR",
        filter=lambda record: record["level"].no >= LOG_LEVELS["ERROR"]
    )
    
    # Start Telegram handler
    asyncio.create_task(telegram_handler.process_queue())
    
    logger.info("📝 Logging system initialized")

def get_logger(name: str):
    """Logger olish - Fixed to return logger instance"""
    return logger.bind(name=name)

def log_trade(symbol: str, action: str, details: Dict[str, Any]):
    """Savdo logini yozish"""
    logger.bind(trade=True).info(
        f"{symbol} | {action} | {json.dumps(details, ensure_ascii=False)}"
    )

def log_performance(metric: str, value: float, details: Optional[Dict] = None):
    """Performance metrikasini log qilish"""
    msg = f"{metric}: {value}"
    if details:
        msg += f" | {json.dumps(details, ensure_ascii=False)}"
    logger.bind(performance=True).info(msg)

def log_exception(func):
    """Exception logging decorator"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
            
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
            
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

class LogContext:
    """Context manager for structured logging"""
    def __init__(self, **kwargs):
        self.context = kwargs
        
    def __enter__(self):
        self.token = logger.contextualize(**self.context)
        self.token.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.token.__exit__(exc_type, exc_val, exc_tb)

# Utility functions
def enable_telegram_logging(min_level: str = "ERROR"):
    """Telegram logging yoqish"""
    telegram_handler.enabled = True
    telegram_handler.min_level = min_level
    logger.info(f"Telegram logging enabled (min level: {min_level})")

def disable_telegram_logging():
    """Telegram logging o'chirish"""
    telegram_handler.enabled = False
    logger.info("Telegram logging disabled")

def set_log_level(level: str):
    """Log darajasini o'zgartirish"""
    logger.remove()
    setup_logging()
    logger.info(f"Log level changed to: {level}")

# Export all necessary items
__all__ = [
    "logger",
    "get_logger",
    "setup_logging",
    "log_trade",
    "log_performance",
    "log_exception",
    "LogContext",
    "enable_telegram_logging",
    "disable_telegram_logging",
    "set_log_level"
]
