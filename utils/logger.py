"""
Logger Setup and Error Handling
O'zbekcha log messages, rotating files, structured logging
"""
import logging
import sys
import os
import json
import traceback
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from enum import Enum
import asyncio
from functools import wraps

class LogLevel(Enum):
    """Log darajalari"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogCategory(Enum):
    """Log kategoriyalari"""
    BOT = "bot"
    API = "api"
    TRADING = "trading"
    ANALYSIS = "analysis"
    TELEGRAM = "telegram"
    SYSTEM = "system"
    ERROR = "error"

class ColoredFormatter(logging.Formatter):
    """Rangli console output formatter"""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "category": getattr(record, 'category', 'system'),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process_id": os.getpid()
        }
        
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
            
        if record.exc_info:
            log_data['exception'] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_data, ensure_ascii=False)

class LoggerManager:
    """Markaziy logger boshqaruv tizimi"""
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False
    
    @classmethod
    def setup(cls, log_level: str = "INFO", log_dir: str = "logs") -> None:
        """Logger tizimini sozlash"""
        if cls._initialized: return
        cls._initialized = True
        
        # Log papkasini yaratish
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Root logger sozlash
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handlers for different categories
        categories = [LogCategory.BOT, LogCategory.API, LogCategory.TRADING, LogCategory.ERROR]
        for category in categories:
            file_handler = RotatingFileHandler(
                log_path / f"{category.value}.log",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            
            # Category-specific logger
            cat_logger = logging.getLogger(category.value)
            cat_logger.addHandler(file_handler)
            cat_logger.setLevel(logging.DEBUG)
            cls._loggers[category.value] = cat_logger
            
        # Error-specific handler
        error_handler = TimedRotatingFileHandler(
            log_path / "error.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)

def get_logger(name: str, category: Optional[LogCategory] = None) -> logging.Logger:
    """Logger olish"""
    if not LoggerManager._initialized:
        LoggerManager.setup()
        
    if category and category.value in LoggerManager._loggers:
        logger = LoggerManager._loggers[category.value].getChild(name)
    else:
        logger = logging.getLogger(name)
        
    # O'zbekcha xabarlar uchun wrapper
    class UzbekLogger:
        def __init__(self, logger: logging.Logger):
            self._logger = logger
            
        def debug(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
            self._logger.debug(msg, *args, extra=self._add_extra(extra), **kwargs)
            
        def info(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
            self._logger.info(msg, *args, extra=self._add_extra(extra), **kwargs)
            
        def warning(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
            self._logger.warning(msg, *args, extra=self._add_extra(extra), **kwargs)
            
        def error(self, msg: str, *args, exc_info=None, extra: Optional[Dict] = None, **kwargs):
            self._logger.error(msg, *args, exc_info=exc_info, extra=self._add_extra(extra), **kwargs)
            
        def critical(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
            self._logger.critical(msg, *args, extra=self._add_extra(extra), **kwargs)
            
        def _add_extra(self, extra: Optional[Dict]) -> Dict:
            """Extra ma'lumotlar qo'shish"""
            result = {'category': category.value if category else 'system'}
            if extra: result['extra_data'] = extra
            return result
            
    return UzbekLogger(logger)

def log_exception(logger: logging.Logger, msg: str = "Kutilmagan xatolik") -> Callable:
    """Exception decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{msg}: {str(e)}", exc_info=True, extra={'function': func.__name__, 'args': str(args)[:100]})
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{msg}: {str(e)}", exc_info=True, extra={'function': func.__name__, 'args': str(args)[:100]})
                raise
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class ErrorTracker:
    """Xatoliklarni kuzatish va tahlil qilish"""
    def __init__(self, max_errors: int = 1000):
        self.errors: list = []
        self.max_errors = max_errors
        self.error_counts: Dict[str, int] = {}
        
    def track_error(self, error_type: str, error_msg: str, context: Optional[Dict] = None) -> None:
        """Xatolikni qayd qilish"""
        error_data = {
            "timestamp": datetime.now(),
            "type": error_type,
            "message": error_msg,
            "context": context or {}
        }
        
        self.errors.append(error_data)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
            
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Xatoliklar xulosasi"""
        if not self.errors: return {"total": 0, "types": {}}
        
        last_hour_errors = [e for e in self.errors if (datetime.now() - e["timestamp"]).seconds < 3600]
        
        return {
            "total": len(self.errors),
            "last_hour": len(last_hour_errors),
            "types": dict(self.error_counts),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            "latest": self.errors[-1] if self.errors else None
        }

# Global error tracker
error_tracker = ErrorTracker()

# O'zbekcha log xabarlari
LOG_MESSAGES = {
    # Bot messages
    "bot_start": "🤖 SignalBot ishga tushdi",
    "bot_stop": "🛑 SignalBot to'xtatildi",
    "bot_pause": "⏸️ SignalBot pauza rejimida",
    "bot_resume": "▶️ SignalBot davom ettirildi",
    
    # Trading messages
    "signal_generated": "📊 Yangi signal yaratildi: {pair}",
    "order_placed": "💰 Buyurtma joylashtirildi: {order_type} {pair}",
    "stop_loss_hit": "🛑 Stop Loss ishga tushdi: {pair} ({loss}%)",
    "take_profit_hit": "🎯 Take Profit erishildi: {pair} ({profit}%)",
    "position_closed": "📈 Pozitsiya yopildi: {pair}",
    
    # API messages
    "api_request": "🌐 API so'rov: {provider} - {endpoint}",
    "api_success": "✅ API javob: {provider} ({time}ms)",
    "api_error": "❌ API xato: {provider} - {error}",
    "api_fallback": "🔄 Fallback API: {from_provider} -> {to_provider}",
    
    # Analysis messages
    "analysis_start": "🔍 Tahlil boshlandi: {type}",
    "analysis_complete": "✅ Tahlil tugadi: {type} ({time}s)",
    "whale_detected": "🐋 Whale harakati aniqlandi: {amount} {coin}",
    "sentiment_change": "📊 Sentiment o'zgarishi: {old} -> {new}",
    
    # System messages
    "config_loaded": "⚙️ Konfiguratsiya yuklandi",
    "database_connected": "💾 Database ulandi",
    "error_critical": "🚨 KRITIK XATO: {error}",
    "performance_warning": "⚠️ Performance ogohlantirish: {metric}"
}

def log_message(key: str, **kwargs) -> str:
    """O'zbekcha log xabarini olish"""
    template = LOG_MESSAGES.get(key, key)
    try: return template.format(**kwargs)
    except: return f"{key}: {kwargs}"
