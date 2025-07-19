# config/config.py - Tuzatilgan versiya
"""
Crypto Trading Bot Configuration Manager
API keys, settings loader, environment variables handler
"""
import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
from datetime import timezone, timedelta
import logging

class TradingMode(Enum):
    """Trading bot rejimi"""
    LIVE = auto()
    PAPER = auto()
    BACKTEST = auto()
    SIGNAL_ONLY = auto()

class APIProvider(Enum):
    """API provider turlari"""
    ONEINCG = "1inch"
    ALCHEMY = "alchemy"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"
    CLAUDE = "claude"
    NEWSAPI = "newsapi"
    REDDIT = "reddit"
    COINGECKO = "coingecko"
    BINANCE = "binance"

@dataclass
class APIConfig:
    """API konfiguratsiya dataclass"""
    provider: APIProvider
    api_key: str
    secret_key: Optional[str] = None
    endpoint: Optional[str] = None
    rate_limit: int = 100
    timeout: int = 30
    priority: int = 1
    enabled: bool = True
    fallback_providers: List[str] = field(default_factory=list)

@dataclass
class TradingConfig:
    """Trading parametrlari"""
    mode: TradingMode = TradingMode.SIGNAL_ONLY
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])  # pairs -> symbols
    risk_percentage: float = 0.5
    max_daily_loss: float = 3.0
    max_positions: int = 3
    stop_loss_percentage: float = 2.0
    take_profit_percentage: float = 3.0
    trailing_stop: bool = True
    kill_zones: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class RiskManagement:
    """Risk management parametrlari"""
    base_risk_percent: float = 0.5
    max_risk_per_trade: float = 2.0
    max_daily_loss: float = 3.0
    max_positions: int = 3
    use_trailing_stop: bool = True
    
@dataclass
class TelegramConfig:
    """Telegram bot sozlamalari"""
    bot_token: str = ""
    channel_id: str = ""
    admin_ids: List[int] = field(default_factory=list)
    authorized_users: List[int] = field(default_factory=list)  # Added
    signal_format: str = "professional"
    language: str = "uz"
    timezone: str = "Asia/Tashkent"
    
class ConfigManager:
    """Asosiy configuration manager"""
    _instance: Optional['ConfigManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.settings_file = self.config_dir / "settings.json"
        self.env_file = self.base_dir / ".env"
        self._config: Dict[str, Any] = {}
        self._api_configs: Dict[APIProvider, List[APIConfig]] = {}
        self._trading_config: Optional[TradingConfig] = None
        self._telegram_config: Optional[TelegramConfig] = None
        self._risk_management: Optional[RiskManagement] = None  # Added
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Logger sozlash"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    async def load_config(self) -> bool:
        """Barcha konfiguratsiyalarni yuklash"""
        try:
            async with self._lock:
                self._load_env_vars()
                await self._load_settings_json()
                self._init_api_configs()
                self._init_trading_config()
                self._init_telegram_config()
                self._init_risk_management()  # Added
                self.logger.info("✅ Konfiguratsiya muvaffaqiyatli yuklandi")
                return True
        except Exception as e:
            self.logger.error(f"❌ Konfiguratsiya yuklashda xato: {e}")
            return False
            
    def _load_env_vars(self) -> None:
        """Environment variables yuklash"""
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
                        
    async def _load_settings_json(self) -> None:
        """JSON settings yuklash"""
        if self.settings_file.exists():
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = self._get_default_settings()
            await self.save_settings()
            
    def _get_default_settings(self) -> Dict[str, Any]:
        """Default sozlamalar"""
        return {
            "trading": {
                "mode": "SIGNAL_ONLY",
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT"],  # Fixed
                "risk_percentage": 0.5,
                "max_daily_loss": 3.0,
                "max_positions": 3,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 3.0,
                "trailing_stop": True,
                "kill_zones": {
                    "asia": {"start": "02:00", "end": "05:00"},
                    "london": {"start": "10:00", "end": "14:00"},
                    "newyork": {"start": "16:00", "end": "20:00"}
                }
            },
            "risk_management": {  # Added
                "base_risk_percent": 0.5,
                "max_risk_per_trade": 2.0,
                "max_daily_loss": 3.0,
                "max_positions": 3,
                "use_trailing_stop": True
            },
            "telegram": {
                "signal_format": "professional",
                "language": "uz",
                "timezone": "Asia/Tashkent",
                "admin_ids": [],
                "authorized_users": []
            },
            "apis": {
                "order_flow": ["1inch", "alchemy", "thegraph"],
                "sentiment": ["huggingface", "gemini", "claude"],
                "news": ["newsapi", "reddit", "cryptopanic"],
                "market": ["ccxt", "coingecko", "binance"]
            },
            "features": {
                "auto_trading": False,
                "signal_only": True,
                "news_filter": True,
                "whale_alerts": True,
                "ai_learning": True
            }
        }
        
    def _init_api_configs(self) -> None:
        """API konfiguratsiyalarni sozlash"""
        # 1inch API
        self._api_configs[APIProvider.ONEINCG] = [
            APIConfig(
                provider=APIProvider.ONEINCG,
                api_key=os.getenv("ONEINCH_API_KEY", ""),
                endpoint="https://api.1inch.dev/fusion",
                rate_limit=100,
                fallback_providers=["alchemy", "thegraph"]
            )
        ]
        
        # Alchemy API
        self._api_configs[APIProvider.ALCHEMY] = [
            APIConfig(
                provider=APIProvider.ALCHEMY,
                api_key=os.getenv("ALCHEMY_API_KEY", ""),
                endpoint=f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', '')}",
                rate_limit=300
            )
        ]
        
        # HuggingFace API
        self._api_configs[APIProvider.HUGGINGFACE] = [
            APIConfig(
                provider=APIProvider.HUGGINGFACE,
                api_key=os.getenv("HUGGINGFACE_API_KEY", ""),
                endpoint="https://api-inference.huggingface.co/models",
                rate_limit=100,
                fallback_providers=["gemini", "claude"]
            )
        ]
        
        # Gemini APIs (5 ta key)
        gemini_configs = []
        for i in range(1, 6):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key:
                gemini_configs.append(APIConfig(
                    provider=APIProvider.GEMINI,
                    api_key=key,
                    endpoint="https://generativelanguage.googleapis.com/v1beta",
                    rate_limit=60,
                    priority=i
                ))
        self._api_configs[APIProvider.GEMINI] = gemini_configs
        
        # Claude API
        self._api_configs[APIProvider.CLAUDE] = [
            APIConfig(
                provider=APIProvider.CLAUDE,
                api_key=os.getenv("CLAUDE_API_KEY", ""),
                endpoint="https://api.anthropic.com/v1",
                rate_limit=50
            )
        ]
        
    def _init_trading_config(self) -> None:
        """Trading konfiguratsiyasini sozlash"""
        cfg = self._config.get("trading", {})
        self._trading_config = TradingConfig(
            mode=TradingMode[cfg.get("mode", "SIGNAL_ONLY")],
            symbols=cfg.get("symbols", ["BTCUSDT", "ETHUSDT"]),  # Fixed
            risk_percentage=cfg.get("risk_percentage", 0.5),
            max_daily_loss=cfg.get("max_daily_loss", 3.0),
            max_positions=cfg.get("max_positions", 3),
            stop_loss_percentage=cfg.get("stop_loss_percentage", 2.0),
            take_profit_percentage=cfg.get("take_profit_percentage", 3.0),
            trailing_stop=cfg.get("trailing_stop", True),
            kill_zones=cfg.get("kill_zones", {})
        )
        
    def _init_telegram_config(self) -> None:
        """Telegram konfiguratsiyasini sozlash"""
        cfg = self._config.get("telegram", {})
        self._telegram_config = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            channel_id=os.getenv("TELEGRAM_CHANNEL_ID", ""),
            admin_ids=cfg.get("admin_ids", []),
            authorized_users=cfg.get("authorized_users", []),  # Added
            signal_format=cfg.get("signal_format", "professional"),
            language=cfg.get("language", "uz"),
            timezone=cfg.get("timezone", "Asia/Tashkent")
        )
        
    def _init_risk_management(self) -> None:
        """Risk management sozlash"""
        cfg = self._config.get("risk_management", {})
        self._risk_management = RiskManagement(
            base_risk_percent=cfg.get("base_risk_percent", 0.5),
            max_risk_per_trade=cfg.get("max_risk_per_trade", 2.0),
            max_daily_loss=cfg.get("max_daily_loss", 3.0),
            max_positions=cfg.get("max_positions", 3),
            use_trailing_stop=cfg.get("use_trailing_stop", True)
        )
        
    async def save_settings(self) -> bool:
        """Sozlamalarni saqlash"""
        try:
            async with self._lock:
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                self.logger.info("✅ Sozlamalar saqlandi")
                return True
        except Exception as e:
            self.logger.error(f"❌ Sozlamalarni saqlashda xato: {e}")
            return False
            
    def get_api_config(self, provider: APIProvider, index: int = 0) -> Optional[APIConfig]:
        """API konfiguratsiyasini olish"""
        configs = self._api_configs.get(provider, [])
        return configs[index] if index < len(configs) else None
        
    def get_all_api_configs(self, provider: APIProvider) -> List[APIConfig]:
        """Barcha API konfiguratsiyalarini olish"""
        return self._api_configs.get(provider, [])
        
    @property
    def trading(self) -> TradingConfig:
        """Trading konfiguratsiyasi"""
        return self._trading_config or TradingConfig()
        
    @property
    def telegram(self) -> TelegramConfig:
        """Telegram konfiguratsiyasi"""
        return self._telegram_config or TelegramConfig()
        
    @property
    def risk_management(self) -> RiskManagement:
        """Risk management konfiguratsiyasi"""
        return self._risk_management or RiskManagement()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Konfiguratsiya qiymatini olish"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
        
    def set(self, key: str, value: Any) -> None:
        """Konfiguratsiya qiymatini o'rnatish"""
        keys = key.split('.')
        cfg = self._config
        for k in keys[:-1]:
            if k not in cfg:
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value
        
    async def reload(self) -> bool:
        """Konfiguratsiyani qayta yuklash"""
        self.logger.info("🔄 Konfiguratsiya qayta yuklanmoqda...")
        return await self.load_config()

# Global instance
config_manager = ConfigManager()
