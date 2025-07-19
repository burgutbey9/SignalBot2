"""
Crypto Trading Bot Configuration Manager
API keys, settings loader, environment variables handler with validation
"""
import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
from datetime import timezone, timedelta
import logging
import re
from cryptography.fernet import Fernet
import base64

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
    
    def __post_init__(self):
        """API key validation"""
        if not self.api_key or self.api_key == "":
            raise ValueError(f"API key for {self.provider.value} cannot be empty")

@dataclass
class TradingConfig:
    """Trading parametrlari"""
    mode: TradingMode = TradingMode.SIGNAL_ONLY
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    risk_percentage: float = 0.5
    max_daily_loss: float = 3.0
    max_positions: int = 3
    stop_loss_percentage: float = 2.0
    take_profit_percentage: float = 3.0
    trailing_stop: bool = True
    kill_zones: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Trading config validation"""
        if self.risk_percentage <= 0 or self.risk_percentage > 5:
            raise ValueError("Risk percentage must be between 0 and 5")
        if self.max_daily_loss <= 0 or self.max_daily_loss > 20:
            raise ValueError("Max daily loss must be between 0 and 20")
        if self.max_positions <= 0 or self.max_positions > 10:
            raise ValueError("Max positions must be between 1 and 10")
    
@dataclass
class TelegramConfig:
    """Telegram bot sozlamalari"""
    bot_token: str = ""
    channel_id: str = ""
    admin_ids: List[int] = field(default_factory=list)
    signal_format: str = "professional"
    language: str = "uz"
    timezone: str = "Asia/Tashkent"
    
    def __post_init__(self):
        """Telegram config validation"""
        if not self.bot_token:
            raise ValueError("Telegram bot token is required")
        if not re.match(r'^\d+:[A-Za-z0-9_-]+$', self.bot_token):
            raise ValueError("Invalid Telegram bot token format")

class ConfigValidationError(Exception):
    """Config validation xatosi"""
    pass

class SecurityManager:
    """API key encryption/decryption"""
    
    def __init__(self):
        self.encryption_enabled = os.getenv("ENCRYPT_API_KEYS", "false").lower() == "true"
        self._cipher_suite = None
        
        if self.encryption_enabled:
            self._init_encryption()
    
    def _init_encryption(self):
        """Encryption ni sozlash"""
        key = os.getenv("API_KEY_ENCRYPTION_KEY")
        if not key:
            # Generate new key if not exists
            key = Fernet.generate_key().decode()
            os.environ["API_KEY_ENCRYPTION_KEY"] = key
            logging.warning("Generated new encryption key. Save it to .env file!")
        
        try:
            if len(key) == 44:  # Fernet key format
                self._cipher_suite = Fernet(key.encode())
            else:
                # Convert 32-char key to Fernet format
                key_bytes = key.encode()[:32].ljust(32, b'0')
                fernet_key = base64.urlsafe_b64encode(key_bytes)
                self._cipher_suite = Fernet(fernet_key)
        except Exception as e:
            logging.error(f"Encryption initialization failed: {e}")
            self.encryption_enabled = False
    
    def encrypt_key(self, api_key: str) -> str:
        """API key ni shifrlash"""
        if not self.encryption_enabled or not self._cipher_suite:
            return api_key
        
        try:
            encrypted = self._cipher_suite.encrypt(api_key.encode())
            return encrypted.decode()
        except Exception as e:
            logging.error(f"Encryption failed: {e}")
            return api_key
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """API key ni deshifrlash"""
        if not self.encryption_enabled or not self._cipher_suite:
            return encrypted_key
        
        try:
            decrypted = self._cipher_suite.decrypt(encrypted_key.encode())
            return decrypted.decode()
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            return encrypted_key

class ConfigManager:
    """Asosiy configuration manager with validation"""
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
        self._validation_errors: List[str] = []
        
        self.logger = self._setup_logger()
        self.security_manager = SecurityManager()
        
        # Required environment variables
        self.required_env_vars = {
            "TELEGRAM_BOT_TOKEN": "Telegram bot token",
            "ONEINCH_API_KEY": "1inch API key", 
            "ALCHEMY_API_KEY": "Alchemy API key"
        }
        
        # Optional but recommended
        self.recommended_env_vars = {
            "GEMINI_API_KEY_1": "Gemini API key",
            "HUGGINGFACE_API_KEY": "HuggingFace API key",
            "NEWSAPI_KEY": "News API key"
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Logger sozlash"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    async def validate_environment(self) -> bool:
        """Environment variables ni tekshirish"""
        self._validation_errors.clear()
        
        # Check required variables
        missing_required = []
        for var, description in self.required_env_vars.items():
            value = os.getenv(var)
            if not value or value.strip() == "":
                missing_required.append(f"{var} ({description})")
        
        if missing_required:
            self._validation_errors.append(
                f"Missing required environment variables: {', '.join(missing_required)}"
            )
        
        # Check optional variables
        missing_optional = []
        for var, description in self.recommended_env_vars.items():
            value = os.getenv(var)
            if not value or value.strip() == "":
                missing_optional.append(f"{var} ({description})")
        
        if missing_optional:
            self.logger.warning(
                f"Missing optional environment variables: {', '.join(missing_optional)}"
            )
        
        # Validate specific formats
        self._validate_api_key_formats()
        
        if self._validation_errors:
            for error in self._validation_errors:
                self.logger.error(f"Validation error: {error}")
            return False
            
        self.logger.info("✅ Environment validation passed")
        return True
    
    def _validate_api_key_formats(self):
        """API key formatlarini tekshirish"""
        
        # Telegram bot token validation
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if telegram_token and not re.match(r'^\d+:[A-Za-z0-9_-]+$', telegram_token):
            self._validation_errors.append("Invalid Telegram bot token format")
        
        # Binance API key validation
        binance_key = os.getenv("BINANCE_API_KEY")
        if binance_key and len(binance_key) != 64:
            self._validation_errors.append("Binance API key should be 64 characters")
        
        # Gemini API keys validation
        for i in range(1, 6):
            gemini_key = os.getenv(f"GEMINI_API_KEY_{i}")
            if gemini_key and not gemini_key.startswith("AIza"):
                self.logger.warning(f"Gemini API key {i} format might be incorrect")
    
    async def load_config(self) -> bool:
        """Barcha konfiguratsiyalarni yuklash"""
        try:
            async with self._lock:
                self.logger.info("🔄 Configuration loading started...")
                
                # Validate environment first
                if not await self.validate_environment():
                    raise ConfigValidationError("Environment validation failed")
                
                # Load environment variables
                self._load_env_vars()
                
                # Load JSON settings
                await self._load_settings_json()
                
                # Initialize configurations
                await self._init_api_configs()
                await self._init_trading_config()
                await self._init_telegram_config()
                
                # Final validation
                if not await self._validate_final_config():
                    raise ConfigValidationError("Final configuration validation failed")
                
                self.logger.info("✅ Configuration loaded successfully")
                return True
                
        except ConfigValidationError as e:
            self.logger.error(f"❌ Configuration validation error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Configuration loading error: {e}")
            return False
    
    def _load_env_vars(self) -> None:
        """Environment variables yuklash"""
        if self.env_file.exists():
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            # Basic validation
                            if not key.replace('_', '').isalnum():
                                self.logger.warning(f"Invalid env var name at line {line_num}: {key}")
                                continue
                                
                            os.environ[key] = value
                        except Exception as e:
                            self.logger.warning(f"Error parsing .env line {line_num}: {e}")
        else:
            self.logger.warning("❌ .env file not found")
    
    async def _load_settings_json(self) -> None:
        """JSON settings yuklash"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                self.logger.info("✅ Settings.json loaded")
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in settings.json: {e}")
                self._config = self._get_default_settings()
            except Exception as e:
                self.logger.error(f"Error loading settings.json: {e}")
                self._config = self._get_default_settings()
        else:
            self._config = self._get_default_settings()
            await self.save_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Default sozlamalar"""
        return {
            "trading": {
                "mode": "SIGNAL_ONLY",
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT"],
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
            "telegram": {
                "signal_format": "professional",
                "language": "uz",
                "timezone": "Asia/Tashkent",
                "admin_ids": []
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
            },
            "security": {
                "encrypt_api_keys": True,
                "rate_limit_buffer": 0.8,
                "max_retries": 3,
                "timeout_seconds": 30
            }
        }
    
    async def _init_api_configs(self) -> None:
        """API konfiguratsiyalarni sozlash with validation"""
        try:
            # 1inch API
            oneinch_key = os.getenv("ONEINCH_API_KEY")
            if oneinch_key:
                self._api_configs[APIProvider.ONEINCG] = [
                    APIConfig(
                        provider=APIProvider.ONEINCG,
                        api_key=self.security_manager.decrypt_key(oneinch_key),
                        endpoint="https://api.1inch.dev/fusion",
                        rate_limit=100,
                        fallback_providers=["alchemy", "thegraph"]
                    )
                ]
            
            # Alchemy API
            alchemy_key = os.getenv("ALCHEMY_API_KEY")
            if alchemy_key:
                self._api_configs[APIProvider.ALCHEMY] = [
                    APIConfig(
                        provider=APIProvider.ALCHEMY,
                        api_key=self.security_manager.decrypt_key(alchemy_key),
                        endpoint=f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}",
                        rate_limit=300
                    )
                ]
            
            # HuggingFace API
            hf_key = os.getenv("HUGGINGFACE_API_KEY")
            if hf_key:
                self._api_configs[APIProvider.HUGGINGFACE] = [
                    APIConfig(
                        provider=APIProvider.HUGGINGFACE,
                        api_key=self.security_manager.decrypt_key(hf_key),
                        endpoint="https://api-inference.huggingface.co/models",
                        rate_limit=100,
                        fallback_providers=["gemini", "claude"]
                    )
                ]
            
            # Gemini APIs (5 ta key)
            gemini_configs = []
            for i in range(1, 6):
                key = os.getenv(f"GEMINI_API_KEY_{i}")
                if key:
                    gemini_configs.append(APIConfig(
                        provider=APIProvider.GEMINI,
                        api_key=self.security_manager.decrypt_key(key),
                        endpoint="https://generativelanguage.googleapis.com/v1beta",
                        rate_limit=60,
                        priority=i
                    ))
            
            if gemini_configs:
                self._api_configs[APIProvider.GEMINI] = gemini_configs
            
            # Claude API
            claude_key = os.getenv("CLAUDE_API_KEY")
            if claude_key:
                self._api_configs[APIProvider.CLAUDE] = [
                    APIConfig(
                        provider=APIProvider.CLAUDE,
                        api_key=self.security_manager.decrypt_key(claude_key),
                        endpoint="https://api.anthropic.com/v1",
                        rate_limit=50
                    )
                ]
                
            self.logger.info(f"✅ Initialized {len(self._api_configs)} API providers")
            
        except Exception as e:
            self.logger.error(f"API config initialization error: {e}")
            raise ConfigValidationError(f"API configuration failed: {e}")
    
    async def _init_trading_config(self) -> None:
        """Trading konfiguratsiyasini sozlash"""
        try:
            cfg = self._config.get("trading", {})
            self._trading_config = TradingConfig(
                mode=TradingMode[cfg.get("mode", "SIGNAL_ONLY")],
                symbols=cfg.get("symbols", ["BTCUSDT", "ETHUSDT"]),
                risk_percentage=cfg.get("risk_percentage", 0.5),
                max_daily_loss=cfg.get("max_daily_loss", 3.0),
                max_positions=cfg.get("max_positions", 3),
                stop_loss_percentage=cfg.get("stop_loss_percentage", 2.0),
                take_profit_percentage=cfg.get("take_profit_percentage", 3.0),
                trailing_stop=cfg.get("trailing_stop", True),
                kill_zones=cfg.get("kill_zones", {})
            )
            self.logger.info("✅ Trading config initialized")
        except Exception as e:
            self.logger.error(f"Trading config error: {e}")
            raise ConfigValidationError(f"Trading configuration failed: {e}")
    
    async def _init_telegram_config(self) -> None:
        """Telegram konfiguratsiyasini sozlash"""
        try:
            cfg = self._config.get("telegram", {})
            self._telegram_config = TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                channel_id=os.getenv("TELEGRAM_CHANNEL_ID", ""),
                admin_ids=cfg.get("admin_ids", []),
                signal_format=cfg.get("signal_format", "professional"),
                language=cfg.get("language", "uz"),
                timezone=cfg.get("timezone", "Asia/Tashkent")
            )
            self.logger.info("✅ Telegram config initialized")
        except Exception as e:
            self.logger.error(f"Telegram config error: {e}")
            raise ConfigValidationError(f"Telegram configuration failed: {e}")
    
    async def _validate_final_config(self) -> bool:
        """Final configuration validation"""
        errors = []
        
        # Check if we have at least one API provider
        if not self._api_configs:
            errors.append("No API providers configured")
        
        # Check trading config
        if not self._trading_config:
            errors.append("Trading configuration missing")
        elif not self._trading_config.symbols:
            errors.append("No trading symbols configured")
        
        # Check telegram config
        if not self._telegram_config:
            errors.append("Telegram configuration missing")
        elif not self._telegram_config.bot_token:
            errors.append("Telegram bot token missing")
        
        if errors:
            for error in errors:
                self.logger.error(f"Final validation error: {error}")
            return False
        
        return True
    
    async def save_settings(self) -> bool:
        """Sozlamalarni saqlash"""
        try:
            async with self._lock:
                # Create backup
                if self.settings_file.exists():
                    backup_file = self.settings_file.with_suffix('.json.backup')
                    self.settings_file.rename(backup_file)
                
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                    
                self.logger.info("✅ Settings saved successfully")
                return True
        except Exception as e:
            self.logger.error(f"❌ Settings save error: {e}")
            return False
    
    def get_api_config(self, provider: APIProvider, index: int = 0) -> Optional[APIConfig]:
        """API konfiguratsiyasini olish"""
        configs = self._api_configs.get(provider, [])
        if index < len(configs):
            config = configs[index]
            # Decrypt API key when accessing
            config.api_key = self.security_manager.decrypt_key(config.api_key)
            return config
        return None
    
    def get_all_api_configs(self, provider: APIProvider) -> List[APIConfig]:
        """Barcha API konfiguratsiyalarini olish"""
        configs = self._api_configs.get(provider, [])
        # Decrypt API keys when accessing
        for config in configs:
            config.api_key = self.security_manager.decrypt_key(config.api_key)
        return configs
    
    @property
    def trading(self) -> TradingConfig:
        """Trading konfiguratsiyasi"""
        return self._trading_config or TradingConfig()
    
    @property
    def telegram(self) -> TelegramConfig:
        """Telegram konfiguratsiyasi"""
        return self._telegram_config or TelegramConfig()
    
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
        self.logger.info("🔄 Configuration reloading...")
        return await self.load_config()
    
    def get_validation_errors(self) -> List[str]:
        """Validation xatolarini olish"""
        return self._validation_errors.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Configuration health status"""
        return {
            "status": "healthy" if not self._validation_errors else "unhealthy",
            "api_providers": len(self._api_configs),
            "trading_mode": self._trading_config.mode.name if self._trading_config else "unknown",
            "telegram_configured": bool(self._telegram_config and self._telegram_config.bot_token),
            "validation_errors": len(self._validation_errors),
            "encryption_enabled": self.security_manager.encryption_enabled
        }

# Global instance
config_manager = ConfigManager()
