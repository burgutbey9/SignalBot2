"""
Configuration Tests
Config loader, settings validation, fallback system tests
"""
import pytest
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from config.config import ConfigManager, TradingMode, Settings
from config.fallback_config import FallbackConfig, ProviderStatus

class TestConfigManager:
    """ConfigManager testlari"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Vaqtinchalik config fayli"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "telegram": {
                    "bot_token": "test_token",
                    "authorized_users": [123456789],
                    "notification_channel": -1001234567890
                },
                "trading": {
                    "mode": "SIGNAL_ONLY",
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "max_positions": 3
                },
                "risk_management": {
                    "base_risk_percent": 0.5,
                    "max_risk_percent": 1.0,
                    "max_daily_loss": 3.0,
                    "stop_loss_percent": 1.5
                },
                "api_keys": {
                    "binance": {
                        "api_key": "test_key",
                        "api_secret": "test_secret"
                    }
                }
            }
            json.dump(config_data, f)
            
        yield f.name
        os.unlink(f.name)
        
    @pytest.fixture
    def config_manager(self, temp_config_file):
        """ConfigManager instance"""
        manager = ConfigManager()
        manager.config_path = temp_config_file
        return manager
        
    @pytest.mark.asyncio
    async def test_load_config(self, config_manager):
        """Config yuklash testi"""
        result = await config_manager.load_config()
        
        assert result is True
        assert config_manager.telegram.bot_token == "test_token"
        assert config_manager.trading.mode == TradingMode.SIGNAL_ONLY
        assert len(config_manager.trading.symbols) == 2
        assert config_manager.risk_management.base_risk_percent == 0.5
        
    @pytest.mark.asyncio
    async def test_save_config(self, config_manager):
        """Config saqlash testi"""
        await config_manager.load_config()
        
        # O'zgartirish
        config_manager.trading.symbols.append("BNBUSDT")
        config_manager.risk_management.base_risk_percent = 0.75
        
        # Saqlash
        result = await config_manager.save_config()
        assert result is True
        
        # Qayta yuklash va tekshirish
        new_manager = ConfigManager()
        new_manager.config_path = config_manager.config_path
        await new_manager.load_config()
        
        assert len(new_manager.trading.symbols) == 3
        assert new_manager.risk_management.base_risk_percent == 0.75
        
    def test_validate_config(self, config_manager):
        """Config validation testi"""
        # Valid config
        config = {
            "telegram": {"bot_token": "token", "authorized_users": [123]},
            "trading": {"mode": "SIGNAL_ONLY", "symbols": ["BTCUSDT"]},
            "risk_management": {"base_risk_percent": 0.5},
            "api_keys": {}
        }
        
        assert config_manager.validate_config(config) is True
        
        # Invalid config - missing required field
        invalid_config = {
            "telegram": {"bot_token": "token"},  # missing authorized_users
            "trading": {"mode": "SIGNAL_ONLY", "symbols": ["BTCUSDT"]}
        }
        
        assert config_manager.validate_config(invalid_config) is False
        
    def test_get_api_key(self, config_manager):
        """API key olish testi"""
        config_manager.api_keys.binance = {"api_key": "test_key", "api_secret": "test_secret"}
        config_manager.api_keys.gemini = ["key1", "key2", "key3"]
        
        # Binance key
        binance_key = config_manager.get_api_key("binance")
        assert binance_key == {"api_key": "test_key", "api_secret": "test_secret"}
        
        # Gemini keys
        gemini_key = config_manager.get_api_key("gemini", index=1)
        assert gemini_key == "key2"
        
        # Non-existent key
        assert config_manager.get_api_key("unknown") is None
        
    def test_trading_mode_change(self, config_manager):
        """Trading mode o'zgartirish testi"""
        config_manager.trading.mode = TradingMode.SIGNAL_ONLY
        assert config_manager.is_trading_enabled() is False
        
        config_manager.trading.mode = TradingMode.PAPER
        assert config_manager.is_trading_enabled() is True
        
        config_manager.trading.mode = TradingMode.LIVE
        assert config_manager.is_trading_enabled() is True
        
    @pytest.mark.asyncio
    async def test_load_corrupted_config(self, config_manager):
        """Buzilgan config yuklash testi"""
        # Buzilgan JSON
        with open(config_manager.config_path, 'w') as f:
            f.write("{invalid json}")
            
        result = await config_manager.load_config()
        assert result is False
        
    @pytest.mark.asyncio
    async def test_config_backup(self, config_manager):
        """Config backup testi"""
        await config_manager.load_config()
        
        # Backup yaratish
        backup_created = await config_manager.create_backup()
        assert backup_created is True
        
        # Backup fayli mavjudligini tekshirish
        backup_dir = Path(config_manager.config_path).parent / "backups"
        assert backup_dir.exists()
        assert len(list(backup_dir.glob("*.json"))) > 0

class TestFallbackConfig:
    """FallbackConfig testlari"""
    
    @pytest.fixture
    def fallback_config(self):
        """FallbackConfig instance"""
        return FallbackConfig()
        
    def test_register_provider(self, fallback_config):
        """Provider ro'yxatdan o'tkazish testi"""
        fallback_config.register_provider(
            "sentiment",
            "huggingface",
            {"api_key": "test_key"},
            priority=1
        )
        
        fallback_config.register_provider(
            "sentiment",
            "gemini",
            {"api_key": "test_key2"},
            priority=2
        )
        
        providers = fallback_config.get_providers("sentiment")
        assert len(providers) == 2
        assert providers[0]["name"] == "huggingface"  # Lower priority first
        
    def test_get_next_provider(self, fallback_config):
        """Keyingi provider olish testi"""
        # Register providers
        fallback_config.register_provider("api", "provider1", {}, 1)
        fallback_config.register_provider("api", "provider2", {}, 2)
        fallback_config.register_provider("api", "provider3", {}, 3)
        
        # Get first provider
        provider = fallback_config.get_next_provider("api")
        assert provider["name"] == "provider1"
        
        # Mark as failed and get next
        fallback_config.mark_provider_failed("api", "provider1")
        provider = fallback_config.get_next_provider("api")
        assert provider["name"] == "provider2"
        
    def test_provider_health_tracking(self, fallback_config):
        """Provider salomatlik tracking testi"""
        fallback_config.register_provider("api", "test_provider", {}, 1)
        
        # Success tracking
        for _ in range(5):
            fallback_config.record_success("api", "test_provider", 100)
            
        health = fallback_config.get_provider_health("api", "test_provider")
        assert health["status"] == ProviderStatus.HEALTHY
        assert health["success_rate"] == 100.0
        
        # Failure tracking
        for _ in range(10):
            fallback_config.record_failure("api", "test_provider", "timeout")
            
        health = fallback_config.get_provider_health("api", "test_provider")
        assert health["status"] == ProviderStatus.UNHEALTHY
        assert health["success_rate"] < 50
        
    def test_circuit_breaker(self, fallback_config):
        """Circuit breaker testi"""
        fallback_config.register_provider("api", "test_provider", {}, 1)
        
        # Multiple failures
        for _ in range(5):
            fallback_config.record_failure("api", "test_provider", "error")
            
        # Provider should be disabled
        provider = fallback_config.get_next_provider("api", skip_unhealthy=True)
        assert provider is None or provider["name"] != "test_provider"
        
    def test_fallback_chain(self, fallback_config):
        """Fallback zanjiri testi"""
        # Create fallback chain
        chain = fallback_config.create_fallback_chain(
            "sentiment",
            ["huggingface", "gemini", "claude"],
            max_retries=2
        )
        
        assert chain["service"] == "sentiment"
        assert len(chain["providers"]) == 3
        assert chain["max_retries"] == 2
        
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, fallback_config):
        """Fallback bilan bajarish testi"""
        # Mock function that fails for first two providers
        call_count = 0
        
        async def mock_api_call(provider_name: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Provider {provider_name} failed")
            return f"Success with {provider_name}"
            
        # Register providers
        for i in range(3):
            fallback_config.register_provider("test", f"provider{i}", {}, i)
            
        # Execute with fallback
        result = await fallback_config.execute_with_fallback(
            "test",
            mock_api_call
        )
        
        assert result == "Success with provider2"
        assert call_count == 3
        
    def test_get_statistics(self, fallback_config):
        """Statistika olish testi"""
        # Register and use providers
        fallback_config.register_provider("api", "provider1", {}, 1)
        fallback_config.register_provider("api", "provider2", {}, 2)
        
        # Record some metrics
        fallback_config.record_success("api", "provider1", 100)
        fallback_config.record_success("api", "provider1", 150)
        fallback_config.record_failure("api", "provider2", "timeout")
        
        stats = fallback_config.get_statistics()
        
        assert "api" in stats
        assert len(stats["api"]["providers"]) == 2
        assert stats["api"]["total_requests"] == 3
        assert stats["api"]["total_failures"] == 1
        
    def test_reset_provider_health(self, fallback_config):
        """Provider health reset testi"""
        fallback_config.register_provider("api", "test_provider", {}, 1)
        
        # Mark as failed
        for _ in range(5):
            fallback_config.record_failure("api", "test_provider", "error")
            
        # Reset health
        fallback_config.reset_provider_health("api", "test_provider")
        
        health = fallback_config.get_provider_health("api", "test_provider")
        assert health["status"] == ProviderStatus.HEALTHY
        assert health["failure_count"] == 0
