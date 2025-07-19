"""
API Tests
Trading APIs, AI clients, news/social API tests
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
from datetime import datetime, timedelta
import json

from src.api.trading_apis import TradingAPIManager, OneInchAPI, AlchemyAPI, CoinGeckoAPI
from src.api.ai_clients import AIClientManager, HuggingFaceClient, GeminiClient, ClaudeClient
from src.api.news_social import NewsSocialAPI, NewsAPIClient, RedditClient
from src.api.telegram_client import TelegramClient, MessageType

class TestTradingAPIs:
    """Trading API testlari"""
    
    @pytest.fixture
    def trading_api_manager(self):
        """TradingAPIManager instance"""
        return TradingAPIManager()
        
    @pytest.fixture
    def mock_response(self):
        """Mock HTTP response"""
        mock = AsyncMock()
        mock.status = 200
        mock.json = AsyncMock(return_value={"success": True})
        return mock
        
    @pytest.mark.asyncio
    async def test_oneinch_order_flow(self, trading_api_manager, mock_response):
        """1inch order flow testi"""
        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            api = OneInchAPI(api_key="test_key")
            
            # Get order flow
            result = await api.get_order_flow("USDT", limit=10)
            assert result is not None
            
            # Get liquidity sources
            liquidity = await api.get_liquidity_sources()
            assert liquidity is not None
            
    @pytest.mark.asyncio
    async def test_alchemy_whale_tracking(self, trading_api_manager):
        """Alchemy whale tracking testi"""
        mock_data = {
            "transfers": [
                {
                    "from": "0x123",
                    "to": "0x456",
                    "value": "1000000000000000000",  # 1 ETH
                    "asset": "ETH",
                    "blockNum": "0x123456"
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_data)
            
            api = AlchemyAPI(api_key="test_key")
            whales = await api.get_whale_transfers("ETH", min_value=100)
            
            assert len(whales) > 0
            assert whales[0]["asset"] == "ETH"
            
    @pytest.mark.asyncio
    async def test_coingecko_market_data(self, trading_api_manager):
        """CoinGecko market data testi"""
        mock_data = {
            "bitcoin": {
                "usd": 45000,
                "usd_24h_change": 2.5,
                "usd_market_cap": 880000000000
            }
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_data)
            
            api = CoinGeckoAPI()
            price_data = await api.get_price_data(["bitcoin"])
            
            assert "bitcoin" in price_data
            assert price_data["bitcoin"]["usd"] == 45000
            
    @pytest.mark.asyncio
    async def test_trading_api_fallback(self, trading_api_manager):
        """Trading API fallback testi"""
        # First API fails
        with patch.object(OneInchAPI, 'get_order_flow', side_effect=Exception("API Error")):
            # Second API succeeds
            with patch.object(AlchemyAPI, 'get_order_flow', return_value={"success": True}):
                result = await trading_api_manager.get_order_flow_with_fallback("USDT")
                assert result["success"] is True
                
    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, trading_api_manager):
        """API rate limiting testi"""
        api = OneInchAPI(api_key="test_key", rate_limit=2)  # 2 calls per minute
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value={})
            
            # Make multiple rapid calls
            start_time = asyncio.get_event_loop().time()
            for _ in range(3):
                await api.get_order_flow("USDT")
                
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Should have rate limited
            assert elapsed > 0.5  # At least some delay

class TestAIClients:
    """AI Client testlari"""
    
    @pytest.fixture
    def ai_client_manager(self):
        """AIClientManager instance"""
        return AIClientManager()
        
    @pytest.mark.asyncio
    async def test_huggingface_sentiment(self, ai_client_manager):
        """HuggingFace sentiment analysis testi"""
        mock_response = [
            [
                {"label": "POSITIVE", "score": 0.95},
                {"label": "NEGATIVE", "score": 0.05}
            ]
        ]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            
            client = HuggingFaceClient(api_key="test_key")
            sentiment = await client.analyze_sentiment("Bitcoin is going to the moon!")
            
            assert sentiment["label"] == "POSITIVE"
            assert sentiment["score"] > 0.9
            
    @pytest.mark.asyncio
    async def test_gemini_analysis(self, ai_client_manager):
        """Gemini AI analysis testi"""
        mock_response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "Bullish sentiment detected. Key factors: whale accumulation, positive news."
                    }]
                }
            }]
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            
            client = GeminiClient(api_key="test_key")
            analysis = await client.analyze_market_sentiment("BTC", ["Whale bought 1000 BTC"])
            
            assert "Bullish" in analysis
            assert "whale accumulation" in analysis
            
    @pytest.mark.asyncio
    async def test_claude_analysis(self, ai_client_manager):
        """Claude AI analysis testi"""
        mock_response = {
            "content": [{
                "text": "Market analysis: Strong buy signal based on technical indicators."
            }]
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            
            client = ClaudeClient(api_key="test_key")
            analysis = await client.analyze_trading_signal({
                "symbol": "BTCUSDT",
                "indicators": {"RSI": 30, "MACD": "bullish"}
            })
            
            assert "Strong buy signal" in analysis
            
    @pytest.mark.asyncio
    async def test_ai_fallback_chain(self, ai_client_manager):
        """AI fallback chain testi"""
        # HuggingFace fails
        with patch.object(HuggingFaceClient, 'analyze_sentiment', side_effect=Exception("API Error")):
            # Gemini succeeds
            with patch.object(GeminiClient, 'analyze_market_sentiment', return_value="Bullish"):
                result = await ai_client_manager.get_sentiment_with_fallback("Test text")
                assert result == "Bullish"

class TestNewsSocialAPIs:
    """News va Social API testlari"""
    
    @pytest.fixture
    def news_social_api(self):
        """NewsSocialAPI instance"""
        return NewsSocialAPI()
        
    @pytest.mark.asyncio
    async def test_newsapi_crypto_news(self, news_social_api):
        """NewsAPI crypto news testi"""
        mock_response = {
            "articles": [
                {
                    "title": "Bitcoin Hits New High",
                    "description": "Bitcoin reaches $50,000",
                    "publishedAt": "2024-01-01T12:00:00Z",
                    "source": {"name": "CoinDesk"}
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            
            client = NewsAPIClient(api_key="test_key")
            news = await client.get_crypto_news(query="bitcoin", limit=10)
            
            assert len(news) > 0
            assert "Bitcoin" in news[0]["title"]
            
    @pytest.mark.asyncio
    async def test_reddit_sentiment(self, news_social_api):
        """Reddit sentiment analysis testi"""
        mock_response = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "BTC to the moon!",
                            "score": 1000,
                            "num_comments": 500,
                            "created_utc": 1640000000
                        }
                    }
                ]
            }
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            
            client = RedditClient()
            posts = await client.get_crypto_posts(subreddit="cryptocurrency", limit=10)
            
            assert len(posts) > 0
            assert posts[0]["score"] == 1000
            
    @pytest.mark.asyncio
    async def test_news_filtering(self, news_social_api):
        """News filtering testi"""
        mock_articles = [
            {"title": "Bitcoin price analysis", "sentiment": 0.8},
            {"title": "Crypto scam alert", "sentiment": -0.9},
            {"title": "Ethereum update", "sentiment": 0.5}
        ]
        
        # Filter positive only
        positive = news_social_api.filter_news(mock_articles, min_sentiment=0.5)
        assert len(positive) == 2
        
        # Filter by keyword
        bitcoin_news = news_social_api.filter_news(mock_articles, keywords=["bitcoin"])
        assert len(bitcoin_news) == 1

class TestTelegramClient:
    """Telegram client testlari"""
    
    @pytest.fixture
    def telegram_client(self):
        """TelegramClient instance"""
        return TelegramClient(bot_token="test_token")
        
    @pytest.mark.asyncio
    async def test_send_signal(self, telegram_client):
        """Signal yuborish testi"""
        signal_data = {
            "symbol": "BTCUSDT",
            "action": "BUY",
            "price": 45000,
            "stop_loss": 44000,
            "take_profit": 46000,
            "confidence": 85,
            "risk": 0.5
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"ok": True, "result": {"message_id": 123}}
            )
            
            result = await telegram_client.send_signal("-1001234567890", signal_data)
            assert result is True
            
    @pytest.mark.asyncio
    async def test_send_with_keyboard(self, telegram_client):
        """Keyboard bilan xabar yuborish testi"""
        keyboard = [
            [{"text": "🟢 AVTO SAVDO", "callback_data": "auto_trade"}],
            [{"text": "🔴 BEKOR QILISH", "callback_data": "cancel"}]
        ]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"ok": True}
            )
            
            result = await telegram_client.send_message(
                "-1001234567890",
                "Test message",
                keyboard=keyboard
            )
            assert result is True
            
    @pytest.mark.asyncio
    async def test_edit_message(self, telegram_client):
        """Xabarni tahrirlash testi"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"ok": True}
            )
            
            result = await telegram_client.edit_message(
                "-1001234567890",
                123,
                "Updated message"
            )
            assert result is True
            
    @pytest.mark.asyncio
    async def test_error_handling(self, telegram_client):
        """Xato handling testi"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"ok": False, "error_code": 429, "description": "Too Many Requests"}
            )
            
            with pytest.raises(Exception) as exc_info:
                await telegram_client.send_message("-1001234567890", "Test")
                
            assert "429" in str(exc_info.value)
