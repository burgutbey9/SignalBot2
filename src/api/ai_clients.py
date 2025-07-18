"""
AI API Clients for Crypto Sentiment Analysis
HuggingFace, Gemini, Claude integration with fallback support
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from datetime import datetime
import re
import hashlib

from config.config import config_manager, APIProvider
from utils.logger import get_logger
from utils.helpers import RateLimiter, retry_manager, TimeUtils, ExponentialBackoff

logger = get_logger(__name__)

class SentimentScore(Enum):
    """Sentiment score turlari"""
    VERY_BULLISH = 5
    BULLISH = 4
    NEUTRAL = 3
    BEARISH = 2
    VERY_BEARISH = 1

@dataclass
class SentimentResult:
    """Sentiment tahlil natijasi"""
    score: SentimentScore
    confidence: float
    summary: str
    keywords: List[str] = field(default_factory=list)
    market_impact: str = "NEUTRAL"
    reasoning: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class NewsAnalysis:
    """Yangilik tahlili"""
    title: str
    source: str
    sentiment: SentimentResult
    importance: str  # HIGH, MEDIUM, LOW
    affected_pairs: List[str] = field(default_factory=list)
    event_type: str = ""  # REGULATORY, TECHNICAL, MARKET, PARTNERSHIP

class AIClientBase:
    """Base AI client class"""
    def __init__(self, provider: APIProvider, rate_limit: int = 60):
        self.provider = provider
        self.rate_limiter = RateLimiter(calls=rate_limit, period=60)
        self.session: Optional[aiohttp.ClientSession] = None
        self.backoff = ExponentialBackoff()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[SentimentResult]:
        """Sentiment tahlil qilish (override in subclass)"""
        raise NotImplementedError
        
    def _parse_sentiment_score(self, score: Union[float, str]) -> SentimentScore:
        """Sentiment skorini parse qilish"""
        if isinstance(score, str):
            score_map = {
                "very bullish": SentimentScore.VERY_BULLISH,
                "bullish": SentimentScore.BULLISH,
                "neutral": SentimentScore.NEUTRAL,
                "bearish": SentimentScore.BEARISH,
                "very bearish": SentimentScore.VERY_BEARISH
            }
            return score_map.get(score.lower(), SentimentScore.NEUTRAL)
        else:
            if score >= 0.8: return SentimentScore.VERY_BULLISH
            elif score >= 0.6: return SentimentScore.BULLISH
            elif score >= 0.4: return SentimentScore.NEUTRAL
            elif score >= 0.2: return SentimentScore.BEARISH
            else: return SentimentScore.VERY_BEARISH

class HuggingFaceClient(AIClientBase):
    """HuggingFace API client"""
    def __init__(self):
        super().__init__(APIProvider.HUGGINGFACE, rate_limit=100)
        self.models = {
            "sentiment": "ProsusAI/finbert",
            "summary": "facebook/bart-large-cnn",
            "crypto": "ElKulako/cryptobert"
        }
        
    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[SentimentResult]:
        """HuggingFace sentiment analysis"""
        config = config_manager.get_api_config(self.provider)
        if not config or not config.api_key:
            logger.error("HuggingFace API key topilmadi")
            return None
            
        try:
            await self.rate_limiter.acquire()
            
            # Crypto-specific model tanlash
            model = self.models["crypto"] if "crypto" in text.lower() or "bitcoin" in text.lower() else self.models["sentiment"]
            
            headers = {"Authorization": f"Bearer {config.api_key}"}
            url = f"{config.endpoint}/{model}"
            
            async with self.session.post(
                url,
                headers=headers,
                json={"inputs": text[:512]},  # Max 512 tokens
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_huggingface_response(data, text)
                else:
                    logger.error(f"HuggingFace API xatosi: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"HuggingFace sentiment tahlilda xato: {e}")
            return None
            
    def _parse_huggingface_response(self, data: Union[List, Dict], text: str) -> SentimentResult:
        """HuggingFace javobini parse qilish"""
        if isinstance(data, list) and data:
            # FinBERT response format
            scores = {item['label']: item['score'] for item in data[0] if isinstance(item, dict)}
            
            # Eng yuqori skorni topish
            best_label = max(scores, key=scores.get)
            confidence = scores[best_label]
            
            # Sentiment mapping
            sentiment_map = {
                "positive": SentimentScore.BULLISH,
                "negative": SentimentScore.BEARISH,
                "neutral": SentimentScore.NEUTRAL
            }
            
            score = sentiment_map.get(best_label.lower(), SentimentScore.NEUTRAL)
            
            # Keywords extraction
            keywords = self._extract_keywords(text)
            
            return SentimentResult(
                score=score,
                confidence=confidence,
                summary=f"{best_label.capitalize()} sentiment detected",
                keywords=keywords,
                market_impact=self._determine_market_impact(score, confidence),
                reasoning=f"HuggingFace {best_label} with {confidence:.2%} confidence",
                provider="HuggingFace"
            )
        
        return SentimentResult(
            score=SentimentScore.NEUTRAL,
            confidence=0.5,
            summary="Unable to determine sentiment",
            provider="HuggingFace"
        )
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Muhim keywordlarni ajratish"""
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "whale", "bullish", "bearish"]
        found_keywords = []
        
        for keyword in crypto_keywords:
            if keyword in text.lower():
                found_keywords.append(keyword.upper())
                
        return found_keywords[:5]  # Top 5
        
    def _determine_market_impact(self, score: SentimentScore, confidence: float) -> str:
        """Bozorga ta'sirini aniqlash"""
        if confidence < 0.6: return "LOW"
        
        if score in [SentimentScore.VERY_BULLISH, SentimentScore.VERY_BEARISH]:
            return "HIGH"
        elif score in [SentimentScore.BULLISH, SentimentScore.BEARISH]:
            return "MEDIUM"
        else:
            return "LOW"

class GeminiClient(AIClientBase):
    """Google Gemini API client"""
    def __init__(self, api_key_index: int = 0):
        super().__init__(APIProvider.GEMINI, rate_limit=60)
        self.api_key_index = api_key_index
        self.model = "gemini-pro"
        
    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[SentimentResult]:
        """Gemini sentiment analysis"""
        config = config_manager.get_api_config(self.provider, self.api_key_index)
        if not config or not config.api_key:
            logger.error(f"Gemini API key {self.api_key_index + 1} topilmadi")
            return None
            
        try:
            await self.rate_limiter.acquire()
            
            prompt = self._create_sentiment_prompt(text, context)
            
            url = f"{config.endpoint}/models/{self.model}:generateContent?key={config.api_key}"
            
            async with self.session.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 500
                    }
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_gemini_response(data)
                else:
                    logger.error(f"Gemini API xatosi: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Gemini sentiment tahlilda xato: {e}")
            return None
            
    def _create_sentiment_prompt(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Sentiment tahlil uchun prompt yaratish"""
        prompt = f"""Analyze the crypto market sentiment of this text and provide a JSON response:

Text: {text}

Context: {json.dumps(context) if context else 'General crypto news'}

Respond ONLY with valid JSON in this format:
{{
    "sentiment": "very bullish|bullish|neutral|bearish|very bearish",
    "confidence": 0.0-1.0,
    "summary": "brief summary",
    "keywords": ["keyword1", "keyword2"],
    "market_impact": "HIGH|MEDIUM|LOW",
    "reasoning": "explanation",
    "affected_pairs": ["BTCUSDT", "ETHUSDT"]
}}"""
        return prompt
        
    def _parse_gemini_response(self, data: Dict[str, Any]) -> Optional[SentimentResult]:
        """Gemini javobini parse qilish"""
        try:
            # Extract JSON from response
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # JSON topish
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.error("Gemini javobida JSON topilmadi")
                return None
                
            result = json.loads(json_match.group())
            
            return SentimentResult(
                score=self._parse_sentiment_score(result.get("sentiment", "neutral")),
                confidence=float(result.get("confidence", 0.5)),
                summary=result.get("summary", ""),
                keywords=result.get("keywords", []),
                market_impact=result.get("market_impact", "MEDIUM"),
                reasoning=result.get("reasoning", ""),
                provider=f"Gemini-{self.api_key_index + 1}"
            )
            
        except Exception as e:
            logger.error(f"Gemini javobini parse qilishda xato: {e}")
            return None

class ClaudeClient(AIClientBase):
    """Anthropic Claude API client"""
    def __init__(self):
        super().__init__(APIProvider.CLAUDE, rate_limit=50)
        self.model = "claude-3-sonnet-20240229"
        
    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[SentimentResult]:
        """Claude sentiment analysis"""
        config = config_manager.get_api_config(self.provider)
        if not config or not config.api_key:
            logger.error("Claude API key topilmadi")
            return None
            
        try:
            await self.rate_limiter.acquire()
            
            headers = {
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            prompt = self._create_claude_prompt(text, context)
            
            async with self.session.post(
                f"{config.endpoint}/messages",
                headers=headers,
                json={
                    "model": self.model,
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_claude_response(data)
                else:
                    logger.error(f"Claude API xatosi: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Claude sentiment tahlilda xato: {e}")
            return None
            
    def _create_claude_prompt(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Claude uchun prompt yaratish"""
        return f"""Analyze this crypto market text for sentiment. Provide a structured analysis.

Text: {text}

Context: {json.dumps(context) if context else 'General crypto market'}

Provide:
1. Sentiment (very bullish/bullish/neutral/bearish/very bearish)
2. Confidence (0-100%)
3. Brief summary
4. Key indicators
5. Market impact assessment
6. Trading implications

Format as JSON."""
        
    def _parse_claude_response(self, data: Dict[str, Any]) -> Optional[SentimentResult]:
        """Claude javobini parse qilish"""
        try:
            content = data.get("content", [{}])[0].get("text", "")
            
            # Basic parsing for Claude response
            sentiment = "neutral"
            confidence = 0.5
            
            # Sentiment keywords check
            if "very bullish" in content.lower(): sentiment = "very bullish"
            elif "bullish" in content.lower(): sentiment = "bullish"
            elif "bearish" in content.lower(): sentiment = "bearish"
            elif "very bearish" in content.lower(): sentiment = "very bearish"
            
            # Confidence extraction
            conf_match = re.search(r'(\d+)%', content)
            if conf_match:
                confidence = float(conf_match.group(1)) / 100
                
            return SentimentResult(
                score=self._parse_sentiment_score(sentiment),
                confidence=confidence,
                summary=content[:200],
                keywords=self._extract_keywords_from_text(content),
                market_impact="MEDIUM",
                reasoning="Claude analysis",
                provider="Claude"
            )
            
        except Exception as e:
            logger.error(f"Claude javobini parse qilishda xato: {e}")
            return None
            
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Matndan keywordlarni ajratish"""
        important_words = ["bitcoin", "ethereum", "bullish", "bearish", "pump", "dump", "whale", "support", "resistance"]
        found = [w.upper() for w in important_words if w in text.lower()]
        return found[:5]

class AIOrchestrator:
    """AI clientlarni boshqarish va fallback"""
    def __init__(self):
        self.clients: Dict[str, List[AIClientBase]] = {
            "primary": [],
            "fallback": []
        }
        self._initialized = False
        
    async def initialize(self):
        """Barcha AI clientlarni sozlash"""
        if self._initialized:
            return
            
        # HuggingFace (primary)
        self.clients["primary"].append(HuggingFaceClient())
        
        # Gemini clients (5 ta)
        gemini_configs = config_manager.get_all_api_configs(APIProvider.GEMINI)
        for i in range(len(gemini_configs)):
            client = GeminiClient(api_key_index=i)
            if i == 0:
                self.clients["primary"].append(client)
            else:
                self.clients["fallback"].append(client)
                
        # Claude (fallback)
        self.clients["fallback"].append(ClaudeClient())
        
        self._initialized = True
        logger.info(f"✅ AI Orchestrator initialized: {len(self.clients['primary'])} primary, {len(self.clients['fallback'])} fallback")
        
    async def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[SentimentResult]:
        """Sentiment tahlil with fallback"""
        if not self._initialized:
            await self.initialize()
            
        # Try primary clients first
        for client in self.clients["primary"]:
            try:
                async with client:
                    result = await client.analyze_sentiment(text, context)
                    if result:
                        logger.info(f"✅ Sentiment analysis successful with {client.provider.value}")
                        return result
            except Exception as e:
                logger.warning(f"Primary client {client.provider.value} failed: {e}")
                continue
                
        # Fallback to secondary clients
        for client in self.clients["fallback"]:
            try:
                async with client:
                    result = await client.analyze_sentiment(text, context)
                    if result:
                        logger.info(f"✅ Sentiment analysis successful with fallback {client.provider.value}")
                        return result
            except Exception as e:
                logger.warning(f"Fallback client {client.provider.value} failed: {e}")
                continue
                
        logger.error("❌ Barcha AI clientlar muvaffaqiyatsiz")
        return None
        
    async def analyze_news_batch(self, news_items: List[Dict[str, Any]]) -> List[NewsAnalysis]:
        """Yangiliklar to'plamini tahlil qilish"""
        results = []
        
        for item in news_items:
            sentiment = await self.analyze_sentiment(
                f"{item.get('title', '')} {item.get('description', '')}",
                {"source": item.get('source', ''), "url": item.get('url', '')}
            )
            
            if sentiment:
                analysis = NewsAnalysis(
                    title=item.get('title', ''),
                    source=item.get('source', ''),
                    sentiment=sentiment,
                    importance=self._determine_importance(sentiment),
                    affected_pairs=self._extract_affected_pairs(item),
                    event_type=self._classify_event_type(item)
                )
                results.append(analysis)
                
        return results
        
    def _determine_importance(self, sentiment: SentimentResult) -> str:
        """Yangilik muhimligini aniqlash"""
        if sentiment.market_impact == "HIGH" and sentiment.confidence > 0.8:
            return "HIGH"
        elif sentiment.market_impact == "MEDIUM" or sentiment.confidence > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
            
    def _extract_affected_pairs(self, news_item: Dict[str, Any]) -> List[str]:
        """Ta'sirlangan trading pairlarni aniqlash"""
        text = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
        
        pairs = []
        pair_map = {
            "bitcoin": "BTCUSDT",
            "btc": "BTCUSDT",
            "ethereum": "ETHUSDT",
            "eth": "ETHUSDT",
            "bnb": "BNBUSDT",
            "solana": "SOLUSDT",
            "sol": "SOLUSDT"
        }
        
        for keyword, pair in pair_map.items():
            if keyword in text and pair not in pairs:
                pairs.append(pair)
                
        return pairs[:3]  # Max 3 pairs
        
    def _classify_event_type(self, news_item: Dict[str, Any]) -> str:
        """Yangilik turini aniqlash"""
        text = f"{news_item.get('title', '')} {news_item.get('description', '')}".lower()
        
        if any(word in text for word in ["regulation", "sec", "government", "law"]):
            return "REGULATORY"
        elif any(word in text for word in ["upgrade", "fork", "development", "release"]):
            return "TECHNICAL"
        elif any(word in text for word in ["partnership", "integration", "collaboration"]):
            return "PARTNERSHIP"
        else:
            return "MARKET"

# Global instance
ai_orchestrator = AIOrchestrator()
