"""
AI Sentiment Analysis for Crypto
HuggingFace, Gemini, Claude API orqali crypto news va social media tahlili
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import numpy as np
import re
from collections import Counter

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils, RateLimiter, PerformanceMonitor
from api.ai_clients import ai_orchestrator
from api.news_social import news_aggregator

logger = get_logger(__name__)

class SentimentScore(Enum):
    """Sentiment darajalari"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

class NewsImpact(Enum):
    """Yangilik ta'siri"""
    CRITICAL = auto()  # Juda muhim
    HIGH = auto()      # Yuqori
    MEDIUM = auto()    # O'rtacha
    LOW = auto()       # Past

class SocialMood(Enum):
    """Ijtimoiy kayfiyat"""
    EXTREME_FEAR = auto()
    FEAR = auto()
    NEUTRAL = auto()
    GREED = auto()
    EXTREME_GREED = auto()

@dataclass
class NewsAnalysis:
    """Yangilik tahlili"""
    title: str
    source: str
    sentiment: SentimentScore
    impact: NewsImpact
    keywords: List[str]
    summary: str
    url: str
    published_at: datetime
    relevance_score: float

@dataclass
class SocialAnalysis:
    """Ijtimoiy media tahlili"""
    platform: str  # Reddit, Twitter, etc
    sentiment_score: float  # -1 to 1
    volume: int  # Post/comment count
    trending_topics: List[str]
    influencer_sentiment: Dict[str, float]
    mood: SocialMood

@dataclass
class MarketSentiment:
    """Umumiy bozor hissiyoti"""
    overall_score: float  # -100 to 100
    news_sentiment: float
    social_sentiment: float
    fear_greed_index: int  # 0-100
    trending_narratives: List[str]
    risk_events: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class SentimentSignal:
    """Sentiment signali"""
    symbol: str
    sentiment: SentimentScore
    confidence: float
    news_impact: List[NewsAnalysis]
    social_trends: Dict[str, Any]
    market_mood: MarketSentiment
    ai_analysis: Dict[str, Any]
    reasoning: List[str]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

class SentimentAnalyzer:
    """Sentiment tahlilchisi"""
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_minute=20)
        self.performance_monitor = PerformanceMonitor()
        
        # Cache
        self._sentiment_cache: Dict[str, SentimentSignal] = {}
        self._news_cache: Dict[str, List[NewsAnalysis]] = {}
        self._social_cache: Dict[str, SocialAnalysis] = {}
        self._market_sentiment: Optional[MarketSentiment] = None
        
        # Keywords for crypto sentiment
        self.bullish_keywords = [
            "moon", "bullish", "pump", "breakout", "rally", "buy",
            "adoption", "institutional", "upgrade", "partnership",
            "halving", "golden cross", "accumulation", "support"
        ]
        
        self.bearish_keywords = [
            "crash", "bearish", "dump", "breakdown", "sell", "fear",
            "regulation", "ban", "hack", "scam", "death cross",
            "resistance", "rejection", "liquidation", "fud"
        ]
        
        # Impact keywords
        self.high_impact_keywords = [
            "sec", "etf", "regulation", "ban", "hack", "bankruptcy",
            "federal reserve", "inflation", "interest rate", "war"
        ]
        
    async def start(self):
        """Analyzer ishga tushirish"""
        logger.info("🧠 Sentiment Analyzer ishga tushmoqda...")
        asyncio.create_task(self._update_market_sentiment())
        asyncio.create_task(self._monitor_breaking_news())
        
    async def analyze(self, symbol: str) -> Optional[SentimentSignal]:
        """To'liq sentiment tahlil"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Get data
            news_data = await self._analyze_news(symbol)
            social_data = await self._analyze_social_media(symbol)
            market_sentiment = await self._get_market_sentiment()
            ai_analysis = await self._get_ai_analysis(symbol, news_data, social_data)
            
            # Generate signal
            signal = await self._generate_signal(
                symbol, news_data, social_data, market_sentiment, ai_analysis
            )
            
            # Cache result
            self._sentiment_cache[symbol] = signal
            
            # Monitor performance
            self.performance_monitor.record("sentiment_analysis_complete", 1)
            
            return signal
            
        except Exception as e:
            logger.error(f"Sentiment analyze xatosi {symbol}: {e}")
            return None
            
    async def _analyze_news(self, symbol: str) -> List[NewsAnalysis]:
        """Yangiliklar tahlili"""
        try:
            # Get news from aggregator
            news_items = await news_aggregator.get_crypto_news(symbol)
            
            if not news_items:
                return []
                
            analyzed_news = []
            
            for item in news_items[:20]:  # Analyze top 20 news
                # Basic sentiment from keywords
                sentiment = self._analyze_text_sentiment(
                    item['title'] + " " + item.get('description', '')
                )
                
                # Determine impact
                impact = self._determine_news_impact(item)
                
                # Extract keywords
                keywords = self._extract_keywords(item['title'])
                
                # Get AI summary if high impact
                summary = item.get('description', '')[:200]
                if impact in [NewsImpact.CRITICAL, NewsImpact.HIGH]:
                    ai_summary = await ai_orchestrator.summarize_text(
                        item['title'] + " " + item.get('description', '')
                    )
                    if ai_summary:
                        summary = ai_summary
                        
                # Calculate relevance
                relevance = self._calculate_relevance(symbol, item)
                
                news_analysis = NewsAnalysis(
                    title=item['title'],
                    source=item['source'],
                    sentiment=sentiment,
                    impact=impact,
                    keywords=keywords,
                    summary=summary,
                    url=item.get('url', ''),
                    published_at=item['published_at'],
                    relevance_score=relevance
                )
                
                analyzed_news.append(news_analysis)
                
            # Sort by impact and relevance
            analyzed_news.sort(key=lambda x: (x.impact.value, x.relevance_score), reverse=True)
            
            # Cache results
            self._news_cache[symbol] = analyzed_news
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"News analysis xatosi: {e}")
            return []
            
    async def _analyze_social_media(self, symbol: str) -> SocialAnalysis:
        """Ijtimoiy media tahlili"""
        try:
            # Get social data
            reddit_data = await news_aggregator.get_reddit_sentiment(symbol)
            twitter_data = await news_aggregator.get_twitter_sentiment(symbol)
            
            # Aggregate sentiment scores
            total_score = 0
            total_volume = 0
            all_topics = []
            
            # Reddit analysis
            if reddit_data:
                reddit_score = reddit_data.get('sentiment_score', 0)
                reddit_volume = reddit_data.get('post_count', 0)
                total_score += reddit_score * reddit_volume
                total_volume += reddit_volume
                all_topics.extend(reddit_data.get('trending_topics', []))
                
            # Twitter analysis  
            if twitter_data:
                twitter_score = twitter_data.get('sentiment_score', 0)
                twitter_volume = twitter_data.get('tweet_count', 0)
                total_score += twitter_score * twitter_volume
                total_volume += twitter_volume
                all_topics.extend(twitter_data.get('trending_topics', []))
                
            # Calculate weighted sentiment
            avg_sentiment = total_score / total_volume if total_volume > 0 else 0
            
            # Get trending topics
            topic_counter = Counter(all_topics)
            trending_topics = [topic for topic, _ in topic_counter.most_common(10)]
            
            # Determine social mood
            mood = self._determine_social_mood(avg_sentiment, total_volume)
            
            # Get influencer sentiment
            influencer_sentiment = {}
            if twitter_data and 'influencers' in twitter_data:
                for influencer in twitter_data['influencers']:
                    influencer_sentiment[influencer['name']] = influencer['sentiment']
                    
            social_analysis = SocialAnalysis(
                platform="Aggregated",
                sentiment_score=avg_sentiment,
                volume=total_volume,
                trending_topics=trending_topics,
                influencer_sentiment=influencer_sentiment,
                mood=mood
            )
            
            self._social_cache[symbol] = social_analysis
            return social_analysis
            
        except Exception as e:
            logger.error(f"Social media analysis xatosi: {e}")
            return SocialAnalysis(
                platform="Error",
                sentiment_score=0,
                volume=0,
                trending_topics=[],
                influencer_sentiment={},
                mood=SocialMood.NEUTRAL
            )
            
    async def _get_market_sentiment(self) -> MarketSentiment:
        """Umumiy bozor hissiyotini olish"""
        try:
            # Check cache
            if self._market_sentiment and (TimeUtils.now_uzb() - self._market_sentiment.timestamp).seconds < 300:
                return self._market_sentiment
                
            # Get Fear & Greed Index
            fear_greed = await news_aggregator.get_fear_greed_index()
            
            # Get market overview
            market_data = await news_aggregator.get_market_overview()
            
            # Analyze overall news sentiment
            all_news = await news_aggregator.get_crypto_news("market")
            news_sentiment = self._calculate_news_sentiment(all_news)
            
            # Get social sentiment
            social_sentiment = await self._calculate_social_sentiment()
            
            # Calculate overall score
            overall_score = (news_sentiment * 0.4 + social_sentiment * 0.3 + (fear_greed - 50) * 2 * 0.3)
            
            # Get trending narratives
            narratives = await self._extract_market_narratives(all_news)
            
            # Identify risk events
            risk_events = await self._identify_risk_events(all_news)
            
            self._market_sentiment = MarketSentiment(
                overall_score=overall_score,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                fear_greed_index=fear_greed,
                trending_narratives=narratives,
                risk_events=risk_events
            )
            
            return self._market_sentiment
            
        except Exception as e:
            logger.error(f"Market sentiment olish xatosi: {e}")
            return MarketSentiment(
                overall_score=0,
                news_sentiment=0,
                social_sentiment=0,
                fear_greed_index=50,
                trending_narratives=[],
                risk_events=[]
            )
            
    async def _get_ai_analysis(self, symbol: str, news: List[NewsAnalysis], 
                              social: SocialAnalysis) -> Dict[str, Any]:
        """AI orqali chuqur tahlil"""
        try:
            # Prepare context
            news_context = "\n".join([
                f"- {n.title} (Impact: {n.impact.name}, Sentiment: {n.sentiment.name})"
                for n in news[:10]
            ])
            
            social_context = f"""
            Social sentiment: {social.sentiment_score:.2f}
            Volume: {social.volume}
            Mood: {social.mood.name}
            Trending: {', '.join(social.trending_topics[:5])}
            """
            
            prompt = f"""
            Analyze crypto sentiment for {symbol}:
            
            Recent News:
            {news_context}
            
            Social Media:
            {social_context}
            
            Provide:
            1. Overall sentiment assessment
            2. Key factors affecting sentiment
            3. Potential market impact
            4. Risk factors
            5. Trading recommendation
            """
            
            # Get AI analysis
            ai_response = await ai_orchestrator.analyze_sentiment(prompt)
            
            if not ai_response:
                return {"error": "AI analysis failed"}
                
            # Parse AI response
            analysis = {
                "summary": ai_response.get("summary", ""),
                "sentiment": ai_response.get("sentiment", "NEUTRAL"),
                "confidence": ai_response.get("confidence", 50),
                "key_factors": ai_response.get("factors", []),
                "market_impact": ai_response.get("impact", "MEDIUM"),
                "risks": ai_response.get("risks", []),
                "recommendation": ai_response.get("recommendation", "HOLD")
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis xatosi: {e}")
            return {"error": str(e)}
            
    def _analyze_text_sentiment(self, text: str) -> SentimentScore:
        """Matn sentimentini tahlil qilish"""
        text_lower = text.lower()
        
        # Count sentiment keywords
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        # Calculate net sentiment
        net_sentiment = bullish_count - bearish_count
        
        if net_sentiment >= 3:
            return SentimentScore.VERY_BULLISH
        elif net_sentiment >= 1:
            return SentimentScore.BULLISH
        elif net_sentiment <= -3:
            return SentimentScore.VERY_BEARISH
        elif net_sentiment <= -1:
            return SentimentScore.BEARISH
        else:
            return SentimentScore.NEUTRAL
            
    def _determine_news_impact(self, news_item: Dict[str, Any]) -> NewsImpact:
        """Yangilik ta'sirini aniqlash"""
        title_lower = news_item['title'].lower()
        content = news_item.get('description', '').lower()
        full_text = title_lower + " " + content
        
        # Check for high impact keywords
        high_impact_count = sum(1 for word in self.high_impact_keywords if word in full_text)
        
        # Check source credibility
        trusted_sources = ['reuters', 'bloomberg', 'coindesk', 'cointelegraph']
        is_trusted = any(source in news_item['source'].lower() for source in trusted_sources)
        
        if high_impact_count >= 2:
            return NewsImpact.CRITICAL
        elif high_impact_count >= 1 or is_trusted:
            return NewsImpact.HIGH
        elif news_item.get('social_score', 0) > 100:
            return NewsImpact.MEDIUM
        else:
            return NewsImpact.LOW
            
    def _extract_keywords(self, text: str) -> List[str]:
        """Kalit so'zlarni ajratish"""
        # Remove common words
        stop_words = {'the', 'is', 'at', 'to', 'and', 'of', 'for', 'in', 'on', 'with', 'as', 'by'}
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Get most common
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(5)]
        
    def _calculate_relevance(self, symbol: str, news_item: Dict[str, Any]) -> float:
        """Yangilik relevance skorini hisoblash"""
        relevance = 0.0
        
        # Check symbol mention
        symbol_clean = symbol.replace("USDT", "").replace("BUSD", "")
        if symbol_clean.lower() in news_item['title'].lower():
            relevance += 0.5
        if symbol_clean.lower() in news_item.get('description', '').lower():
            relevance += 0.3
            
        # Check for related tokens
        related_terms = {
            "BTC": ["bitcoin", "btc", "satoshi"],
            "ETH": ["ethereum", "eth", "vitalik"],
            "BNB": ["binance", "bnb", "cz"],
        }
        
        if symbol_clean in related_terms:
            for term in related_terms[symbol_clean]:
                if term in news_item['title'].lower():
                    relevance += 0.2
                    
        # Time relevance (newer = more relevant)
        age_hours = (TimeUtils.now_uzb() - news_item['published_at']).total_seconds() / 3600
        if age_hours < 1:
            relevance += 0.3
        elif age_hours < 6:
            relevance += 0.2
        elif age_hours < 24:
            relevance += 0.1
            
        return min(1.0, relevance)
        
    def _determine_social_mood(self, sentiment_score: float, volume: int) -> SocialMood:
        """Ijtimoiy kayfiyatni aniqlash"""
        # High volume amplifies mood
        if volume > 1000:
            if sentiment_score > 0.5:
                return SocialMood.EXTREME_GREED
            elif sentiment_score < -0.5:
                return SocialMood.EXTREME_FEAR
                
        # Normal mood determination
        if sentiment_score > 0.3:
            return SocialMood.GREED
        elif sentiment_score < -0.3:
            return SocialMood.FEAR
        else:
            return SocialMood.NEUTRAL
            
    def _calculate_news_sentiment(self, news_items: List[Dict[str, Any]]) -> float:
        """Yangiliklar sentimentini hisoblash"""
        if not news_items:
            return 0.0
            
        total_score = 0
        total_weight = 0
        
        for item in news_items[:50]:  # Top 50 news
            sentiment = self._analyze_text_sentiment(
                item['title'] + " " + item.get('description', '')
            )
            
            # Weight by recency
            age_hours = (TimeUtils.now_uzb() - item['published_at']).total_seconds() / 3600
            weight = 1 / (1 + age_hours / 24)  # Decay over 24 hours
            
            total_score += sentiment.value * weight * 20  # Scale to -100 to 100
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0
        
    async def _calculate_social_sentiment(self) -> float:
        """Ijtimoiy sentiment hisoblash"""
        try:
            # Get aggregated social data
            social_data = await news_aggregator.get_aggregated_social_sentiment()
            
            if not social_data:
                return 0.0
                
            # Weight different platforms
            weights = {
                'reddit': 0.4,
                'twitter': 0.4,
                'telegram': 0.2
            }
            
            total_score = 0
            total_weight = 0
            
            for platform, data in social_data.items():
                if platform in weights:
                    score = data.get('sentiment_score', 0) * 100  # Scale to -100 to 100
                    weight = weights[platform]
                    total_score += score * weight
                    total_weight += weight
                    
            return total_score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Social sentiment calculation xatosi: {e}")
            return 0.0
            
    async def _extract_market_narratives(self, news_items: List[Dict[str, Any]]) -> List[str]:
        """Bozor narrativlarini ajratish"""
        try:
            # Extract all text
            all_text = " ".join([
                item['title'] + " " + item.get('description', '')
                for item in news_items[:100]
            ])
            
            # Use AI to extract narratives
            ai_narratives = await ai_orchestrator.extract_narratives(all_text)
            
            if ai_narratives:
                return ai_narratives[:10]  # Top 10 narratives
                
            # Fallback to keyword extraction
            keywords = self._extract_keywords(all_text)
            return keywords[:10]
            
        except Exception as e:
            logger.error(f"Narrative extraction xatosi: {e}")
            return []
            
    async def _identify_risk_events(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Risk hodisalarini aniqlash"""
        risk_events = []
        
        risk_keywords = [
            'hack', 'exploit', 'bankruptcy', 'lawsuit', 'investigation',
            'regulation', 'ban', 'delisting', 'default', 'liquidation'
        ]
        
        for item in news_items[:50]:
            text_lower = (item['title'] + " " + item.get('description', '')).lower()
            
            for keyword in risk_keywords:
                if keyword in text_lower:
                    risk_events.append({
                        "type": keyword.upper(),
                        "title": item['title'],
                        "source": item['source'],
                        "timestamp": item['published_at'],
                        "severity": "HIGH" if keyword in ['hack', 'bankruptcy', 'ban'] else "MEDIUM"
                    })
                    break
                    
        # Sort by severity and recency
        risk_events.sort(key=lambda x: (
            0 if x['severity'] == 'HIGH' else 1,
            x['timestamp']
        ), reverse=True)
        
        return risk_events[:10]  # Top 10 risk events
        
    async def _generate_signal(self, symbol: str, news: List[NewsAnalysis],
                             social: SocialAnalysis, market: MarketSentiment,
                             ai_analysis: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Sentiment signalini yaratish"""
        try:
            reasoning = []
            
            # Calculate overall sentiment
            news_weight = 0.35
            social_weight = 0.25
            market_weight = 0.2
            ai_weight = 0.2
            
            # News sentiment score
            news_score = 0
            if news:
                news_sentiments = [n.sentiment.value for n in news[:10]]
                news_score = np.mean(news_sentiments) * 50  # Scale to -100 to 100
                
                # Add reasoning for high impact news
                for n in news[:3]:
                    if n.impact in [NewsImpact.CRITICAL, NewsImpact.HIGH]:
                        reasoning.append(f"📰 {n.impact.name}: {n.title[:80]}...")
                        
            # Social sentiment score
            social_score = social.sentiment_score * 100
            
            if social.mood == SocialMood.EXTREME_GREED:
                reasoning.append("🔥 Ijtimoiy tarmoqlarda extreme greed")
            elif social.mood == SocialMood.EXTREME_FEAR:
                reasoning.append("😱 Ijtimoiy tarmoqlarda extreme fear")
                
            # Market sentiment
            if market.fear_greed_index > 75:
                reasoning.append(f"📊 Fear & Greed Index: {market.fear_greed_index} (Extreme Greed)")
            elif market.fear_greed_index < 25:
                reasoning.append(f"📊 Fear & Greed Index: {market.fear_greed_index} (Extreme Fear)")
                
            # AI sentiment
            ai_sentiment_map = {
                "VERY_BULLISH": 100,
                "BULLISH": 50,
                "NEUTRAL": 0,
                "BEARISH": -50,
                "VERY_BEARISH": -100
            }
            ai_score = ai_sentiment_map.get(ai_analysis.get("sentiment", "NEUTRAL"), 0)
            
            # Calculate weighted sentiment
            overall_score = (
                news_score * news_weight +
                social_score * social_weight +
                market.overall_score * market_weight +
                ai_score * ai_weight
            )
            
            # Determine sentiment category
            if overall_score > 50:
                sentiment = SentimentScore.VERY_BULLISH
            elif overall_score > 20:
                sentiment = SentimentScore.BULLISH
            elif overall_score < -50:
                sentiment = SentimentScore.VERY_BEARISH
            elif overall_score < -20:
                sentiment = SentimentScore.BEARISH
            else:
                sentiment = SentimentScore.NEUTRAL
                
            # Add AI reasoning
            if ai_analysis.get("key_factors"):
                for factor in ai_analysis["key_factors"][:3]:
                    reasoning.append(f"🤖 {factor}")
                    
            # Risk events
            if market.risk_events:
                for event in market.risk_events[:2]:
                    reasoning.append(f"⚠️ {event['type']}: {event['title'][:60]}...")
                    
            # Trending topics
            if social.trending_topics:
                trending = ", ".join(social.trending_topics[:5])
                reasoning.append(f"📈 Trending: {trending}")
                
            # Calculate confidence
            confidence = self._calculate_confidence(news, social, ai_analysis)
            
            # Create signal
            signal = SentimentSignal(
                symbol=symbol,
                sentiment=sentiment,
                confidence=confidence,
                news_impact=news[:10],
                social_trends={
                    "mood": social.mood.name,
                    "volume": social.volume,
                    "trending": social.trending_topics[:10],
                    "influencers": social.influencer_sentiment
                },
                market_mood=market,
                ai_analysis=ai_analysis,
                reasoning=reasoning
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Sentiment signal generation xatosi: {e}")
            return None
            
    def _calculate_confidence(self, news: List[NewsAnalysis], 
                            social: SocialAnalysis, ai_analysis: Dict[str, Any]) -> float:
        """Signal ishonchini hisoblash"""
        confidence = 50.0
        
        # News volume and quality
        if len(news) >= 10:
            confidence += 10
        elif len(news) < 3:
            confidence -= 10
            
        # High impact news
        high_impact_count = sum(1 for n in news if n.impact in [NewsImpact.CRITICAL, NewsImpact.HIGH])
        confidence += min(20, high_impact_count * 5)
        
        # Social volume
        if social.volume > 1000:
            confidence += 15
        elif social.volume < 100:
            confidence -= 10
            
        # AI confidence
        ai_confidence = ai_analysis.get("confidence", 50)
        confidence = (confidence + ai_confidence) / 2
        
        # Normalize
        confidence = max(0, min(100, confidence))
        
        return confidence
        
    async def _update_market_sentiment(self):
        """Bozor sentimentini yangilash"""
        while True:
            try:
                await self._get_market_sentiment()
                logger.info("📊 Market sentiment yangilandi")
                
                # Alert on extreme conditions
                if self._market_sentiment:
                    if self._market_sentiment.fear_greed_index > 90:
                        logger.warning("⚠️ Extreme Greed detected in market!")
                    elif self._market_sentiment.fear_greed_index < 10:
                        logger.warning("⚠️ Extreme Fear detected in market!")
                        
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Market sentiment update xatosi: {e}")
                await asyncio.sleep(300)
                
    async def _monitor_breaking_news(self):
        """Muhim yangiliklarni kuzatish"""
        while True:
            try:
                # Get latest news
                breaking_news = await news_aggregator.get_breaking_news()
                
                for news in breaking_news:
                    impact = self._determine_news_impact(news)
                    
                    if impact == NewsImpact.CRITICAL:
                        logger.warning(f"🚨 CRITICAL NEWS: {news['title']}")
                        
                        # Analyze affected symbols
                        affected_symbols = self._extract_affected_symbols(news)
                        
                        for symbol in affected_symbols:
                            if symbol in config_manager.trading.symbols:
                                # Force sentiment update
                                await self.analyze(symbol)
                                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Breaking news monitor xatosi: {e}")
                await asyncio.sleep(60)
                
    def _extract_affected_symbols(self, news: Dict[str, Any]) -> List[str]:
        """Yangilikdan ta'sirlangan symbollarni ajratish"""
        affected = []
        text = (news['title'] + " " + news.get('description', '')).upper()
        
        # Check all trading symbols
        for symbol in config_manager.trading.symbols:
            token = symbol.replace("USDT", "").replace("BUSD", "")
            if token in text or token.lower() in news['title'].lower():
                affected.append(symbol)
                
        return affected
        
    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Sentiment xulosasi"""
        signal = self._sentiment_cache.get(symbol)
        
        if not signal:
            return {"error": "No sentiment data available"}
            
        return {
            "symbol": symbol,
            "sentiment": signal.sentiment.name,
            "confidence": signal.confidence,
            "news_count": len(signal.news_impact),
            "social_mood": signal.social_trends.get("mood"),
            "market_fear_greed": signal.market_mood.fear_greed_index,
            "ai_recommendation": signal.ai_analysis.get("recommendation", "HOLD"),
            "timestamp": signal.timestamp.isoformat()
        }

# Global instance
sentiment_analyzer = SentimentAnalyzer()
