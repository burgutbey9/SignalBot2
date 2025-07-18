"""
News and Social Media API Clients for Crypto Sentiment
NewsAPI, Reddit, CryptoPanic integration
"""
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
from enum import Enum, auto

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import RateLimiter, retry_manager, TimeUtils, safe_request

logger = get_logger(__name__)

class NewsSource(Enum):
    """Yangilik manbalari"""
    NEWSAPI = auto()
    REDDIT = auto()
    CRYPTOPANIC = auto()
    TWITTER = auto()
    CUSTOM = auto()

@dataclass
class NewsItem:
    """Yangilik elementi"""
    id: str
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    author: Optional[str] = None
    image_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)  # Mentioned coins
    importance: int = 3  # 1-5 scale
    social_score: Optional[float] = None
    
@dataclass
class RedditPost:
    """Reddit post ma'lumotlari"""
    id: str
    title: str
    content: str
    subreddit: str
    author: str
    score: int
    comments: int
    created_at: datetime
    url: str
    flair: Optional[str] = None
    awards: int = 0
    upvote_ratio: float = 0.0

class NewsAPIClient:
    """NewsAPI.org client"""
    def __init__(self):
        self.api_key = config_manager.get("apis.newsapi.key", "")
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter(calls=100, period=86400)  # 100/day
        
    async def fetch_crypto_news(self, 
                               query: str = "cryptocurrency OR bitcoin OR ethereum",
                               from_date: Optional[datetime] = None,
                               language: str = "en",
                               sort_by: str = "publishedAt") -> List[NewsItem]:
        """Crypto yangiliklarni olish"""
        if not self.api_key:
            logger.error("NewsAPI key topilmadi")
            return []
            
        try:
            await self.rate_limiter.acquire()
            
            if not from_date:
                from_date = datetime.now() - timedelta(hours=24)
                
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "sortBy": sort_by,
                "language": language,
                "apiKey": self.api_key,
                "pageSize": 100
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/everything"
                result = await safe_request(session, "GET", url, params=params)
                
                if result["status"] == 200:
                    data = result["data"]
                    return self._parse_news_response(data)
                else:
                    logger.error(f"NewsAPI xatosi: {result['status']}")
                    return []
                    
        except Exception as e:
            logger.error(f"NewsAPI fetch xatosi: {e}")
            return []
            
    def _parse_news_response(self, data: Dict[str, Any]) -> List[NewsItem]:
        """NewsAPI javobini parse qilish"""
        news_items = []
        
        for article in data.get("articles", []):
            try:
                # Skip qilish kerak bo'lgan manbalar
                if article.get("source", {}).get("name", "").lower() in ["removed", "[removed]"]:
                    continue
                    
                news_item = NewsItem(
                    id=hashlib.md5(article.get("url", "").encode()).hexdigest()[:16],
                    title=article.get("title", ""),
                    description=article.get("description", "") or article.get("content", "")[:200],
                    source=article.get("source", {}).get("name", "Unknown"),
                    url=article.get("url", ""),
                    published_at=datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "+00:00")),
                    author=article.get("author"),
                    image_url=article.get("urlToImage"),
                    categories=self._extract_categories(article),
                    mentions=self._extract_coin_mentions(article)
                )
                
                news_items.append(news_item)
                
            except Exception as e:
                logger.warning(f"Article parse xatosi: {e}")
                continue
                
        return news_items
        
    def _extract_categories(self, article: Dict[str, Any]) -> List[str]:
        """Kategoriyalarni aniqlash"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        categories = []
        
        category_keywords = {
            "defi": ["defi", "decentralized finance", "yield", "liquidity"],
            "regulation": ["sec", "regulation", "government", "law", "legal"],
            "market": ["price", "pump", "dump", "bull", "bear", "rally"],
            "technology": ["upgrade", "fork", "development", "release", "update"],
            "hack": ["hack", "exploit", "vulnerability", "attack", "breach"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
                
        return categories[:3]
        
    def _extract_coin_mentions(self, article: Dict[str, Any]) -> List[str]:
        """Tilga olingan coinlarni topish"""
        text = f"{article.get('title', '')} {article.get('description', '')}".upper()
        
        coins = {
            "BTC": ["BITCOIN", "BTC"],
            "ETH": ["ETHEREUM", "ETH"],
            "BNB": ["BINANCE", "BNB"],
            "SOL": ["SOLANA", "SOL"],
            "LINK": ["CHAINLINK", "LINK"],
            "UNI": ["UNISWAP", "UNI"],
            "AAVE": ["AAVE"],
            "MATIC": ["POLYGON", "MATIC"],
            "AVAX": ["AVALANCHE", "AVAX"],
            "DOT": ["POLKADOT", "DOT"],
            "ADA": ["CARDANO", "ADA"],
            "ATOM": ["COSMOS", "ATOM"]
        }
        
        mentioned = []
        for symbol, keywords in coins.items():
            if any(keyword in text for keyword in keywords):
                mentioned.append(symbol)
                
        return mentioned[:5]

class RedditClient:
    """Reddit API client"""
    def __init__(self):
        self.client_id = config_manager.get("apis.reddit.client_id", "")
        self.client_secret = config_manager.get("apis.reddit.client_secret", "")
        self.user_agent = "CryptoSentimentBot/1.0"
        self.base_url = "https://oauth.reddit.com"
        self.auth_url = "https://www.reddit.com/api/v1/access_token"
        self.rate_limiter = RateLimiter(calls=60, period=60)
        self.access_token = None
        self.token_expires = 0
        
    async def _get_access_token(self) -> Optional[str]:
        """Reddit access token olish"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
            
        try:
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            headers = {"User-Agent": self.user_agent}
            data = {"grant_type": "client_credentials"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.auth_url, auth=auth, data=data, headers=headers) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data.get("access_token")
                        self.token_expires = time.time() + token_data.get("expires_in", 3600) - 60
                        return self.access_token
                    else:
                        logger.error(f"Reddit auth xatosi: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Reddit token olishda xato: {e}")
            return None
            
    async def fetch_crypto_posts(self, 
                                subreddits: List[str] = None,
                                sort: str = "hot",
                                time_filter: str = "day",
                                limit: int = 100) -> List[RedditPost]:
        """Reddit crypto postlarini olish"""
        if not subreddits:
            subreddits = ["cryptocurrency", "bitcoin", "ethereum", "defi", "cryptomarkets"]
            
        if not self.client_id or not self.client_secret:
            logger.error("Reddit credentials topilmadi")
            return []
            
        token = await self._get_access_token()
        if not token:
            return []
            
        all_posts = []
        
        for subreddit in subreddits:
            try:
                posts = await self._fetch_subreddit_posts(subreddit, sort, time_filter, limit)
                all_posts.extend(posts)
            except Exception as e:
                logger.error(f"r/{subreddit} fetch xatosi: {e}")
                continue
                
        return all_posts
        
    async def _fetch_subreddit_posts(self, subreddit: str, sort: str, time_filter: str, limit: int) -> List[RedditPost]:
        """Bitta subredditdan postlar olish"""
        await self.rate_limiter.acquire()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }
        
        params = {"limit": min(limit, 100), "t": time_filter}
        url = f"{self.base_url}/r/{subreddit}/{sort}"
        
        async with aiohttp.ClientSession() as session:
            result = await safe_request(session, "GET", url, headers=headers, params=params)
            
            if result["status"] == 200:
                return self._parse_reddit_response(result["data"], subreddit)
            else:
                logger.error(f"Reddit API xatosi: {result['status']}")
                return []
                
    def _parse_reddit_response(self, data: Dict[str, Any], subreddit: str) -> List[RedditPost]:
        """Reddit javobini parse qilish"""
        posts = []
        
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            
            # Filter criteria
            if post_data.get("is_video") or post_data.get("stickied"):
                continue
                
            try:
                post = RedditPost(
                    id=post_data.get("id", ""),
                    title=post_data.get("title", ""),
                    content=post_data.get("selftext", "")[:1000],  # Limit content
                    subreddit=subreddit,
                    author=post_data.get("author", "deleted"),
                    score=post_data.get("score", 0),
                    comments=post_data.get("num_comments", 0),
                    created_at=datetime.fromtimestamp(post_data.get("created_utc", 0)),
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    flair=post_data.get("link_flair_text"),
                    awards=post_data.get("total_awards_received", 0),
                    upvote_ratio=post_data.get("upvote_ratio", 0.0)
                )
                
                posts.append(post)
                
            except Exception as e:
                logger.warning(f"Reddit post parse xatosi: {e}")
                continue
                
        return posts

class CryptoPanicClient:
    """CryptoPanic API client (alternative news source)"""
    def __init__(self):
        self.api_key = config_manager.get("apis.cryptopanic.key", "")
        self.base_url = "https://cryptopanic.com/api/v1"
        self.rate_limiter = RateLimiter(calls=1000, period=3600)  # 1000/hour
        
    async def fetch_news(self, filter_type: str = "rising", currencies: Optional[List[str]] = None) -> List[NewsItem]:
        """CryptoPanic yangiliklarini olish"""
        if not self.api_key:
            logger.warning("CryptoPanic API key yo'q, skip")
            return []
            
        try:
            await self.rate_limiter.acquire()
            
            params = {
                "auth_token": self.api_key,
                "filter": filter_type,  # rising, hot, bullish, bearish
                "public": "true"
            }
            
            if currencies:
                params["currencies"] = ",".join(currencies)
                
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/posts/"
                result = await safe_request(session, "GET", url, params=params)
                
                if result["status"] == 200:
                    return self._parse_cryptopanic_response(result["data"])
                else:
                    logger.error(f"CryptoPanic xatosi: {result['status']}")
                    return []
                    
        except Exception as e:
            logger.error(f"CryptoPanic fetch xatosi: {e}")
            return []
            
    def _parse_cryptopanic_response(self, data: Dict[str, Any]) -> List[NewsItem]:
        """CryptoPanic javobini parse qilish"""
        news_items = []
        
        for result in data.get("results", []):
            try:
                # Votes dan importance hisoblash
                votes = result.get("votes", {})
                importance = self._calculate_importance(votes)
                
                news_item = NewsItem(
                    id=str(result.get("id", "")),
                    title=result.get("title", ""),
                    description=result.get("body", "")[:200] if result.get("body") else "",
                    source=result.get("source", {}).get("title", "Unknown"),
                    url=result.get("url", ""),
                    published_at=datetime.fromisoformat(result.get("published_at", "").replace("Z", "+00:00")),
                    categories=[result.get("kind", "news")],
                    mentions=[c["code"] for c in result.get("currencies", [])],
                    importance=importance,
                    social_score=self._calculate_social_score(votes)
                )
                
                news_items.append(news_item)
                
            except Exception as e:
                logger.warning(f"CryptoPanic item parse xatosi: {e}")
                continue
                
        return news_items
        
    def _calculate_importance(self, votes: Dict[str, Any]) -> int:
        """Ovozlar asosida muhimlikni hisoblash"""
        total = votes.get("positive", 0) + votes.get("negative", 0)
        
        if total > 100: return 5
        elif total > 50: return 4
        elif total > 20: return 3
        elif total > 5: return 2
        else: return 1
        
    def _calculate_social_score(self, votes: Dict[str, Any]) -> float:
        """Social score hisoblash"""
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)
        total = positive + negative
        
        if total == 0:
            return 0.5
            
        return positive / total

class SocialMediaAggregator:
    """Barcha social media manbalarini birlashtirish"""
    def __init__(self):
        self.news_api = NewsAPIClient()
        self.reddit = RedditClient()
        self.cryptopanic = CryptoPanicClient()
        self._cache: Dict[str, Tuple[List[Any], datetime]] = {}
        self.cache_duration = timedelta(minutes=15)
        
    async def fetch_all_news(self, 
                           query: Optional[str] = None,
                           from_hours: int = 24,
                           include_reddit: bool = True) -> Dict[str, List[Any]]:
        """Barcha manbalardan yangiliklar olish"""
        cache_key = f"{query}_{from_hours}_{include_reddit}"
        
        # Cache tekshirish
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.info("📰 Using cached news data")
                return cached_data
                
        results = {
            "news": [],
            "reddit": [],
            "cryptopanic": []
        }
        
        from_date = datetime.now() - timedelta(hours=from_hours)
        
        # Parallel fetch
        tasks = []
        
        # NewsAPI
        if query:
            tasks.append(self.news_api.fetch_crypto_news(query, from_date))
        else:
            tasks.append(self.news_api.fetch_crypto_news(from_date=from_date))
            
        # Reddit
        if include_reddit:
            tasks.append(self.reddit.fetch_crypto_posts())
            
        # CryptoPanic
        tasks.append(self.cryptopanic.fetch_news())
        
        try:
            fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(fetched_results):
                if isinstance(result, Exception):
                    logger.error(f"Fetch xatosi: {result}")
                    continue
                    
                if i == 0:  # NewsAPI
                    results["news"] = result
                elif i == 1 and include_reddit:  # Reddit
                    results["reddit"] = result
                elif (i == 2 and not include_reddit) or (i == 2 and include_reddit):  # CryptoPanic
                    results["cryptopanic"] = result
                    
            # Cache saqlash
            self._cache[cache_key] = (results, datetime.now())
            
            # Statistics
            total_items = sum(len(items) for items in results.values())
            logger.info(f"📰 Fetched {total_items} items: News={len(results['news'])}, Reddit={len(results['reddit'])}, CryptoPanic={len(results['cryptopanic'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Social media aggregation xatosi: {e}")
            return results
            
    async def get_trending_topics(self, hours: int = 6) -> List[Dict[str, Any]]:
        """Trending mavzularni aniqlash"""
        all_news = await self.fetch_all_news(from_hours=hours, include_reddit=True)
        
        # Combine all sources
        all_items = []
        all_items.extend(all_news.get("news", []))
        all_items.extend([self._reddit_to_news(p) for p in all_news.get("reddit", [])])
        all_items.extend(all_news.get("cryptopanic", []))
        
        # Extract topics
        topic_counts = {}
        
        for item in all_items:
            # Title va description dan topiclar
            text = f"{getattr(item, 'title', '')} {getattr(item, 'description', '')}".lower()
            
            # Common crypto topics
            topics = {
                "bitcoin": ["bitcoin", "btc"],
                "ethereum": ["ethereum", "eth"],
                "defi": ["defi", "decentralized finance"],
                "nft": ["nft", "non-fungible"],
                "regulation": ["sec", "regulation", "government"],
                "hack": ["hack", "exploit", "attack"],
                "bullish": ["bull", "bullish", "pump", "moon"],
                "bearish": ["bear", "bearish", "dump", "crash"]
            }
            
            for topic, keywords in topics.items():
                if any(keyword in text for keyword in keywords):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    
        # Sort by count
        trending = [
            {"topic": topic, "count": count, "percentage": (count / len(all_items)) * 100}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return trending[:10]  # Top 10
        
    def _reddit_to_news(self, post: RedditPost) -> NewsItem:
        """Reddit postni NewsItem formatiga o'tkazish"""
        return NewsItem(
            id=post.id,
            title=post.title,
            description=post.content[:200],
            source=f"r/{post.subreddit}",
            url=post.url,
            published_at=post.created_at,
            author=post.author,
            categories=["reddit"],
            social_score=post.upvote_ratio
        )
        
    async def get_market_mood(self) -> Dict[str, Any]:
        """Umumiy bozor kayfiyatini aniqlash"""
        trending = await self.get_trending_topics(hours=12)
        
        bullish_count = next((t["count"] for t in trending if t["topic"] == "bullish"), 0)
        bearish_count = next((t["count"] for t in trending if t["topic"] == "bearish"), 0)
        
        total_sentiment = bullish_count + bearish_count
        
        if total_sentiment > 0:
            bullish_ratio = bullish_count / total_sentiment
        else:
            bullish_ratio = 0.5
            
        mood = "NEUTRAL"
        if bullish_ratio > 0.7:
            mood = "VERY_BULLISH"
        elif bullish_ratio > 0.6:
            mood = "BULLISH"
        elif bullish_ratio < 0.3:
            mood = "VERY_BEARISH"
        elif bullish_ratio < 0.4:
            mood = "BEARISH"
            
        return {
            "mood": mood,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": 1 - bullish_ratio,
            "trending_topics": trending[:5],
            "timestamp": TimeUtils.now_uzb()
        }

# Global instance
social_media_aggregator = SocialMediaAggregator()

import hashlib
import time
