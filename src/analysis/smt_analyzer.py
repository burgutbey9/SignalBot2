"""
SMT (Smart Money Theory) Analysis for Crypto Trading
Whale tracking, accumulation/distribution fazalari, on-chain tahlil
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import numpy as np
import pandas as pd

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils, RateLimiter, PerformanceMonitor
from api.trading_apis import unified_client
from core.timezone_handler import timezone_handler

logger = get_logger(__name__)

class WhalePhase(Enum):
    """Whale/Smart Money fazalari"""
    ACCUMULATION = auto()    # Yig'ish fazasi
    MARKUP = auto()          # Narx ko'tarilishi
    DISTRIBUTION = auto()    # Tarqatish fazasi
    MARKDOWN = auto()        # Narx tushishi
    REACCUMULATION = auto()  # Qayta yig'ish
    REDISTRIBUTION = auto()  # Qayta tarqatish

class VolumeProfile(Enum):
    """Hajm profili"""
    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    SPIKE = auto()
    CLIMAX = auto()

class OnChainSignal(Enum):
    """On-chain signallari"""
    WHALE_ACCUMULATION = auto()
    WHALE_DISTRIBUTION = auto()
    EXCHANGE_INFLOW = auto()
    EXCHANGE_OUTFLOW = auto()
    LARGE_TRANSFER = auto()
    DORMANT_MOVEMENT = auto()

@dataclass
class WhaleTransaction:
    """Whale tranzaksiyasi"""
    amount: float
    from_address: str
    to_address: str
    type: str  # TRANSFER, EXCHANGE_IN, EXCHANGE_OUT
    timestamp: datetime
    value_usd: float
    
@dataclass
class MarketProfile:
    """Bozor profili"""
    phase: WhalePhase
    volume_profile: VolumeProfile
    whale_activity: str  # LOW, MEDIUM, HIGH
    retail_sentiment: str  # FEAR, NEUTRAL, GREED
    divergence: bool = False
    strength: float = 0.0

@dataclass
class OnChainMetrics:
    """On-chain metrikalar"""
    exchange_netflow: float  # Positive = inflow, Negative = outflow
    large_transactions: int
    active_addresses: int
    dormant_coins_moved: float
    whale_accumulation_score: float
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class SMTSignal:
    """SMT Signal ma'lumotlari"""
    symbol: str
    phase: WhalePhase
    action: str  # BUY, SELL, HOLD
    strength: float
    on_chain_score: float
    volume_analysis: Dict[str, Any]
    whale_movements: List[Dict[str, Any]]
    reasoning: List[str]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

class SMTAnalyzer:
    """Smart Money Theory tahlilchisi"""
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_minute=20)
        self.performance_monitor = PerformanceMonitor()
        
        # Cache
        self._market_profiles: Dict[str, MarketProfile] = {}
        self._whale_transactions: Dict[str, List[WhaleTransaction]] = {}
        self._on_chain_metrics: Dict[str, OnChainMetrics] = {}
        self._volume_cache: Dict[str, pd.DataFrame] = {}
        
        # Parameters
        self.whale_threshold = 10000  # $10k minimum for whale tx
        self.volume_lookback = 20  # Volume analysis period
        self.accumulation_days = 30  # Accumulation phase check
        
    async def start(self):
        """Analyzer ishga tushirish"""
        logger.info("🐋 SMT Analyzer ishga tushmoqda...")
        asyncio.create_task(self._monitor_whale_activity())
        asyncio.create_task(self._update_on_chain_metrics())
        
    async def analyze(self, symbol: str) -> Optional[SMTSignal]:
        """To'liq SMT tahlil"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Get market data
            candles = await unified_client.get_candles(symbol, "1h", 168)  # 1 week
            if candles.empty:
                return None
                
            # Analyze components
            phase = await self._identify_market_phase(symbol, candles)
            volume_analysis = await self._analyze_volume_profile(symbol, candles)
            whale_activity = await self._get_whale_activity(symbol)
            on_chain = await self._get_on_chain_metrics(symbol)
            
            # Generate signal
            signal = await self._generate_signal(
                symbol, phase, volume_analysis, whale_activity, on_chain, candles
            )
            
            # Monitor performance
            self.performance_monitor.record("smt_analysis_complete", 1)
            
            return signal
            
        except Exception as e:
            logger.error(f"SMT analyze xatosi {symbol}: {e}")
            return None
            
    async def _identify_market_phase(self, symbol: str, candles: pd.DataFrame) -> WhalePhase:
        """Bozor fazasini aniqlash (Wyckoff Method)"""
        try:
            if len(candles) < 50:
                return WhalePhase.ACCUMULATION
                
            # Price and volume data
            prices = candles['close'].values
            volumes = candles['volume'].values
            
            # Calculate indicators
            sma20 = candles['close'].rolling(20).mean()
            sma50 = candles['close'].rolling(50).mean()
            
            # Volume analysis
            avg_volume = volumes[-20:].mean()
            recent_volume = volumes[-5:].mean()
            volume_trend = recent_volume / avg_volume
            
            # Price range analysis
            price_range = prices.max() - prices.min()
            current_position = (prices[-1] - prices.min()) / price_range
            
            # Trend analysis
            trend_up = sma20.iloc[-1] > sma50.iloc[-1]
            price_above_avg = prices[-1] > sma20.iloc[-1]
            
            # Phase identification logic
            if current_position < 0.3:  # Lower 30% of range
                if volume_trend > 1.2:  # Increasing volume
                    phase = WhalePhase.ACCUMULATION
                else:
                    phase = WhalePhase.MARKDOWN
                    
            elif current_position > 0.7:  # Upper 30% of range
                if volume_trend > 1.2:  # High volume
                    phase = WhalePhase.DISTRIBUTION
                else:
                    phase = WhalePhase.MARKUP
                    
            else:  # Middle range
                if trend_up and price_above_avg:
                    if volume_trend < 0.8:  # Decreasing volume
                        phase = WhalePhase.REACCUMULATION
                    else:
                        phase = WhalePhase.MARKUP
                else:
                    if volume_trend > 1.2:  # Increasing volume
                        phase = WhalePhase.REDISTRIBUTION
                    else:
                        phase = WhalePhase.MARKDOWN
                        
            # Cache result
            self._market_profiles[symbol] = MarketProfile(
                phase=phase,
                volume_profile=self._classify_volume_profile(volume_trend),
                whale_activity="MEDIUM",
                retail_sentiment="NEUTRAL",
                strength=self._calculate_phase_strength(candles, phase)
            )
            
            return phase
            
        except Exception as e:
            logger.error(f"Market phase aniqlash xatosi: {e}")
            return WhalePhase.ACCUMULATION
            
    async def _analyze_volume_profile(self, symbol: str, candles: pd.DataFrame) -> Dict[str, Any]:
        """Hajm profilini tahlil qilish"""
        try:
            volumes = candles['volume'].values
            prices = candles['close'].values
            
            # Volume indicators
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            recent_volume = np.mean(volumes[-10:])
            
            # Volume spikes
            volume_spikes = []
            for i, vol in enumerate(volumes):
                if vol > avg_volume + 2 * volume_std:
                    volume_spikes.append({
                        "index": i,
                        "volume": vol,
                        "price": prices[i],
                        "ratio": vol / avg_volume
                    })
                    
            # Volume trend
            volume_ma = pd.Series(volumes).rolling(10).mean()
            volume_trend = "INCREASING" if volume_ma.iloc[-1] > volume_ma.iloc[-10] else "DECREASING"
            
            # Price-volume divergence
            price_change = (prices[-1] - prices[-10]) / prices[-10]
            volume_change = (recent_volume - volumes[-20:-10].mean()) / volumes[-20:-10].mean()
            divergence = (price_change > 0 and volume_change < 0) or (price_change < 0 and volume_change > 0)
            
            # Volume profile levels
            price_bins = np.linspace(prices.min(), prices.max(), 20)
            volume_profile = []
            
            for i in range(len(price_bins) - 1):
                mask = (prices >= price_bins[i]) & (prices < price_bins[i+1])
                bin_volume = volumes[mask].sum()
                volume_profile.append({
                    "price_level": (price_bins[i] + price_bins[i+1]) / 2,
                    "volume": bin_volume,
                    "percentage": bin_volume / volumes.sum() * 100
                })
                
            # Find high volume nodes (HVN)
            volume_profile.sort(key=lambda x: x['volume'], reverse=True)
            hvn_levels = [vp['price_level'] for vp in volume_profile[:3]]
            
            return {
                "average_volume": avg_volume,
                "recent_volume": recent_volume,
                "volume_trend": volume_trend,
                "volume_spikes": volume_spikes,
                "divergence": divergence,
                "hvn_levels": hvn_levels,
                "volume_ratio": recent_volume / avg_volume
            }
            
        except Exception as e:
            logger.error(f"Volume profile tahlil xatosi: {e}")
            return {}
            
    async def _get_whale_activity(self, symbol: str) -> List[Dict[str, Any]]:
        """Whale faoliyatini olish"""
        try:
            # Get on-chain whale movements
            whale_data = await unified_client.get_whale_transactions(symbol)
            
            whale_movements = []
            for tx in whale_data:
                if tx['value_usd'] >= self.whale_threshold:
                    movement = {
                        "type": self._classify_whale_movement(tx),
                        "amount": tx['amount'],
                        "value_usd": tx['value_usd'],
                        "timestamp": tx['timestamp'],
                        "impact": self._assess_whale_impact(tx)
                    }
                    whale_movements.append(movement)
                    
            # Analyze patterns
            accumulation_count = sum(1 for m in whale_movements if m['type'] == "ACCUMULATION")
            distribution_count = sum(1 for m in whale_movements if m['type'] == "DISTRIBUTION")
            
            # Calculate whale score
            if accumulation_count > distribution_count * 1.5:
                whale_sentiment = "BULLISH"
            elif distribution_count > accumulation_count * 1.5:
                whale_sentiment = "BEARISH"
            else:
                whale_sentiment = "NEUTRAL"
                
            return {
                "movements": whale_movements[-10:],  # Last 10 movements
                "accumulation_count": accumulation_count,
                "distribution_count": distribution_count,
                "sentiment": whale_sentiment,
                "total_volume": sum(m['value_usd'] for m in whale_movements)
            }
            
        except Exception as e:
            logger.error(f"Whale activity olish xatosi: {e}")
            return {"movements": [], "sentiment": "NEUTRAL"}
            
    async def _get_on_chain_metrics(self, symbol: str) -> OnChainMetrics:
        """On-chain metrikalarni olish"""
        try:
            # Get on-chain data
            metrics = await unified_client.get_on_chain_metrics(symbol)
            
            # Process exchange flows
            exchange_inflow = metrics.get('exchange_inflow', 0)
            exchange_outflow = metrics.get('exchange_outflow', 0)
            netflow = exchange_inflow - exchange_outflow
            
            # Large transactions
            large_tx_count = metrics.get('large_transactions_count', 0)
            
            # Active addresses
            active_addresses = metrics.get('active_addresses', 0)
            
            # Dormant coins
            dormant_moved = metrics.get('dormant_coins_moved', 0)
            
            # Calculate accumulation score
            accumulation_score = self._calculate_accumulation_score(
                netflow, large_tx_count, active_addresses, dormant_moved
            )
            
            on_chain = OnChainMetrics(
                exchange_netflow=netflow,
                large_transactions=large_tx_count,
                active_addresses=active_addresses,
                dormant_coins_moved=dormant_moved,
                whale_accumulation_score=accumulation_score
            )
            
            self._on_chain_metrics[symbol] = on_chain
            return on_chain
            
        except Exception as e:
            logger.error(f"On-chain metrics olish xatosi: {e}")
            return OnChainMetrics(
                exchange_netflow=0,
                large_transactions=0,
                active_addresses=0,
                dormant_coins_moved=0,
                whale_accumulation_score=50
            )
            
    def _classify_whale_movement(self, transaction: Dict[str, Any]) -> str:
        """Whale harakatini tasniflash"""
        # Exchange addresses check
        if "exchange" in transaction.get('to_address', '').lower():
            return "DISTRIBUTION"  # Selling
        elif "exchange" in transaction.get('from_address', '').lower():
            return "ACCUMULATION"  # Buying
        elif transaction.get('to_address', '') in ['cold_wallet', 'storage']:
            return "ACCUMULATION"  # Long-term hold
        else:
            return "TRANSFER"  # Neutral
            
    def _assess_whale_impact(self, transaction: Dict[str, Any]) -> str:
        """Whale ta'sirini baholash"""
        value = transaction['value_usd']
        
        if value > 1000000:  # $1M+
            return "HIGH"
        elif value > 100000:  # $100k+
            return "MEDIUM"
        else:
            return "LOW"
            
    def _calculate_accumulation_score(self, netflow: float, large_tx: int, 
                                    active_addr: int, dormant: float) -> float:
        """Accumulation skorini hisoblash"""
        score = 50.0  # Base score
        
        # Exchange netflow (negative = accumulation)
        if netflow < 0:
            score += min(30, abs(netflow) / 1000000 * 10)  # Max 30 points
        else:
            score -= min(30, netflow / 1000000 * 10)
            
        # Large transactions
        score += min(10, large_tx / 100 * 10)  # Max 10 points
        
        # Active addresses
        if active_addr > 10000:
            score += 5
        elif active_addr < 5000:
            score -= 5
            
        # Dormant coins movement (bearish signal)
        if dormant > 0:
            score -= min(10, dormant / 1000000 * 5)
            
        return max(0, min(100, score))
        
    def _classify_volume_profile(self, volume_trend: float) -> VolumeProfile:
        """Hajm profilini tasniflash"""
        if volume_trend > 2.0:
            return VolumeProfile.CLIMAX
        elif volume_trend > 1.5:
            return VolumeProfile.SPIKE
        elif volume_trend > 1.1:
            return VolumeProfile.INCREASING
        elif volume_trend < 0.7:
            return VolumeProfile.DECREASING
        else:
            return VolumeProfile.STABLE
            
    def _calculate_phase_strength(self, candles: pd.DataFrame, phase: WhalePhase) -> float:
        """Faza kuchini hisoblash"""
        try:
            # Price momentum
            returns = candles['close'].pct_change()
            momentum = returns.rolling(20).mean().iloc[-1]
            
            # Volume confirmation
            volume_ma = candles['volume'].rolling(20).mean()
            volume_strength = candles['volume'].iloc[-1] / volume_ma.iloc[-1]
            
            # Phase-specific strength
            if phase == WhalePhase.ACCUMULATION:
                # Low volatility + stable volume = strong accumulation
                volatility = returns.rolling(20).std().iloc[-1]
                strength = (1 - volatility * 10) * 50 + volume_strength * 25
                
            elif phase == WhalePhase.DISTRIBUTION:
                # High volume + price weakness = strong distribution
                strength = volume_strength * 50 + (1 - momentum) * 25
                
            elif phase in [WhalePhase.MARKUP, WhalePhase.MARKDOWN]:
                # Trend strength
                strength = abs(momentum) * 100 + volume_strength * 25
                
            else:
                # Re-accumulation/redistribution
                strength = 50 + volume_strength * 25
                
            return max(0, min(100, strength))
            
        except:
            return 50.0
            
    async def _generate_signal(self, symbol: str, phase: WhalePhase,
                             volume_analysis: Dict[str, Any], whale_activity: Dict[str, Any],
                             on_chain: OnChainMetrics, candles: pd.DataFrame) -> Optional[SMTSignal]:
        """SMT signalini yaratish"""
        try:
            reasoning = []
            action = "HOLD"
            strength = 50.0
            
            # Phase-based action
            if phase == WhalePhase.ACCUMULATION:
                action = "BUY"
                reasoning.append("🟢 Accumulation fazasi aniqlandi")
                strength += 20
                
            elif phase == WhalePhase.MARKUP:
                if volume_analysis.get('volume_trend') == "INCREASING":
                    action = "BUY"
                    reasoning.append("📈 Markup fazasi + hajm oshmoqda")
                    strength += 15
                else:
                    action = "HOLD"
                    reasoning.append("📊 Markup fazasi, lekin hajm kamayyapti")
                    
            elif phase == WhalePhase.DISTRIBUTION:
                action = "SELL"
                reasoning.append("🔴 Distribution fazasi aniqlandi")
                strength -= 20
                
            elif phase == WhalePhase.MARKDOWN:
                action = "SELL"
                reasoning.append("📉 Markdown fazasi")
                strength -= 15
                
            # Whale activity adjustment
            whale_sentiment = whale_activity.get('sentiment', 'NEUTRAL')
            if whale_sentiment == "BULLISH":
                reasoning.append(f"🐋 Whale accumulation: {whale_activity['accumulation_count']} ta")
                strength += 15
                if action == "SELL":
                    action = "HOLD"  # Conflict resolution
                    
            elif whale_sentiment == "BEARISH":
                reasoning.append(f"🐳 Whale distribution: {whale_activity['distribution_count']} ta")
                strength -= 15
                if action == "BUY":
                    action = "HOLD"  # Conflict resolution
                    
            # On-chain signals
            if on_chain.exchange_netflow < -1000000:  # $1M+ outflow
                reasoning.append(f"💸 Exchange outflow: ${abs(on_chain.exchange_netflow):,.0f}")
                strength += 10
                
            elif on_chain.exchange_netflow > 1000000:  # $1M+ inflow
                reasoning.append(f"💰 Exchange inflow: ${on_chain.exchange_netflow:,.0f}")
                strength -= 10
                
            # Volume analysis
            if volume_analysis.get('divergence'):
                reasoning.append("⚠️ Price-volume divergence aniqlandi")
                strength -= 5
                
            # Volume spikes
            recent_spikes = [s for s in volume_analysis.get('volume_spikes', []) if s['index'] > -10]
            if recent_spikes:
                spike = recent_spikes[-1]
                reasoning.append(f"📊 Volume spike: {spike['ratio']:.1f}x average")
                
            # HVN levels
            current_price = candles['close'].iloc[-1]
            hvn_levels = volume_analysis.get('hvn_levels', [])
            
            for level in hvn_levels:
                if abs(current_price - level) / current_price < 0.02:  # Within 2%
                    reasoning.append(f"🎯 HVN level yaqinida: ${level:.2f}")
                    
            # Normalize strength
            strength = max(0, min(100, strength))
            
            # Create signal
            signal = SMTSignal(
                symbol=symbol,
                phase=phase,
                action=action,
                strength=strength,
                on_chain_score=on_chain.whale_accumulation_score,
                volume_analysis=volume_analysis,
                whale_movements=whale_activity.get('movements', [])[-5:],
                reasoning=reasoning
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"SMT signal generation xatosi: {e}")
            return None
            
    async def _monitor_whale_activity(self):
        """Whale faoliyatini monitoring qilish"""
        while True:
            try:
                symbols = config_manager.trading.symbols
                
                for symbol in symbols:
                    # Get real-time whale alerts
                    whale_alerts = await unified_client.get_whale_alerts(symbol)
                    
                    for alert in whale_alerts:
                        if alert['value_usd'] > 100000:  # $100k+
                            logger.info(f"🐋 Whale Alert {symbol}: ${alert['value_usd']:,.0f} {alert['type']}")
                            
                            # Store transaction
                            tx = WhaleTransaction(
                                amount=alert['amount'],
                                from_address=alert.get('from', 'unknown'),
                                to_address=alert.get('to', 'unknown'),
                                type=alert['type'],
                                timestamp=TimeUtils.now_uzb(),
                                value_usd=alert['value_usd']
                            )
                            
                            if symbol not in self._whale_transactions:
                                self._whale_transactions[symbol] = []
                            self._whale_transactions[symbol].append(tx)
                            
                            # Keep only recent transactions
                            self._whale_transactions[symbol] = self._whale_transactions[symbol][-100:]
                            
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                await asyncio.sleep(60)  # Full cycle every minute
                
            except Exception as e:
                logger.error(f"Whale monitoring xatosi: {e}")
                await asyncio.sleep(300)
                
    async def _update_on_chain_metrics(self):
        """On-chain metrikalarni yangilash"""
        while True:
            try:
                symbols = config_manager.trading.symbols
                
                for symbol in symbols:
                    await self._get_on_chain_metrics(symbol)
                    await asyncio.sleep(10)  # Rate limiting
                    
                # Wait before next update
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"On-chain update xatosi: {e}")
                await asyncio.sleep(300)
                
    def get_market_overview(self) -> Dict[str, Any]:
        """Bozor umumiy ko'rinishi"""
        overview = {
            "timestamp": TimeUtils.now_uzb().isoformat(),
            "symbols": {}
        }
        
        for symbol, profile in self._market_profiles.items():
            on_chain = self._on_chain_metrics.get(symbol)
            
            overview["symbols"][symbol] = {
                "phase": profile.phase.name,
                "volume_profile": profile.volume_profile.name,
                "whale_activity": profile.whale_activity,
                "accumulation_score": on_chain.whale_accumulation_score if on_chain else 50,
                "strength": profile.strength
            }
            
        return overview

# Global instance
smt_analyzer = SMTAnalyzer()
