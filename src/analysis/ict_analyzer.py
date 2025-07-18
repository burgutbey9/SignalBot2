"""
ICT (Inner Circle Trader) Analysis for Crypto Trading
Crypto uchun ICT kontseptsiyalari: PDH/PDL, SSL/BSL, FVG, Order Blocks
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
from core.timezone_handler import timezone_handler, TradingSession

logger = get_logger(__name__)

class MarketStructure(Enum):
    """Bozor strukturasi"""
    BULLISH = auto()
    BEARISH = auto()
    RANGING = auto()
    TRANSITIONING = auto()

class LiquidityType(Enum):
    """Likvidlik turlari"""
    BSL = auto()  # Buy Side Liquidity
    SSL = auto()  # Sell Side Liquidity
    EQH = auto()  # Equal Highs
    EQL = auto()  # Equal Lows

@dataclass
class PriceLevel:
    """Narx darajasi"""
    price: float
    timestamp: datetime
    type: str  # HIGH, LOW, CLOSE
    strength: int = 1  # 1-5 kuchlilik darajasi

@dataclass
class FairValueGap:
    """Fair Value Gap (FVG)"""
    high: float
    low: float
    timestamp: datetime
    type: str  # BULLISH, BEARISH
    filled: bool = False
    strength: float = 0.0  # Gap kuchi

@dataclass
class OrderBlock:
    """Order Block"""
    high: float
    low: float
    timestamp: datetime
    type: str  # BULLISH, BEARISH
    volume: float
    strength: float = 0.0
    mitigated: bool = False

@dataclass
class LiquidityPool:
    """Likvidlik havzasi"""
    price: float
    type: LiquidityType
    timestamp: datetime
    swept: bool = False
    volume: float = 0.0

@dataclass
class ICTSignal:
    """ICT Signal ma'lumotlari"""
    symbol: str
    structure: MarketStructure
    bias: str  # BULLISH, BEARISH, NEUTRAL
    entry_zones: List[Dict[str, float]]
    stop_loss: float
    take_profit: List[float]
    confidence: float
    reasoning: List[str]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

class ICTAnalyzer:
    """ICT tahlil qiluvchi"""
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_minute=30)
        self.performance_monitor = PerformanceMonitor()
        
        # Cache
        self._candle_cache: Dict[str, pd.DataFrame] = {}
        self._pdh_pdl_cache: Dict[str, Dict[str, PriceLevel]] = {}
        self._structure_cache: Dict[str, MarketStructure] = {}
        
        # Analysis components
        self._order_blocks: Dict[str, List[OrderBlock]] = {}
        self._fvg_list: Dict[str, List[FairValueGap]] = {}
        self._liquidity_pools: Dict[str, List[LiquidityPool]] = {}
        
        # Parameters
        self.fvg_min_size = 0.001  # 0.1% minimum FVG size
        self.ob_lookback = 10  # Order block lookback candles
        self.structure_lookback = 50  # Market structure lookback
        
    async def start(self):
        """Analyzer ishga tushirish"""
        logger.info("🎯 ICT Analyzer ishga tushmoqda...")
        asyncio.create_task(self._update_loop())
        
    async def analyze(self, symbol: str) -> Optional[ICTSignal]:
        """To'liq ICT tahlil"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Get candles
            candles = await self._get_candles(symbol, "15m", limit=200)
            if candles.empty:
                return None
                
            # Analyze components
            structure = await self._analyze_market_structure(symbol, candles)
            pdh_pdl = await self._get_pdh_pdl(symbol)
            order_blocks = await self._find_order_blocks(symbol, candles)
            fvgs = await self._find_fvg(symbol, candles)
            liquidity = await self._find_liquidity_pools(symbol, candles)
            
            # Generate signal
            signal = await self._generate_signal(
                symbol, structure, pdh_pdl, order_blocks, fvgs, liquidity, candles
            )
            
            # Monitor performance
            self.performance_monitor.record("ict_analysis_complete", 1)
            
            return signal
            
        except Exception as e:
            logger.error(f"ICT analyze xatosi {symbol}: {e}")
            return None
            
    async def _get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Shamlarni olish"""
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._candle_cache:
                cached = self._candle_cache[cache_key]
                if len(cached) >= limit and (TimeUtils.now_uzb() - cached.index[-1]).seconds < 60:
                    return cached
                    
            # Fetch new candles
            candles = await unified_client.get_candles(symbol, timeframe, limit)
            
            if not candles.empty:
                self._candle_cache[cache_key] = candles
                
            return candles
            
        except Exception as e:
            logger.error(f"Candles olish xatosi: {e}")
            return pd.DataFrame()
            
    async def _analyze_market_structure(self, symbol: str, candles: pd.DataFrame) -> MarketStructure:
        """Bozor strukturasini tahlil qilish"""
        try:
            if len(candles) < self.structure_lookback:
                return MarketStructure.RANGING
                
            # Get swing points
            highs = candles['high'].rolling(5).max() == candles['high']
            lows = candles['low'].rolling(5).min() == candles['low']
            
            swing_highs = candles[highs]['high'].values[-10:]
            swing_lows = candles[lows]['low'].values[-10:]
            
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return MarketStructure.RANGING
                
            # Check trend
            hh = swing_highs[-1] > swing_highs[-2]  # Higher High
            hl = swing_lows[-1] > swing_lows[-2]    # Higher Low
            lh = swing_highs[-1] < swing_highs[-2]  # Lower High
            ll = swing_lows[-1] < swing_lows[-2]    # Lower Low
            
            if hh and hl:
                structure = MarketStructure.BULLISH
            elif lh and ll:
                structure = MarketStructure.BEARISH
            elif (hh and ll) or (lh and hl):
                structure = MarketStructure.TRANSITIONING
            else:
                structure = MarketStructure.RANGING
                
            self._structure_cache[symbol] = structure
            return structure
            
        except Exception as e:
            logger.error(f"Market structure tahlil xatosi: {e}")
            return MarketStructure.RANGING
            
    async def _get_pdh_pdl(self, symbol: str) -> Dict[str, PriceLevel]:
        """Previous Day High/Low olish"""
        try:
            # Check cache
            today = TimeUtils.now_uzb().date()
            cache_key = f"{symbol}_{today}"
            
            if cache_key in self._pdh_pdl_cache:
                return self._pdh_pdl_cache[cache_key]
                
            # Get daily candles
            daily_candles = await self._get_candles(symbol, "1d", limit=2)
            
            if len(daily_candles) >= 2:
                prev_day = daily_candles.iloc[-2]
                
                pdh_pdl = {
                    "PDH": PriceLevel(
                        price=prev_day['high'],
                        timestamp=prev_day.name,
                        type="HIGH",
                        strength=3
                    ),
                    "PDL": PriceLevel(
                        price=prev_day['low'],
                        timestamp=prev_day.name,
                        type="LOW",
                        strength=3
                    )
                }
                
                self._pdh_pdl_cache[cache_key] = pdh_pdl
                return pdh_pdl
                
            return {}
            
        except Exception as e:
            logger.error(f"PDH/PDL olish xatosi: {e}")
            return {}
            
    async def _find_order_blocks(self, symbol: str, candles: pd.DataFrame) -> List[OrderBlock]:
        """Order Blocklarni topish"""
        try:
            order_blocks = []
            
            for i in range(self.ob_lookback, len(candles)):
                current = candles.iloc[i]
                prev = candles.iloc[i-1]
                
                # Bullish Order Block (last down candle before up move)
                if prev['close'] < prev['open'] and current['close'] > current['open']:
                    if current['close'] > prev['high']:  # Strong move up
                        ob = OrderBlock(
                            high=prev['high'],
                            low=prev['low'],
                            timestamp=prev.name,
                            type="BULLISH",
                            volume=prev['volume'],
                            strength=self._calculate_ob_strength(prev, current)
                        )
                        order_blocks.append(ob)
                        
                # Bearish Order Block (last up candle before down move)
                elif prev['close'] > prev['open'] and current['close'] < current['open']:
                    if current['close'] < prev['low']:  # Strong move down
                        ob = OrderBlock(
                            high=prev['high'],
                            low=prev['low'],
                            timestamp=prev.name,
                            type="BEARISH",
                            volume=prev['volume'],
                            strength=self._calculate_ob_strength(prev, current)
                        )
                        order_blocks.append(ob)
                        
            # Keep only unmitigated and recent blocks
            valid_blocks = []
            current_price = candles.iloc[-1]['close']
            
            for ob in order_blocks[-20:]:  # Last 20 blocks
                if ob.type == "BULLISH" and current_price > ob.low:
                    if current_price < ob.high:  # Not fully mitigated
                        valid_blocks.append(ob)
                elif ob.type == "BEARISH" and current_price < ob.high:
                    if current_price > ob.low:  # Not fully mitigated
                        valid_blocks.append(ob)
                        
            self._order_blocks[symbol] = valid_blocks
            return valid_blocks
            
        except Exception as e:
            logger.error(f"Order Block topish xatosi: {e}")
            return []
            
    async def _find_fvg(self, symbol: str, candles: pd.DataFrame) -> List[FairValueGap]:
        """Fair Value Gap topish"""
        try:
            fvgs = []
            
            for i in range(2, len(candles)):
                candle1 = candles.iloc[i-2]
                candle2 = candles.iloc[i-1]
                candle3 = candles.iloc[i]
                
                # Bullish FVG
                if candle1['high'] < candle3['low']:
                    gap_size = (candle3['low'] - candle1['high']) / candle1['high']
                    if gap_size >= self.fvg_min_size:
                        fvg = FairValueGap(
                            high=candle3['low'],
                            low=candle1['high'],
                            timestamp=candle2.name,
                            type="BULLISH",
                            strength=gap_size
                        )
                        fvgs.append(fvg)
                        
                # Bearish FVG
                elif candle1['low'] > candle3['high']:
                    gap_size = (candle1['low'] - candle3['high']) / candle3['high']
                    if gap_size >= self.fvg_min_size:
                        fvg = FairValueGap(
                            high=candle1['low'],
                            low=candle3['high'],
                            timestamp=candle2.name,
                            type="BEARISH",
                            strength=gap_size
                        )
                        fvgs.append(fvg)
                        
            # Check if FVGs are filled
            current_price = candles.iloc[-1]['close']
            valid_fvgs = []
            
            for fvg in fvgs[-15:]:  # Last 15 FVGs
                if fvg.type == "BULLISH":
                    if current_price < fvg.high:
                        fvg.filled = current_price <= fvg.low
                        valid_fvgs.append(fvg)
                else:  # BEARISH
                    if current_price > fvg.low:
                        fvg.filled = current_price >= fvg.high
                        valid_fvgs.append(fvg)
                        
            self._fvg_list[symbol] = valid_fvgs
            return valid_fvgs
            
        except Exception as e:
            logger.error(f"FVG topish xatosi: {e}")
            return []
            
    async def _find_liquidity_pools(self, symbol: str, candles: pd.DataFrame) -> List[LiquidityPool]:
        """Likvidlik havzalarini topish"""
        try:
            liquidity_pools = []
            
            # Find equal highs/lows
            high_counts = candles['high'].value_counts()
            low_counts = candles['low'].value_counts()
            
            # Equal Highs (BSL)
            for price, count in high_counts.items():
                if count >= 2:  # At least 2 touches
                    pool = LiquidityPool(
                        price=price,
                        type=LiquidityType.EQH,
                        timestamp=candles[candles['high'] == price].index[-1],
                        volume=candles[candles['high'] == price]['volume'].sum()
                    )
                    liquidity_pools.append(pool)
                    
            # Equal Lows (SSL)
            for price, count in low_counts.items():
                if count >= 2:  # At least 2 touches
                    pool = LiquidityPool(
                        price=price,
                        type=LiquidityType.EQL,
                        timestamp=candles[candles['low'] == price].index[-1],
                        volume=candles[candles['low'] == price]['volume'].sum()
                    )
                    liquidity_pools.append(pool)
                    
            # Find swing highs/lows for BSL/SSL
            window = 5
            
            # Swing Highs (BSL)
            swing_highs = candles[candles['high'] == candles['high'].rolling(window*2+1, center=True).max()]
            for idx, row in swing_highs.iterrows():
                pool = LiquidityPool(
                    price=row['high'],
                    type=LiquidityType.BSL,
                    timestamp=idx,
                    volume=row['volume']
                )
                liquidity_pools.append(pool)
                
            # Swing Lows (SSL)
            swing_lows = candles[candles['low'] == candles['low'].rolling(window*2+1, center=True).min()]
            for idx, row in swing_lows.iterrows():
                pool = LiquidityPool(
                    price=row['low'],
                    type=LiquidityType.SSL,
                    timestamp=idx,
                    volume=row['volume']
                )
                liquidity_pools.append(pool)
                
            # Check if liquidity is swept
            current_price = candles.iloc[-1]['close']
            current_high = candles.iloc[-1]['high']
            current_low = candles.iloc[-1]['low']
            
            valid_pools = []
            for pool in liquidity_pools[-30:]:  # Last 30 pools
                if pool.type in [LiquidityType.BSL, LiquidityType.EQH]:
                    pool.swept = current_high > pool.price
                else:  # SSL, EQL
                    pool.swept = current_low < pool.price
                    
                # Keep unswept pools and recently swept
                if not pool.swept or (TimeUtils.now_uzb() - pool.timestamp).hours < 24:
                    valid_pools.append(pool)
                    
            self._liquidity_pools[symbol] = valid_pools
            return valid_pools
            
        except Exception as e:
            logger.error(f"Liquidity pools topish xatosi: {e}")
            return []
            
    def _calculate_ob_strength(self, prev_candle: pd.Series, current_candle: pd.Series) -> float:
        """Order Block kuchini hisoblash"""
        try:
            # Volume ratio
            volume_ratio = current_candle['volume'] / prev_candle['volume']
            
            # Price movement
            price_move = abs(current_candle['close'] - prev_candle['close']) / prev_candle['close']
            
            # Candle body size
            prev_body = abs(prev_candle['close'] - prev_candle['open']) / prev_candle['open']
            
            # Combined strength (0-1)
            strength = min(1.0, (volume_ratio * 0.3 + price_move * 50 + prev_body * 20) / 3)
            
            return strength
            
        except:
            return 0.5
            
    async def _generate_signal(self, symbol: str, structure: MarketStructure,
                             pdh_pdl: Dict[str, PriceLevel], order_blocks: List[OrderBlock],
                             fvgs: List[FairValueGap], liquidity: List[LiquidityPool],
                             candles: pd.DataFrame) -> Optional[ICTSignal]:
        """ICT signalini yaratish"""
        try:
            current_price = candles.iloc[-1]['close']
            reasoning = []
            entry_zones = []
            bias = "NEUTRAL"
            
            # Market structure bias
            if structure == MarketStructure.BULLISH:
                bias = "BULLISH"
                reasoning.append("📈 Bullish market structure (HH/HL)")
            elif structure == MarketStructure.BEARISH:
                bias = "BEARISH"
                reasoning.append("📉 Bearish market structure (LH/LL)")
                
            # Kill zone check
            is_kz, kz_info = timezone_handler.is_kill_zone_active()
            if is_kz:
                reasoning.append(f"⏰ {kz_info.name} faol")
                
            # Find entry zones based on bias
            if bias == "BULLISH":
                # Look for bullish order blocks
                for ob in order_blocks:
                    if ob.type == "BULLISH" and ob.strength > 0.6:
                        entry_zones.append({
                            "type": "Order Block",
                            "high": ob.high,
                            "low": ob.low,
                            "strength": ob.strength
                        })
                        reasoning.append(f"🟩 Bullish OB: {ob.low:.2f}-{ob.high:.2f}")
                        
                # Look for bullish FVGs
                for fvg in fvgs:
                    if fvg.type == "BULLISH" and not fvg.filled:
                        entry_zones.append({
                            "type": "FVG",
                            "high": fvg.high,
                            "low": fvg.low,
                            "strength": fvg.strength
                        })
                        reasoning.append(f"🔲 Bullish FVG: {fvg.low:.2f}-{fvg.high:.2f}")
                        
                # Check for SSL sweep
                for pool in liquidity:
                    if pool.type == LiquidityType.SSL and pool.swept:
                        reasoning.append(f"🎯 SSL swept at {pool.price:.2f}")
                        
            elif bias == "BEARISH":
                # Look for bearish order blocks
                for ob in order_blocks:
                    if ob.type == "BEARISH" and ob.strength > 0.6:
                        entry_zones.append({
                            "type": "Order Block",
                            "high": ob.high,
                            "low": ob.low,
                            "strength": ob.strength
                        })
                        reasoning.append(f"🟥 Bearish OB: {ob.low:.2f}-{ob.high:.2f}")
                        
                # Look for bearish FVGs
                for fvg in fvgs:
                    if fvg.type == "BEARISH" and not fvg.filled:
                        entry_zones.append({
                            "type": "FVG",
                            "high": fvg.high,
                            "low": fvg.low,
                            "strength": fvg.strength
                        })
                        reasoning.append(f"🔳 Bearish FVG: {fvg.low:.2f}-{fvg.high:.2f}")
                        
                # Check for BSL sweep
                for pool in liquidity:
                    if pool.type == LiquidityType.BSL and pool.swept:
                        reasoning.append(f"🎯 BSL swept at {pool.price:.2f}")
                        
            # No valid entry zones
            if not entry_zones:
                return None
                
            # Sort entry zones by strength
            entry_zones.sort(key=lambda x: x['strength'], reverse=True)
            
            # Calculate stop loss and take profit
            best_zone = entry_zones[0]
            
            if bias == "BULLISH":
                entry_price = (best_zone['high'] + best_zone['low']) / 2
                stop_loss = best_zone['low'] - (best_zone['high'] - best_zone['low']) * 0.5
                
                # Find next liquidity for TP
                tp_targets = []
                for pool in liquidity:
                    if pool.type == LiquidityType.BSL and pool.price > entry_price:
                        tp_targets.append(pool.price)
                        
                # Add structure based targets
                recent_high = candles['high'].rolling(20).max().iloc[-1]
                tp_targets.append(recent_high)
                
            else:  # BEARISH
                entry_price = (best_zone['high'] + best_zone['low']) / 2
                stop_loss = best_zone['high'] + (best_zone['high'] - best_zone['low']) * 0.5
                
                # Find next liquidity for TP
                tp_targets = []
                for pool in liquidity:
                    if pool.type == LiquidityType.SSL and pool.price < entry_price:
                        tp_targets.append(pool.price)
                        
                # Add structure based targets
                recent_low = candles['low'].rolling(20).min().iloc[-1]
                tp_targets.append(recent_low)
                
            # Sort take profits
            if bias == "BULLISH":
                tp_targets.sort()
            else:
                tp_targets.sort(reverse=True)
                
            # Calculate confidence
            confidence = self._calculate_confidence(structure, entry_zones, is_kz)
            
            # Generate signal
            signal = ICTSignal(
                symbol=symbol,
                structure=structure,
                bias=bias,
                entry_zones=entry_zones[:3],  # Top 3 zones
                stop_loss=stop_loss,
                take_profit=tp_targets[:3] if tp_targets else [entry_price * 1.02],
                confidence=confidence,
                reasoning=reasoning
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation xatosi: {e}")
            return None
            
    def _calculate_confidence(self, structure: MarketStructure, 
                            entry_zones: List[Dict], is_kill_zone: bool) -> float:
        """Signal ishonchini hisoblash"""
        confidence = 50.0
        
        # Market structure
        if structure in [MarketStructure.BULLISH, MarketStructure.BEARISH]:
            confidence += 20
        elif structure == MarketStructure.RANGING:
            confidence -= 10
            
        # Entry zones
        confidence += min(30, len(entry_zones) * 10)
        
        # Zone strength
        avg_strength = np.mean([z['strength'] for z in entry_zones])
        confidence += avg_strength * 20
        
        # Kill zone bonus
        if is_kill_zone:
            confidence += 10
            
        # Normalize
        confidence = max(0, min(100, confidence))
        
        return confidence
        
    async def _update_loop(self):
        """Doimiy yangilanish"""
        while True:
            try:
                # Update all symbols
                symbols = config_manager.trading.symbols
                
                for symbol in symbols:
                    await self.analyze(symbol)
                    await asyncio.sleep(1)  # Rate limiting
                    
                # Wait before next update
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"ICT update loop xatosi: {e}")
                await asyncio.sleep(60)
                
    def get_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Tahlil xulosasi"""
        return {
            "symbol": symbol,
            "structure": self._structure_cache.get(symbol, MarketStructure.RANGING).name,
            "order_blocks": len(self._order_blocks.get(symbol, [])),
            "fvgs": len(self._fvg_list.get(symbol, [])),
            "liquidity_pools": len(self._liquidity_pools.get(symbol, [])),
            "last_update": TimeUtils.now_uzb().isoformat()
        }

# Global instance
ict_analyzer = ICTAnalyzer()
