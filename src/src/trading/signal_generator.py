"""
Crypto Signal Generator
ICT, SMT, Order Flow va Sentiment tahlillarini birlashtirish
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import numpy as np
from decimal import Decimal

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils, PerformanceMonitor
from utils.database import DatabaseManager, TradingSignal as DBSignal
from core.risk_manager import risk_manager
from core.timezone_handler import timezone_handler
from analysis.ict_analyzer import ict_analyzer, ICTSignal
from analysis.smt_analyzer import smt_analyzer, SMTSignal
from analysis.order_flow import order_flow_analyzer, OrderFlowSignal
from analysis.sentiment import sentiment_analyzer, SentimentSignal

logger = get_logger(__name__)

class SignalType(Enum):
    """Signal turlari"""
    STRONG_BUY = auto()
    BUY = auto()
    NEUTRAL = auto()
    SELL = auto()
    STRONG_SELL = auto()

class SignalSource(Enum):
    """Signal manbalari"""
    ICT = auto()
    SMT = auto()
    ORDER_FLOW = auto()
    SENTIMENT = auto()
    COMBINED = auto()

@dataclass
class TradingSignal:
    """Trading signali"""
    symbol: str
    type: SignalType
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    risk_reward_ratio: float
    confidence: float
    sources: List[SignalSource]
    analysis: Dict[str, Any]
    reasoning: List[str]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)
    expires_at: Optional[datetime] = None

@dataclass
class SignalMetrics:
    """Signal metrikalari"""
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    best_source: Optional[SignalSource] = None

class SignalGenerator:
    """Signal yaratuvchi"""
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.performance_monitor = PerformanceMonitor()
        
        # Signal cache
        self._active_signals: Dict[str, TradingSignal] = {}
        self._signal_history: List[TradingSignal] = []
        self._metrics: SignalMetrics = SignalMetrics()
        
        # Weights for different analysis types
        self.analysis_weights = {
            SignalSource.ICT: 0.3,
            SignalSource.SMT: 0.3,
            SignalSource.ORDER_FLOW: 0.25,
            SignalSource.SENTIMENT: 0.15
        }
        
        # Signal thresholds
        self.min_confidence = 65  # Minimum confidence for signal
        self.strong_signal_threshold = 85  # Strong signal threshold
        
    async def start(self):
        """Generator ishga tushirish"""
        logger.info("🎯 Signal Generator ishga tushmoqda...")
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Start signal generation loop
        asyncio.create_task(self._generate_signals_loop())
        asyncio.create_task(self._cleanup_expired_signals())
        
    async def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Signal yaratish"""
        try:
            logger.info(f"📊 {symbol} uchun signal yaratilmoqda...")
            
            # Get all analyses
            analyses = await self._get_all_analyses(symbol)
            
            if not analyses:
                logger.warning(f"{symbol} uchun tahlil topilmadi")
                return None
                
            # Combine analyses
            combined_analysis = await self._combine_analyses(symbol, analyses)
            
            if not combined_analysis:
                return None
                
            # Generate trading signal
            signal = await self._create_trading_signal(symbol, combined_analysis)
            
            if signal:
                # Validate signal
                if await self._validate_signal(signal):
                    # Store signal
                    await self._store_signal(signal)
                    
                    # Update metrics
                    self._metrics.total_signals += 1
                    
                    logger.info(f"✅ {symbol} uchun signal yaratildi: {signal.action}")
                    return signal
                else:
                    logger.warning(f"❌ {symbol} signal validation failed")
                    
            return None
            
        except Exception as e:
            logger.error(f"Signal generation xatosi {symbol}: {e}")
            return None
            
    async def _get_all_analyses(self, symbol: str) -> Dict[SignalSource, Any]:
        """Barcha tahlillarni olish"""
        analyses = {}
        
        try:
            # Get ICT analysis
            ict_signal = await ict_analyzer.analyze(symbol)
            if ict_signal:
                analyses[SignalSource.ICT] = ict_signal
                
            # Get SMT analysis
            smt_signal = await smt_analyzer.analyze(symbol)
            if smt_signal:
                analyses[SignalSource.SMT] = smt_signal
                
            # Get Order Flow analysis
            flow_signal = await order_flow_analyzer.analyze(symbol)
            if flow_signal:
                analyses[SignalSource.ORDER_FLOW] = flow_signal
                
            # Get Sentiment analysis
            sentiment_signal = await sentiment_analyzer.analyze(symbol)
            if sentiment_signal:
                analyses[SignalSource.SENTIMENT] = sentiment_signal
                
        except Exception as e:
            logger.error(f"Analyses olish xatosi: {e}")
            
        return analyses
        
    async def _combine_analyses(self, symbol: str, analyses: Dict[SignalSource, Any]) -> Optional[Dict[str, Any]]:
        """Tahlillarni birlashtirish"""
        try:
            if not analyses:
                return None
                
            combined = {
                "symbol": symbol,
                "analyses": analyses,
                "scores": {},
                "action": "HOLD",
                "confidence": 0,
                "reasoning": []
            }
            
            # Calculate scores for each analysis
            total_score = 0
            total_weight = 0
            
            # ICT Analysis
            if SignalSource.ICT in analyses:
                ict = analyses[SignalSource.ICT]
                ict_score = self._calculate_ict_score(ict)
                combined["scores"]["ICT"] = ict_score
                total_score += ict_score * self.analysis_weights[SignalSource.ICT]
                total_weight += self.analysis_weights[SignalSource.ICT]
                
                # Add reasoning
                if ict.bias == "BULLISH":
                    combined["reasoning"].append(f"📈 ICT: {ict.structure.name} structure")
                elif ict.bias == "BEARISH":
                    combined["reasoning"].append(f"📉 ICT: {ict.structure.name} structure")
                    
            # SMT Analysis
            if SignalSource.SMT in analyses:
                smt = analyses[SignalSource.SMT]
                smt_score = self._calculate_smt_score(smt)
                combined["scores"]["SMT"] = smt_score
                total_score += smt_score * self.analysis_weights[SignalSource.SMT]
                total_weight += self.analysis_weights[SignalSource.SMT]
                
                # Add reasoning
                combined["reasoning"].append(f"🐋 SMT: {smt.phase.name} phase")
                if smt.on_chain_score > 70:
                    combined["reasoning"].append(f"📊 On-chain: Accumulation score {smt.on_chain_score:.0f}")
                    
            # Order Flow Analysis
            if SignalSource.ORDER_FLOW in analyses:
                flow = analyses[SignalSource.ORDER_FLOW]
                flow_score = self._calculate_flow_score(flow)
                combined["scores"]["ORDER_FLOW"] = flow_score
                total_score += flow_score * self.analysis_weights[SignalSource.ORDER_FLOW]
                total_weight += self.analysis_weights[SignalSource.ORDER_FLOW]
                
                # Add reasoning
                if flow.buy_pressure > 0.6:
                    combined["reasoning"].append(f"💹 Order Flow: Buy pressure {flow.buy_pressure:.1%}")
                elif flow.sell_pressure > 0.6:
                    combined["reasoning"].append(f"📉 Order Flow: Sell pressure {flow.sell_pressure:.1%}")
                    
            # Sentiment Analysis
            if SignalSource.SENTIMENT in analyses:
                sentiment = analyses[SignalSource.SENTIMENT]
                sentiment_score = self._calculate_sentiment_score(sentiment)
                combined["scores"]["SENTIMENT"] = sentiment_score
                total_score += sentiment_score * self.analysis_weights[SignalSource.SENTIMENT]
                total_weight += self.analysis_weights[SignalSource.SENTIMENT]
                
                # Add reasoning
                combined["reasoning"].append(f"🧠 Sentiment: {sentiment.sentiment.name}")
                if sentiment.market_mood.fear_greed_index > 75:
                    combined["reasoning"].append(f"🔥 Fear & Greed: {sentiment.market_mood.fear_greed_index} (Greed)")
                elif sentiment.market_mood.fear_greed_index < 25:
                    combined["reasoning"].append(f"😱 Fear & Greed: {sentiment.market_mood.fear_greed_index} (Fear)")
                    
            # Calculate overall score
            if total_weight > 0:
                overall_score = total_score / total_weight
                combined["overall_score"] = overall_score
                combined["confidence"] = self._calculate_confidence(analyses, overall_score)
                
                # Determine action
                if overall_score > 70:
                    combined["action"] = "BUY"
                elif overall_score < 30:
                    combined["action"] = "SELL"
                else:
                    combined["action"] = "HOLD"
                    
                return combined
                
            return None
            
        except Exception as e:
            logger.error(f"Analyses birlashtirish xatosi: {e}")
            return None
            
    def _calculate_ict_score(self, ict_signal: ICTSignal) -> float:
        """ICT skorini hisoblash"""
        score = 50.0
        
        # Market structure
        if ict_signal.bias == "BULLISH":
            score += 20
        elif ict_signal.bias == "BEARISH":
            score -= 20
            
        # Entry zones quality
        if len(ict_signal.entry_zones) >= 3:
            score += 15
        elif len(ict_signal.entry_zones) >= 2:
            score += 10
        elif len(ict_signal.entry_zones) >= 1:
            score += 5
            
        # Confidence adjustment
        score += (ict_signal.confidence - 50) * 0.3
        
        return max(0, min(100, score))
        
    def _calculate_smt_score(self, smt_signal: SMTSignal) -> float:
        """SMT skorini hisoblash"""
        score = 50.0
        
        # Phase analysis
        phase_scores = {
            "ACCUMULATION": 30,
            "MARKUP": 20,
            "DISTRIBUTION": -30,
            "MARKDOWN": -20,
            "REACCUMULATION": 10,
            "REDISTRIBUTION": -10
        }
        
        phase_name = smt_signal.phase.name
        score += phase_scores.get(phase_name, 0)
        
        # On-chain score
        score += (smt_signal.on_chain_score - 50) * 0.4
        
        # Action adjustment
        if smt_signal.action == "BUY":
            score += 10
        elif smt_signal.action == "SELL":
            score -= 10
            
        return max(0, min(100, score))
        
    def _calculate_flow_score(self, flow_signal: OrderFlowSignal) -> float:
        """Order flow skorini hisoblash"""
        score = 50.0
        
        # Flow type
        flow_type_scores = {
            "BUY_AGGRESSIVE": 30,
            "SELL_AGGRESSIVE": -30,
            "BUY_PASSIVE": 10,
            "SELL_PASSIVE": -10,
            "NEUTRAL": 0
        }
        
        flow_type_name = flow_signal.flow_type.name
        score += flow_type_scores.get(flow_type_name, 0)
        
        # Pressure analysis
        if flow_signal.buy_pressure > 0.7:
            score += 20
        elif flow_signal.sell_pressure > 0.7:
            score -= 20
            
        # DEX activity
        if flow_signal.dex_activity.get("total_liquidity", 0) > 5000000:  # $5M+
            score += 10
            
        # Event impacts
        for event in flow_signal.events:
            if event.get("impact") == "HIGH":
                if "BUY" in event.get("type", ""):
                    score += 5
                elif "SELL" in event.get("type", ""):
                    score -= 5
                    
        return max(0, min(100, score))
        
    def _calculate_sentiment_score(self, sentiment_signal: SentimentSignal) -> float:
        """Sentiment skorini hisoblash"""
        score = 50.0
        
        # Sentiment value
        sentiment_values = {
            "VERY_BULLISH": 40,
            "BULLISH": 20,
            "NEUTRAL": 0,
            "BEARISH": -20,
            "VERY_BEARISH": -40
        }
        
        sentiment_name = sentiment_signal.sentiment.name
        score += sentiment_values.get(sentiment_name, 0)
        
        # Market mood adjustment
        fear_greed = sentiment_signal.market_mood.fear_greed_index
        score += (fear_greed - 50) * 0.2
        
        # AI analysis
        ai_rec = sentiment_signal.ai_analysis.get("recommendation", "HOLD")
        if ai_rec == "BUY":
            score += 10
        elif ai_rec == "SELL":
            score -= 10
            
        return max(0, min(100, score))
        
    def _calculate_confidence(self, analyses: Dict[SignalSource, Any], overall_score: float) -> float:
        """Signal ishonchini hisoblash"""
        confidence = 50.0
        
        # Number of confirming analyses
        analysis_count = len(analyses)
        confidence += analysis_count * 5
        
        # Score extremity (stronger signals = higher confidence)
        score_distance = abs(overall_score - 50)
        confidence += score_distance * 0.5
        
        # Individual analysis confidences
        total_analysis_confidence = 0
        confidence_count = 0
        
        if SignalSource.ICT in analyses:
            total_analysis_confidence += analyses[SignalSource.ICT].confidence
            confidence_count += 1
            
        if SignalSource.SMT in analyses:
            total_analysis_confidence += analyses[SignalSource.SMT].strength
            confidence_count += 1
            
        if SignalSource.ORDER_FLOW in analyses:
            total_analysis_confidence += analyses[SignalSource.ORDER_FLOW].strength
            confidence_count += 1
            
        if SignalSource.SENTIMENT in analyses:
            total_analysis_confidence += analyses[SignalSource.SENTIMENT].confidence
            confidence_count += 1
            
        if confidence_count > 0:
            avg_analysis_confidence = total_analysis_confidence / confidence_count
            confidence = (confidence + avg_analysis_confidence) / 2
            
        # Kill zone bonus
        is_kz, _ = timezone_handler.is_kill_zone_active()
        if is_kz:
            confidence += 5
            
        return max(0, min(100, confidence))
        
    async def _create_trading_signal(self, symbol: str, combined_analysis: Dict[str, Any]) -> Optional[TradingSignal]:
        """Trading signalini yaratish"""
        try:
            # Check minimum confidence
            if combined_analysis["confidence"] < self.min_confidence:
                logger.info(f"{symbol} signal confidence past: {combined_analysis['confidence']:.1f}%")
                return None
                
            # Skip if action is HOLD
            if combined_analysis["action"] == "HOLD":
                return None
                
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
                
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            
            if combined_analysis["action"] == "BUY":
                # For BUY signals
                stop_loss = await self._calculate_stop_loss(symbol, "BUY", entry_price, combined_analysis)
                take_profits = await self._calculate_take_profits(symbol, "BUY", entry_price, stop_loss, combined_analysis)
                
            else:  # SELL
                # For SELL signals
                stop_loss = await self._calculate_stop_loss(symbol, "SELL", entry_price, combined_analysis)
                take_profits = await self._calculate_take_profits(symbol, "SELL", entry_price, stop_loss, combined_analysis)
                
            # Calculate position size
            account_balance = await self._get_account_balance()
            position_info = risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, account_balance
            )
            
            if position_info["size"] == 0:
                logger.warning(f"{symbol} uchun position size 0")
                return None
                
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else risk * 2
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Determine signal type
            if combined_analysis["overall_score"] > self.strong_signal_threshold:
                signal_type = SignalType.STRONG_BUY if combined_analysis["action"] == "BUY" else SignalType.STRONG_SELL
            else:
                signal_type = SignalType.BUY if combined_analysis["action"] == "BUY" else SignalType.SELL
                
            # Get active sources
            sources = list(combined_analysis["analyses"].keys())
            
            # Set expiration (signals valid for 1 hour)
            expires_at = TimeUtils.now_uzb() + timedelta(hours=1)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                type=signal_type,
                action=combined_analysis["action"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profits,
                position_size=position_info["size"],
                risk_reward_ratio=risk_reward_ratio,
                confidence=combined_analysis["confidence"],
                sources=sources,
                analysis=combined_analysis,
                reasoning=combined_analysis["reasoning"],
                expires_at=expires_at
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Trading signal yaratish xatosi: {e}")
            return None
            
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Hozirgi narxni olish"""
        try:
            from api.trading_apis import unified_client
            ticker = await unified_client.get_ticker(symbol)
            return ticker.get("last_price", 0) if ticker else None
        except Exception as e:
            logger.error(f"Current price olish xatosi: {e}")
            return None
            
    async def _get_account_balance(self) -> float:
        """Account balansini olish"""
        try:
            from api.trading_apis import unified_client
            balance = await unified_client.get_balance("USDT")
            return balance.get("free", 10000)  # Default 10k for testing
        except Exception as e:
            logger.error(f"Account balance olish xatosi: {e}")
            return 10000  # Default balance
            
    async def _calculate_stop_loss(self, symbol: str, side: str, entry_price: float, 
                                  analysis: Dict[str, Any]) -> float:
        """Stop loss hisoblash"""
        try:
            # Get ATR for dynamic stop
            from api.trading_apis import unified_client
            candles = await unified_client.get_candles(symbol, "1h", 20)
            
            if not candles.empty:
                # Calculate ATR
                high = candles['high'].values
                low = candles['low'].values
                close = candles['close'].values
                
                tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
                atr = np.mean(tr[-14:])  # 14-period ATR
                
                # ATR-based stop
                atr_multiplier = 1.5 if analysis["confidence"] > 80 else 2.0
                stop_distance = atr * atr_multiplier
            else:
                # Default 1.5% stop
                stop_distance = entry_price * 0.015
                
            # Check for support/resistance from ICT
            if SignalSource.ICT in analysis["analyses"]:
                ict = analysis["analyses"][SignalSource.ICT]
                
                if side == "BUY" and ict.entry_zones:
                    # Use lowest entry zone as stop reference
                    lowest_zone = min(ict.entry_zones, key=lambda x: x["low"])
                    potential_stop = lowest_zone["low"] - stop_distance * 0.2
                    if potential_stop < entry_price:
                        stop_distance = entry_price - potential_stop
                        
            # Calculate final stop loss
            if side == "BUY":
                stop_loss = entry_price - stop_distance
            else:  # SELL
                stop_loss = entry_price + stop_distance
                
            return round(stop_loss, 2)
            
        except Exception as e:
            logger.error(f"Stop loss calculation xatosi: {e}")
            # Default stop loss
            if side == "BUY":
                return round(entry_price * 0.985, 2)  # 1.5% stop
            else:
                return round(entry_price * 1.015, 2)
                
    async def _calculate_take_profits(self, symbol: str, side: str, entry_price: float,
                                    stop_loss: float, analysis: Dict[str, Any]) -> List[float]:
        """Take profit hisoblash"""
        try:
            risk = abs(entry_price - stop_loss)
            take_profits = []
            
            # Multiple take profit levels
            tp_ratios = [1.5, 2.0, 3.0]  # Risk/Reward ratios
            
            # Adjust ratios based on confidence
            if analysis["confidence"] > 85:
                tp_ratios = [2.0, 3.0, 4.0]  # Higher targets for strong signals
            elif analysis["confidence"] < 70:
                tp_ratios = [1.0, 1.5, 2.0]  # Conservative targets
                
            # Check for resistance/support from analyses
            if SignalSource.ICT in analysis["analyses"]:
                ict = analysis["analyses"][SignalSource.ICT]
                if ict.take_profit:
                    # Use ICT targets if available
                    take_profits.extend(ict.take_profit[:2])
                    
            # Calculate standard TP levels
            for ratio in tp_ratios:
                if side == "BUY":
                    tp = entry_price + (risk * ratio)
                else:  # SELL
                    tp = entry_price - (risk * ratio)
                    
                if tp not in take_profits:
                    take_profits.append(round(tp, 2))
                    
            # Sort take profits
            if side == "BUY":
                take_profits.sort()
            else:
                take_profits.sort(reverse=True)
                
            return take_profits[:3]  # Return top 3 TP levels
            
        except Exception as e:
            logger.error(f"Take profit calculation xatosi: {e}")
            # Default take profits
            risk = abs(entry_price - stop_loss)
            if side == "BUY":
                return [
                    round(entry_price + risk * 1.5, 2),
                    round(entry_price + risk * 2.0, 2),
                    round(entry_price + risk * 3.0, 2)
                ]
            else:
                return [
                    round(entry_price - risk * 1.5, 2),
                    round(entry_price - risk * 2.0, 2),
                    round(entry_price - risk * 3.0, 2)
                ]
                
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Signalni tekshirish"""
        try:
            # Check risk/reward ratio
            if signal.risk_reward_ratio < 1.0:
                logger.warning(f"Risk/Reward ratio past: {signal.risk_reward_ratio:.2f}")
                return False
                
            # Check position size
            if signal.position_size <= 0:
                logger.warning("Position size 0 yoki manfiy")
                return False
                
            # Check if similar signal exists
            if signal.symbol in self._active_signals:
                existing = self._active_signals[signal.symbol]
                if existing.action == signal.action:
                    time_diff = (signal.timestamp - existing.timestamp).seconds
                    if time_diff < 3600:  # Less than 1 hour
                        logger.info("Similar signal already exists")
                        return False
                        
            # Check market hours
            if not timezone_handler.is_trading_hours(signal.symbol):
                logger.warning("Trading soatlari emas")
                # Allow crypto 24/7
                if "USDT" not in signal.symbol:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Signal validation xatosi: {e}")
            return False
            
    async def _store_signal(self, signal: TradingSignal):
        """Signalni saqlash"""
        try:
            # Store in active signals
            self._active_signals[signal.symbol] = signal
            
            # Add to history
            self._signal_history.append(signal)
            if len(self._signal_history) > 1000:  # Keep last 1000 signals
                self._signal_history = self._signal_history[-1000:]
                
            # Save to database
            if self.db_manager:
                db_signal = DBSignal(
                    symbol=signal.symbol,
                    type=signal.type.name,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit_1=signal.take_profit[0] if signal.take_profit else 0,
                    take_profit_2=signal.take_profit[1] if len(signal.take_profit) > 1 else 0,
                    take_profit_3=signal.take_profit[2] if len(signal.take_profit) > 2 else 0,
                    confidence=signal.confidence,
                    timestamp=signal.timestamp
                )
                
                await self.db_manager.save_signal(db_signal)
                
        except Exception as e:
            logger.error(f"Signal saqlash xatosi: {e}")
            
    async def _generate_signals_loop(self):
        """Doimiy signal yaratish"""
        while True:
            try:
                # Get active symbols
                symbols = config_manager.trading.symbols
                
                for symbol in symbols:
                    # Generate signal
                    signal = await self.generate_signal(symbol)
                    
                    if signal:
                        # Notify bot manager
                        from core.bot_manager import bot_manager
                        await bot_manager.process_signal(signal.__dict__)
                        
                    # Rate limiting
                    await asyncio.sleep(5)
                    
                # Wait before next cycle
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Signal generation loop xatosi: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_expired_signals(self):
        """Muddati o'tgan signallarni tozalash"""
        while True:
            try:
                now = TimeUtils.now_uzb()
                
                # Clean expired signals
                expired_symbols = []
                for symbol, signal in self._active_signals.items():
                    if signal.expires_at and signal.expires_at < now:
                        expired_symbols.append(symbol)
                        
                for symbol in expired_symbols:
                    del self._active_signals[symbol]
                    logger.info(f"Expired signal removed: {symbol}")
                    
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Signal cleanup xatosi: {e}")
                await asyncio.sleep(300)
                
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Aktiv signallarni olish"""
        return [
            {
                "symbol": signal.symbol,
                "type": signal.type.name,
                "action": signal.action,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "expires_at": signal.expires_at.isoformat() if signal.expires_at else None
            }
            for signal in self._active_signals.values()
        ]
        
    def get_signal_metrics(self) -> Dict[str, Any]:
        """Signal metrikalarini olish"""
        return {
            "total_signals": self._metrics.total_signals,
            "successful_signals": self._metrics.successful_signals,
            "failed_signals": self._metrics.failed_signals,
            "win_rate": self._metrics.win_rate,
            "avg_profit": self._metrics.avg_profit,
            "avg_loss": self._metrics.avg_loss,
            "profit_factor": self._metrics.profit_factor,
            "best_source": self._metrics.best_source.name if self._metrics.best_source else None,
            "active_signals": len(self._active_signals)
        }

# Global instance
signal_generator = SignalGenerator()
