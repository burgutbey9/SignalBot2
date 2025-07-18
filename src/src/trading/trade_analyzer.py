"""
Trade Analyzer - Stop Loss/Take Profit sabab tahlili
Savdolar tahlili, performance tracking, stop/tp sabablari
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
from utils.helpers import TimeUtils, PerformanceMonitor
from utils.database import DatabaseManager, Trade, TradeAnalysis
from api.trading_apis import unified_client

logger = get_logger(__name__)

class TradeResult(Enum):
    """Savdo natijalari"""
    WIN = auto()
    LOSS = auto()
    BREAKEVEN = auto()
    ACTIVE = auto()
    CANCELLED = auto()

class ExitReason(Enum):
    """Chiqish sabablari"""
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()
    MANUAL_CLOSE = auto()
    TIME_STOP = auto()
    REVERSAL_SIGNAL = auto()
    RISK_MANAGEMENT = auto()

class StopLossReason(Enum):
    """Stop Loss sabablari"""
    MARKET_REVERSAL = auto()           # Bozor teskari aylandi
    FALSE_BREAKOUT = auto()            # Soxta breakout
    WHALE_MANIPULATION = auto()        # Whale manipulyatsiyasi
    NEWS_IMPACT = auto()               # Yangilik ta'siri
    TECHNICAL_BREAKDOWN = auto()       # Texnik breakdown
    LIQUIDITY_HUNT = auto()            # Likvidlik ovi
    VOLATILITY_SPIKE = auto()          # Volatillik portlashi
    CORRELATION_BREAKDOWN = auto()     # Korrelyatsiya buzilishi

class TakeProfitReason(Enum):
    """Take Profit sabablari"""
    TARGET_REACHED = auto()            # Maqsadga yetildi
    RESISTANCE_HIT = auto()            # Qarshilikka urdi
    OVERBOUGHT = auto()                # Haddan tashqari sotib olingan
    DIVERGENCE = auto()                # Divergence aniqlandi
    WHALE_DISTRIBUTION = auto()        # Whale distribution
    MOMENTUM_LOSS = auto()             # Momentum yo'qoldi
    TIME_TARGET = auto()               # Vaqt maqsadi
    PARTIAL_PROFIT = auto()            # Qisman foyda

@dataclass
class TradePerformance:
    """Savdo performansi"""
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_percent: float
    duration: timedelta
    result: TradeResult
    exit_reason: ExitReason
    detailed_reason: str
    market_conditions: Dict[str, Any]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class AnalysisReport:
    """Tahlil hisoboti"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    best_trade: Optional[TradePerformance]
    worst_trade: Optional[TradePerformance]
    stop_loss_reasons: Dict[StopLossReason, int]
    take_profit_reasons: Dict[TakeProfitReason, int]

class TradeAnalyzer:
    """Savdo tahlilchisi"""
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.performance_monitor = PerformanceMonitor()
        
        # Trade cache
        self._active_trades: Dict[str, TradePerformance] = {}
        self._completed_trades: List[TradePerformance] = []
        self._analysis_cache: Dict[str, AnalysisReport] = {}
        
        # Analysis parameters
        self.lookback_period = 30  # days
        self.min_trades_for_analysis = 10
        
    async def start(self):
        """Analyzer ishga tushirish"""
        logger.info("📊 Trade Analyzer ishga tushmoqda...")
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Load historical trades
        await self._load_historical_trades()
        
        # Start monitoring
        asyncio.create_task(self._monitor_active_trades())
        asyncio.create_task(self._generate_reports())
        
    async def analyze_trade_exit(self, trade: Dict[str, Any], exit_price: float, 
                               exit_type: str = "manual") -> TradePerformance:
        """Savdo chiqishini tahlil qilish"""
        try:
            # Calculate PnL
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            if trade['side'] == 'BUY':
                pnl = (exit_price - entry_price) * position_size
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl = (entry_price - exit_price) * position_size
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
            # Determine result
            if pnl > 0:
                result = TradeResult.WIN
            elif pnl < 0:
                result = TradeResult.LOSS
            else:
                result = TradeResult.BREAKEVEN
                
            # Analyze exit reason
            exit_reason, detailed_reason = await self._analyze_exit_reason(
                trade, exit_price, exit_type, result
            )
            
            # Get market conditions at exit
            market_conditions = await self._get_market_conditions(trade['symbol'])
            
            # Calculate duration
            duration = TimeUtils.now_uzb() - trade['entry_time']
            
            # Create performance record
            performance = TradePerformance(
                trade_id=trade['id'],
                symbol=trade['symbol'],
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                pnl=pnl,
                pnl_percent=pnl_percent,
                duration=duration,
                result=result,
                exit_reason=exit_reason,
                detailed_reason=detailed_reason,
                market_conditions=market_conditions
            )
            
            # Store analysis
            await self._store_trade_analysis(performance)
            
            # Update metrics
            self._completed_trades.append(performance)
            if trade['id'] in self._active_trades:
                del self._active_trades[trade['id']]
                
            logger.info(f"Trade {trade['id']} tahlil qilindi: {result.name} ({pnl_percent:+.2f}%)")
            
            return performance
            
        except Exception as e:
            logger.error(f"Trade exit analysis xatosi: {e}")
            return None
            
    async def _analyze_exit_reason(self, trade: Dict[str, Any], exit_price: float,
                                 exit_type: str, result: TradeResult) -> Tuple[ExitReason, str]:
        """Chiqish sababini tahlil qilish"""
        try:
            detailed_reason = ""
            
            # Check if hit stop loss
            if exit_type == "stop_loss" or (
                trade['side'] == 'BUY' and exit_price <= trade['stop_loss']
            ) or (
                trade['side'] == 'SELL' and exit_price >= trade['stop_loss']
            ):
                exit_reason = ExitReason.STOP_LOSS
                sl_reason = await self._analyze_stop_loss_reason(trade, exit_price)
                detailed_reason = self._get_stop_loss_explanation(sl_reason, trade)
                
            # Check if hit take profit
            elif exit_type == "take_profit" or self._check_tp_hit(trade, exit_price):
                exit_reason = ExitReason.TAKE_PROFIT
                tp_reason = await self._analyze_take_profit_reason(trade, exit_price)
                detailed_reason = self._get_take_profit_explanation(tp_reason, trade)
                
            # Check other exit types
            elif exit_type == "trailing_stop":
                exit_reason = ExitReason.TRAILING_STOP
                detailed_reason = "Trailing stop faollashdi - foyda himoyalandi"
                
            elif exit_type == "time_stop":
                exit_reason = ExitReason.TIME_STOP
                detailed_reason = f"Vaqt limiti - {trade.get('max_duration', 24)} soat"
                
            elif exit_type == "reversal_signal":
                exit_reason = ExitReason.REVERSAL_SIGNAL
                detailed_reason = "Teskari signal aniqlandi"
                
            elif exit_type == "risk_management":
                exit_reason = ExitReason.RISK_MANAGEMENT
                detailed_reason = "Risk management qoidalari - pozitsiya yopildi"
                
            else:
                exit_reason = ExitReason.MANUAL_CLOSE
                detailed_reason = "Manual yopildi"
                
            return exit_reason, detailed_reason
            
        except Exception as e:
            logger.error(f"Exit reason analysis xatosi: {e}")
            return ExitReason.MANUAL_CLOSE, "Tahlil xatosi"
            
    async def _analyze_stop_loss_reason(self, trade: Dict[str, Any], exit_price: float) -> StopLossReason:
        """Stop loss sababini aniqlash"""
        try:
            symbol = trade['symbol']
            
            # Get recent market data
            candles = await unified_client.get_candles(symbol, "5m", 100)
            if candles.empty:
                return StopLossReason.TECHNICAL_BREAKDOWN
                
            # Check for liquidity hunt
            if await self._check_liquidity_hunt(symbol, trade, candles):
                return StopLossReason.LIQUIDITY_HUNT
                
            # Check for whale manipulation
            whale_activity = await self._check_whale_activity(symbol)
            if whale_activity.get('manipulation_detected'):
                return StopLossReason.WHALE_MANIPULATION
                
            # Check for news impact
            recent_news = await self._check_recent_news(symbol)
            if recent_news.get('high_impact'):
                return StopLossReason.NEWS_IMPACT
                
            # Check for volatility spike
            volatility = self._calculate_volatility(candles)
            if volatility > candles['close'].std() * 2:
                return StopLossReason.VOLATILITY_SPIKE
                
            # Check for false breakout
            if await self._check_false_breakout(trade, candles):
                return StopLossReason.FALSE_BREAKOUT
                
            # Check correlation breakdown
            if await self._check_correlation_breakdown(symbol):
                return StopLossReason.CORRELATION_BREAKDOWN
                
            # Default to market reversal
            return StopLossReason.MARKET_REVERSAL
            
        except Exception as e:
            logger.error(f"Stop loss reason analysis xatosi: {e}")
            return StopLossReason.TECHNICAL_BREAKDOWN
            
    async def _analyze_take_profit_reason(self, trade: Dict[str, Any], exit_price: float) -> TakeProfitReason:
        """Take profit sababini aniqlash"""
        try:
            symbol = trade['symbol']
            
            # Get market data
            candles = await unified_client.get_candles(symbol, "15m", 50)
            if candles.empty:
                return TakeProfitReason.TARGET_REACHED
                
            # Check if exact TP hit
            for tp in trade.get('take_profits', []):
                if abs(exit_price - tp) / tp < 0.001:  # Within 0.1%
                    return TakeProfitReason.TARGET_REACHED
                    
            # Check for resistance
            if await self._check_resistance_hit(symbol, exit_price, candles):
                return TakeProfitReason.RESISTANCE_HIT
                
            # Check for overbought conditions
            rsi = self._calculate_rsi(candles['close'])
            if rsi > 70:
                return TakeProfitReason.OVERBOUGHT
                
            # Check for divergence
            if self._check_divergence(candles):
                return TakeProfitReason.DIVERGENCE
                
            # Check whale distribution
            whale_activity = await self._check_whale_activity(symbol)
            if whale_activity.get('distribution_detected'):
                return TakeProfitReason.WHALE_DISTRIBUTION
                
            # Check momentum
            momentum = self._calculate_momentum(candles)
            if momentum < 0:
                return TakeProfitReason.MOMENTUM_LOSS
                
            # Check if partial profit
            if exit_price < max(trade.get('take_profits', [exit_price])):
                return TakeProfitReason.PARTIAL_PROFIT
                
            return TakeProfitReason.TARGET_REACHED
            
        except Exception as e:
            logger.error(f"Take profit reason analysis xatosi: {e}")
            return TakeProfitReason.TARGET_REACHED
            
    def _check_tp_hit(self, trade: Dict[str, Any], exit_price: float) -> bool:
        """TP hit qilinganini tekshirish"""
        if trade['side'] == 'BUY':
            return any(exit_price >= tp for tp in trade.get('take_profits', []))
        else:  # SELL
            return any(exit_price <= tp for tp in trade.get('take_profits', []))
            
    async def _check_liquidity_hunt(self, symbol: str, trade: Dict[str, Any], 
                                   candles: pd.DataFrame) -> bool:
        """Likvidlik ovini tekshirish"""
        try:
            # Check if price swept stops and reversed
            stop_level = trade['stop_loss']
            
            if trade['side'] == 'BUY':
                # Check if price went below stop and recovered
                min_after_entry = candles['low'].min()
                current_price = candles['close'].iloc[-1]
                
                if min_after_entry < stop_level and current_price > trade['entry_price']:
                    return True
                    
            else:  # SELL
                # Check if price went above stop and recovered
                max_after_entry = candles['high'].max()
                current_price = candles['close'].iloc[-1]
                
                if max_after_entry > stop_level and current_price < trade['entry_price']:
                    return True
                    
            return False
            
        except:
            return False
            
    async def _check_whale_activity(self, symbol: str) -> Dict[str, Any]:
        """Whale faoliyatini tekshirish"""
        try:
            # Get whale data from SMT analyzer
            from analysis.smt_analyzer import smt_analyzer
            whale_data = await smt_analyzer._get_whale_activity(symbol)
            
            result = {
                'manipulation_detected': False,
                'distribution_detected': False
            }
            
            if whale_data:
                # Check for manipulation patterns
                if whale_data.get('sentiment') == 'BEARISH' and \
                   whale_data.get('distribution_count', 0) > 5:
                    result['manipulation_detected'] = True
                    
                # Check for distribution
                if whale_data.get('distribution_count', 0) > \
                   whale_data.get('accumulation_count', 0) * 2:
                    result['distribution_detected'] = True
                    
            return result
            
        except:
            return {'manipulation_detected': False, 'distribution_detected': False}
            
    async def _check_recent_news(self, symbol: str) -> Dict[str, Any]:
        """Yaqinda chiqqan yangiliklarni tekshirish"""
        try:
            # Get news from sentiment analyzer
            from analysis.sentiment import sentiment_analyzer
            news_data = await sentiment_analyzer._analyze_news(symbol)
            
            result = {'high_impact': False}
            
            if news_data:
                # Check for high impact news in last hour
                one_hour_ago = TimeUtils.now_uzb() - timedelta(hours=1)
                
                for news in news_data:
                    if news.published_at > one_hour_ago and \
                       news.impact.name in ['CRITICAL', 'HIGH']:
                        result['high_impact'] = True
                        break
                        
            return result
            
        except:
            return {'high_impact': False}
            
    def _calculate_volatility(self, candles: pd.DataFrame) -> float:
        """Volatillikni hisoblash"""
        try:
            returns = candles['close'].pct_change().dropna()
            return returns.std() * np.sqrt(len(returns))
        except:
            return 0
            
    async def _check_false_breakout(self, trade: Dict[str, Any], candles: pd.DataFrame) -> bool:
        """Soxta breakoutni tekshirish"""
        try:
            # Check if price broke key level and reversed
            entry_price = trade['entry_price']
            
            if trade['side'] == 'BUY':
                # Check if broke resistance and fell back
                high_after_entry = candles['high'].max()
                if high_after_entry > entry_price * 1.01:  # Broke 1% above
                    current_price = candles['close'].iloc[-1]
                    if current_price < entry_price:
                        return True
                        
            else:  # SELL
                # Check if broke support and bounced back
                low_after_entry = candles['low'].min()
                if low_after_entry < entry_price * 0.99:  # Broke 1% below
                    current_price = candles['close'].iloc[-1]
                    if current_price > entry_price:
                        return True
                        
            return False
            
        except:
            return False
            
    async def _check_correlation_breakdown(self, symbol: str) -> bool:
        """Korrelyatsiya buzilishini tekshirish"""
        try:
            # Check correlation with major pairs
            if "BTC" not in symbol:
                btc_candles = await unified_client.get_candles("BTCUSDT", "1h", 24)
                symbol_candles = await unified_client.get_candles(symbol, "1h", 24)
                
                if not btc_candles.empty and not symbol_candles.empty:
                    correlation = btc_candles['close'].corr(symbol_candles['close'])
                    
                    # If normally correlated but broke down
                    if abs(correlation) < 0.3:  # Low correlation
                        return True
                        
            return False
            
        except:
            return False
            
    async def _check_resistance_hit(self, symbol: str, price: float, candles: pd.DataFrame) -> bool:
        """Qarshilik darajasiga urilganini tekshirish"""
        try:
            # Find recent highs
            recent_highs = candles['high'].rolling(20).max()
            
            # Check if price is near any recent high
            for high in recent_highs.unique():
                if abs(price - high) / high < 0.005:  # Within 0.5%
                    return True
                    
            return False
            
        except:
            return False
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI hisoblash"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
            
        except:
            return 50
            
    def _check_divergence(self, candles: pd.DataFrame) -> bool:
        """Divergence tekshirish"""
        try:
            prices = candles['close']
            rsi = self._calculate_rsi(prices)
            
            # Price making new high but RSI not
            price_high = prices.iloc[-1] > prices.iloc[-20:].max()
            rsi_high = rsi > 70
            
            if price_high and not rsi_high:
                return True
                
            # Price making new low but RSI not
            price_low = prices.iloc[-1] < prices.iloc[-20:].min()
            rsi_low = rsi < 30
            
            if price_low and not rsi_low:
                return True
                
            return False
            
        except:
            return False
            
    def _calculate_momentum(self, candles: pd.DataFrame) -> float:
        """Momentum hisoblash"""
        try:
            # Simple momentum - rate of change
            current_price = candles['close'].iloc[-1]
            past_price = candles['close'].iloc[-10]
            
            momentum = (current_price - past_price) / past_price * 100
            return momentum
            
        except:
            return 0
            
    def _get_stop_loss_explanation(self, reason: StopLossReason, trade: Dict[str, Any]) -> str:
        """Stop loss sababini tushuntirish"""
        explanations = {
            StopLossReason.MARKET_REVERSAL: f"Bozor {trade['side']} signalga qarshi aylandi",
            StopLossReason.FALSE_BREAKOUT: "False breakout - narx key level buzib qaytdi",
            StopLossReason.WHALE_MANIPULATION: "Whale manipulation aniqlandi - katta sell pressure",
            StopLossReason.NEWS_IMPACT: "Yuqori ta'sirli yangilik - bozor sentimenti o'zgardi",
            StopLossReason.TECHNICAL_BREAKDOWN: "Texnik support/resistance buzildi",
            StopLossReason.LIQUIDITY_HUNT: "Likvidlik ovi - stop hunt aniqlandi",
            StopLossReason.VOLATILITY_SPIKE: "Kutilmagan volatillik portlashi",
            StopLossReason.CORRELATION_BREAKDOWN: "Asosiy juftliklar bilan korrelyatsiya buzildi"
        }
        
        return explanations.get(reason, "Noma'lum sabab")
        
    def _get_take_profit_explanation(self, reason: TakeProfitReason, trade: Dict[str, Any]) -> str:
        """Take profit sababini tushuntirish"""
        explanations = {
            TakeProfitReason.TARGET_REACHED: f"Belgilangan maqsad narxga yetildi",
            TakeProfitReason.RESISTANCE_HIT: "Kuchli qarshilik darajasiga yetildi",
            TakeProfitReason.OVERBOUGHT: "RSI > 70 - overbought holati",
            TakeProfitReason.DIVERGENCE: "Price/indicator divergence aniqlandi",
            TakeProfitReason.WHALE_DISTRIBUTION: "Whale distribution boshlandi",
            TakeProfitReason.MOMENTUM_LOSS: "Momentum yo'qoldi - trend kuchsizlandi",
            TakeProfitReason.TIME_TARGET: "Vaqt bo'yicha maqsadga yetildi",
            TakeProfitReason.PARTIAL_PROFIT: "Qisman foyda olindi"
        }
        
        return explanations.get(reason, "Maqsadga yetildi")
        
    async def _get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Bozor sharoitlarini olish"""
        try:
            conditions = {
                "timestamp": TimeUtils.now_uzb().isoformat(),
                "symbol": symbol
            }
            
            # Get current data
            ticker = await unified_client.get_ticker(symbol)
            if ticker:
                conditions["price"] = ticker.get("last_price", 0)
                conditions["volume_24h"] = ticker.get("volume", 0)
                conditions["price_change_24h"] = ticker.get("price_change_percent", 0)
                
            # Get market sentiment
            from analysis.sentiment import sentiment_analyzer
            sentiment = sentiment_analyzer.get_sentiment_summary(symbol)
            conditions["sentiment"] = sentiment.get("sentiment", "NEUTRAL")
            
            # Get volatility
            candles = await unified_client.get_candles(symbol, "1h", 24)
            if not candles.empty:
                conditions["volatility"] = candles['close'].std() / candles['close'].mean() * 100
                
            return conditions
            
        except Exception as e:
            logger.error(f"Market conditions olish xatosi: {e}")
            return {"error": str(e)}
            
    async def _store_trade_analysis(self, performance: TradePerformance):
        """Savdo tahlilini saqlash"""
        try:
            if self.db_manager:
                analysis = TradeAnalysis(
                    trade_id=performance.trade_id,
                    symbol=performance.symbol,
                    result=performance.result.name,
                    pnl=performance.pnl,
                    pnl_percent=performance.pnl_percent,
                    exit_reason=performance.exit_reason.name,
                    detailed_reason=performance.detailed_reason,
                    duration_minutes=int(performance.duration.total_seconds() / 60),
                    market_conditions=performance.market_conditions,
                    timestamp=performance.timestamp
                )
                
                await self.db_manager.save_trade_analysis(analysis)
                
        except Exception as e:
            logger.error(f"Trade analysis saqlash xatosi: {e}")
            
    async def _load_historical_trades(self):
        """Tarixiy savdolarni yuklash"""
        try:
            if self.db_manager:
                # Load last 30 days of trades
                start_date = TimeUtils.now_uzb() - timedelta(days=self.lookback_period)
                trades = await self.db_manager.get_trades_after(start_date)
                
                # Convert to performance records
                for trade in trades:
                    if trade.status == "CLOSED":
                        perf = TradePerformance(
                            trade_id=trade.id,
                            symbol=trade.symbol,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            position_size=trade.position_size,
                            pnl=trade.pnl,
                            pnl_percent=trade.pnl_percent,
                            duration=timedelta(minutes=trade.duration_minutes),
                            result=TradeResult[trade.result],
                            exit_reason=ExitReason[trade.exit_reason],
                            detailed_reason=trade.detailed_reason,
                            market_conditions=trade.market_conditions,
                            timestamp=trade.closed_at
                        )
                        self._completed_trades.append(perf)
                        
                logger.info(f"{len(self._completed_trades)} ta tarixiy savdo yuklandi")
                
        except Exception as e:
            logger.error(f"Historical trades yuklash xatosi: {e}")
            
    async def _monitor_active_trades(self):
        """Aktiv savdolarni monitoring qilish"""
        while True:
            try:
                # Monitor all active trades
                for trade_id, trade in self._active_trades.items():
                    # Check if should close
                    should_close, reason = await self._check_trade_exit(trade)
                    
                    if should_close:
                        # Get current price
                        ticker = await unified_client.get_ticker(trade.symbol)
                        if ticker:
                            exit_price = ticker.get("last_price", 0)
                            
                            # Analyze exit
                            await self.analyze_trade_exit(
                                trade.__dict__, exit_price, reason
                            )
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Active trades monitoring xatosi: {e}")
                await asyncio.sleep(30)
                
    async def _check_trade_exit(self, trade: TradePerformance) -> Tuple[bool, str]:
        """Savdo chiqish shartlarini tekshirish"""
        try:
            # Time stop check
            if trade.duration > timedelta(hours=24):
                return True, "time_stop"
                
            # Get current price
            ticker = await unified_client.get_ticker(trade.symbol)
            if not ticker:
                return False, ""
                
            current_price = ticker.get("last_price", 0)
            
            # Check stop loss
            if trade.position_size > 0:  # Long position
                if current_price <= trade.entry_price * 0.98:  # 2% stop
                    return True, "stop_loss"
            else:  # Short position
                if current_price >= trade.entry_price * 1.02:  # 2% stop
                    return True, "stop_loss"
                    
            # Check for reversal signal
            # This would integrate with signal generator
            
            return False, ""
            
        except:
            return False, ""
            
    async def _generate_reports(self):
        """Hisobot yaratish"""
        while True:
            try:
                # Generate overall report
                report = await self.generate_performance_report()
                
                # Cache report
                self._analysis_cache["overall"] = report
                
                # Generate per-symbol reports
                symbols = set(trade.symbol for trade in self._completed_trades)
                
                for symbol in symbols:
                    symbol_report = await self.generate_performance_report(symbol)
                    self._analysis_cache[symbol] = symbol_report
                    
                logger.info("Performance hisobotlari yangilandi")
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Report generation xatosi: {e}")
                await asyncio.sleep(3600)
                
    async def generate_performance_report(self, symbol: Optional[str] = None) -> AnalysisReport:
        """Performance hisoboti yaratish"""
        try:
            # Filter trades by symbol
            if symbol:
                trades = [t for t in self._completed_trades if t.symbol == symbol]
            else:
                trades = self._completed_trades
                
            if not trades:
                return self._empty_report()
                
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.result == TradeResult.WIN])
            losing_trades = len([t for t in trades if t.result == TradeResult.LOSS])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # PnL calculations
            wins = [t.pnl_percent for t in trades if t.result == TradeResult.WIN]
            losses = [abs(t.pnl_percent) for t in trades if t.result == TradeResult.LOSS]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = [t.pnl_percent for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if returns else 0
            
            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(trades)
            
            # Best and worst trades
            best_trade = max(trades, key=lambda t: t.pnl_percent) if trades else None
            worst_trade = min(trades, key=lambda t: t.pnl_percent) if trades else None
            
            # Stop loss reasons
            sl_reasons = {}
            for t in trades:
                if t.exit_reason == ExitReason.STOP_LOSS:
                    # Parse detailed reason to get StopLossReason
                    for reason in StopLossReason:
                        if reason.name in t.detailed_reason:
                            sl_reasons[reason] = sl_reasons.get(reason, 0) + 1
                            break
                            
            # Take profit reasons
            tp_reasons = {}
            for t in trades:
                if t.exit_reason == ExitReason.TAKE_PROFIT:
                    # Parse detailed reason to get TakeProfitReason
                    for reason in TakeProfitReason:
                        if reason.name in t.detailed_reason:
                            tp_reasons[reason] = tp_reasons.get(reason, 0) + 1
                            break
                            
            report = AnalysisReport(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                best_trade=best_trade,
                worst_trade=worst_trade,
                stop_loss_reasons=sl_reasons,
                take_profit_reasons=tp_reasons
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation xatosi: {e}")
            return self._empty_report()
            
    def _calculate_max_drawdown(self, trades: List[TradePerformance]) -> float:
        """Maksimal drawdown hisoblash"""
        if not trades:
            return 0
            
        cumulative_pnl = []
        running_total = 0
        
        for trade in sorted(trades, key=lambda t: t.timestamp):
            running_total += trade.pnl_percent
            cumulative_pnl.append(running_total)
            
        if not cumulative_pnl:
            return 0
            
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
            
        return max_dd
        
    def _empty_report(self) -> AnalysisReport:
        """Bo'sh hisobot"""
        return AnalysisReport(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown=0,
            best_trade=None,
            worst_trade=None,
            stop_loss_reasons={},
            take_profit_reasons={}
        )
        
    def get_trade_summary(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Savdo xulosasini olish"""
        # Check active trades
        if trade_id in self._active_trades:
            trade = self._active_trades[trade_id]
            return self._format_trade_summary(trade)
            
        # Check completed trades
        for trade in self._completed_trades:
            if trade.trade_id == trade_id:
                return self._format_trade_summary(trade)
                
        return None
        
    def _format_trade_summary(self, trade: TradePerformance) -> Dict[str, Any]:
        """Savdo xulosasini formatlash"""
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "position_size": trade.position_size,
            "pnl": f"{trade.pnl:+.2f}",
            "pnl_percent": f"{trade.pnl_percent:+.2f}%",
            "duration": str(trade.duration),
            "result": trade.result.name,
            "exit_reason": trade.exit_reason.name,
            "detailed_reason": trade.detailed_reason,
            "timestamp": trade.timestamp.isoformat()
        }
        
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Performance statistikalarini olish"""
        # Get cached report
        if symbol:
            report = self._analysis_cache.get(symbol)
        else:
            report = self._analysis_cache.get("overall")
            
        if not report:
            return {"error": "No analysis available"}
            
        return {
            "total_trades": report.total_trades,
            "win_rate": f"{report.win_rate:.1f}%",
            "avg_win": f"{report.avg_win:.2f}%",
            "avg_loss": f"{report.avg_loss:.2f}%",
            "profit_factor": f"{report.profit_factor:.2f}",
            "sharpe_ratio": f"{report.sharpe_ratio:.2f}",
            "max_drawdown": f"{report.max_drawdown:.2f}%",
            "stop_loss_reasons": {
                reason.name: count 
                for reason, count in report.stop_loss_reasons.items()
            },
            "take_profit_reasons": {
                reason.name: count
                for reason, count in report.take_profit_reasons.items()
            }
        }
        
    async def analyze_losing_streak(self) -> Dict[str, Any]:
        """Ketma-ket yo'qotishlarni tahlil qilish"""
        try:
            losing_streak = 0
            max_losing_streak = 0
            streak_trades = []
            current_streak_trades = []
            
            for trade in sorted(self._completed_trades, key=lambda t: t.timestamp):
                if trade.result == TradeResult.LOSS:
                    losing_streak += 1
                    current_streak_trades.append(trade)
                    
                    if losing_streak > max_losing_streak:
                        max_losing_streak = losing_streak
                        streak_trades = current_streak_trades.copy()
                else:
                    losing_streak = 0
                    current_streak_trades = []
                    
            # Analyze common factors in losing streaks
            common_factors = {}
            
            if streak_trades:
                # Most common exit reasons
                exit_reasons = [t.exit_reason for t in streak_trades]
                common_factors["most_common_exit"] = max(set(exit_reasons), key=exit_reasons.count)
                
                # Average loss
                common_factors["avg_loss"] = np.mean([t.pnl_percent for t in streak_trades])
                
                # Time analysis
                hours = [t.timestamp.hour for t in streak_trades]
                common_factors["common_hour"] = max(set(hours), key=hours.count)
                
            return {
                "max_losing_streak": max_losing_streak,
                "current_losing_streak": losing_streak,
                "common_factors": common_factors,
                "recommendation": self._get_streak_recommendation(max_losing_streak)
            }
            
        except Exception as e:
            logger.error(f"Losing streak analysis xatosi: {e}")
            return {"error": str(e)}
            
    def _get_streak_recommendation(self, streak: int) -> str:
        """Ketma-ket yo'qotishlar uchun tavsiya"""
        if streak >= 5:
            return "🚨 CRITICAL: 5+ ketma-ket loss. Risk kamaytirilsin yoki trading to'xtatilsin!"
        elif streak >= 3:
            return "⚠️ WARNING: 3+ ketma-ket loss. Strategiya qayta ko'rib chiqilsin."
        elif streak >= 2:
            return "📊 CAUTION: 2 ketma-ket loss. Bozor sharoitlarini qayta baholang."
        else:
            return "✅ Normal holat"

# Global instance
trade_analyzer = TradeAnalyzer()
