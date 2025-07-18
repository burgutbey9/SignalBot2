"""
Risk Management and Stop Loss System
Risk boshqaruv, stop loss, daily limits
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils, CryptoUtils
from utils.database import DatabaseManager, TradeRecord, RiskMetrics

logger = get_logger(__name__)

class RiskLevel(Enum):
    """Risk darajasi"""
    MINIMAL = 0.25
    LOW = 0.5
    MEDIUM = 0.75
    HIGH = 1.0
    EXTREME = 1.5

class RiskEvent(Enum):
    """Risk hodisalari"""
    STOP_LOSS_HIT = auto()
    DAILY_LOSS_LIMIT = auto()
    POSITION_LIMIT = auto()
    VOLATILITY_SPIKE = auto()
    NEWS_EVENT = auto()
    CORRELATION_RISK = auto()
    LIQUIDITY_RISK = auto()

@dataclass
class PositionRisk:
    """Pozitsiya riski"""
    pair: str
    entry_price: float
    current_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    potential_loss: float
    potential_profit: float
    risk_reward_ratio: float
    time_in_position: float  # hours
    
    @property
    def pnl_percentage(self) -> float:
        """Current PnL percentage"""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
        
    @property
    def distance_to_stop(self) -> float:
        """Stop lossgacha masofa (%)"""
        return abs((self.current_price - self.stop_loss) / self.current_price) * 100

@dataclass
class RiskAnalysis:
    """Risk tahlili"""
    total_risk: float
    risk_level: RiskLevel
    open_positions: int
    daily_loss: float
    volatility_score: float
    correlation_risk: float
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

class RiskManager:
    """Risk management system"""
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.current_positions: Dict[str, PositionRisk] = {}
        self.daily_losses: Dict[str, float] = {}  # date -> loss
        self.risk_events: List[Tuple[RiskEvent, datetime]] = []
        self._config = config_manager.trading
        self._dynamic_risk_enabled = True
        self._current_risk_percentage = self._config.risk_percentage
        self._last_risk_update = TimeUtils.now_uzb()
        
    async def initialize(self):
        """Risk manager sozlash"""
        self.db_manager = DatabaseManager()
        await self.db_manager.initialize()
        await self._load_risk_metrics()
        logger.info("✅ Risk Manager initialized")
        
    async def check_pre_trade_risk(self, signal_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Savdo oldidan risk tekshirish"""
        try:
            # 1. Daily loss limit check
            if self._check_daily_loss_limit():
                return False, "Kunlik zarar limiti oshib ketdi"
                
            # 2. Position limit check
            if len(self.current_positions) >= self._config.max_positions:
                return False, f"Maksimal pozitsiya soni ({self._config.max_positions}) ga yetdi"
                
            # 3. Correlation risk check
            correlation_risk = await self._check_correlation_risk(signal_data["pair"])
            if correlation_risk > 0.8:
                return False, "Juda yuqori korrelyatsiya riski"
                
            # 4. Volatility check
            volatility = await self._check_volatility(signal_data["pair"])
            if volatility > 5.0:  # 5% dan yuqori
                return False, "Bozor juda volatil"
                
            # 5. News event check
            if await self._check_news_events():
                return False, "Muhim yangilik kutilmoqda"
                
            # 6. Position size validation
            position_size = self._calculate_position_size(
                signal_data["price"],
                signal_data["stop_loss"],
                signal_data.get("balance", 10000)
            )
            
            if position_size <= 0:
                return False, "Pozitsiya hajmi juda kichik"
                
            # Update signal with calculated values
            signal_data["position_size"] = position_size
            signal_data["risk_amount"] = position_size * (signal_data["sl_percentage"] / 100)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Pre-trade risk check xatosi: {e}")
            return False, f"Risk tekshiruv xatosi: {str(e)}"
            
    def _calculate_position_size(self, entry_price: float, stop_loss: float, balance: float) -> float:
        """Pozitsiya hajmini hisoblash"""
        risk_amount = balance * (self._current_risk_percentage / 100)
        stop_loss_distance = abs(entry_price - stop_loss) / entry_price
        
        if stop_loss_distance == 0:
            return 0
            
        position_size = risk_amount / stop_loss_distance
        
        # Max position size check (leverage consideration)
        max_position = balance * 0.95  # 95% of balance max
        
        return min(position_size, max_position)
        
    def _check_daily_loss_limit(self) -> bool:
        """Kunlik zarar limitini tekshirish"""
        today = str(TimeUtils.now_uzb().date())
        daily_loss = self.daily_losses.get(today, 0.0)
        
        return abs(daily_loss) >= self._config.max_daily_loss
        
    async def _check_correlation_risk(self, pair: str) -> float:
        """Korrelyatsiya riskini tekshirish"""
        if not self.current_positions:
            return 0.0
            
        # Simplified correlation check
        base_currency = pair[:3]
        correlated_positions = sum(1 for p in self.current_positions.values() 
                                 if p.pair.startswith(base_currency))
                                 
        return correlated_positions / len(self.current_positions)
        
    async def _check_volatility(self, pair: str) -> float:
        """Volatillikni tekshirish"""
        # Get recent price data from database
        if not self.db_manager:
            return 0.0
            
        recent_trades = await self.db_manager.get_recent_trades(pair, hours=1)
        if len(recent_trades) < 2:
            return 0.0
            
        prices = [t.price for t in recent_trades]
        
        # Calculate volatility
        if len(prices) > 1:
            mean_price = statistics.mean(prices)
            std_dev = statistics.stdev(prices)
            volatility = (std_dev / mean_price) * 100
            return volatility
            
        return 0.0
        
    async def _check_news_events(self) -> bool:
        """Muhim yangilik borligini tekshirish"""
        # This would integrate with news API
        # For now, return False
        return False
        
    async def add_position(self, position_data: Dict[str, Any]) -> bool:
        """Yangi pozitsiya qo'shish"""
        try:
            position = PositionRisk(
                pair=position_data["pair"],
                entry_price=position_data["entry_price"],
                current_price=position_data["entry_price"],
                position_size=position_data["position_size"],
                stop_loss=position_data["stop_loss"],
                take_profit=position_data["take_profit"],
                risk_amount=position_data["risk_amount"],
                potential_loss=position_data["risk_amount"],
                potential_profit=position_data.get("potential_profit", 0),
                risk_reward_ratio=position_data.get("risk_reward_ratio", 0),
                time_in_position=0
            )
            
            self.current_positions[position_data["id"]] = position
            
            # Adjust dynamic risk
            await self._adjust_dynamic_risk()
            
            logger.info(f"✅ Pozitsiya qo'shildi: {position.pair} @ {position.entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Pozitsiya qo'shishda xato: {e}")
            return False
            
    async def update_position(self, position_id: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Pozitsiya yangilash"""
        if position_id not in self.current_positions:
            return None
            
        position = self.current_positions[position_id]
        position.current_price = current_price
        position.time_in_position = (TimeUtils.now_uzb() - position.timestamp).total_seconds() / 3600
        
        # Check stop loss
        if current_price <= position.stop_loss:
            return await self._handle_stop_loss(position_id)
            
        # Check take profit
        if current_price >= position.take_profit:
            return await self._handle_take_profit(position_id)
            
        # Check trailing stop
        if self._config.trailing_stop:
            await self._update_trailing_stop(position)
            
        return {
            "action": "HOLD",
            "pnl_percentage": position.pnl_percentage,
            "distance_to_stop": position.distance_to_stop
        }
        
    async def _handle_stop_loss(self, position_id: str) -> Dict[str, Any]:
        """Stop loss bajarish"""
        position = self.current_positions[position_id]
        
        # Calculate loss
        loss_amount = position.risk_amount
        loss_percentage = ((position.stop_loss - position.entry_price) / position.entry_price) * 100
        
        # Update daily loss
        today = str(TimeUtils.now_uzb().date())
        self.daily_losses[today] = self.daily_losses.get(today, 0) + loss_percentage
        
        # Record risk event
        self.risk_events.append((RiskEvent.STOP_LOSS_HIT, TimeUtils.now_uzb()))
        
        # Remove position
        del self.current_positions[position_id]
        
        # Save to database
        if self.db_manager:
            await self.db_manager.save_trade_result({
                "position_id": position_id,
                "pair": position.pair,
                "result": "STOP_LOSS",
                "pnl": loss_amount,
                "pnl_percentage": loss_percentage
            })
            
        # Adjust risk after stop loss
        await self._adjust_risk_after_stop()
        
        logger.warning(f"🛑 Stop loss: {position.pair} @ {position.stop_loss} ({loss_percentage:.2f}%)")
        
        return {
            "action": "STOP_LOSS",
            "price": position.stop_loss,
            "pnl": loss_amount,
            "pnl_percentage": loss_percentage
        }
        
    async def _handle_take_profit(self, position_id: str) -> Dict[str, Any]:
        """Take profit bajarish"""
        position = self.current_positions[position_id]
        
        # Calculate profit
        profit_percentage = ((position.take_profit - position.entry_price) / position.entry_price) * 100
        profit_amount = position.position_size * (profit_percentage / 100)
        
        # Update daily PnL
        today = str(TimeUtils.now_uzb().date())
        self.daily_losses[today] = self.daily_losses.get(today, 0) + profit_percentage
        
        # Remove position
        del self.current_positions[position_id]
        
        # Save to database
        if self.db_manager:
            await self.db_manager.save_trade_result({
                "position_id": position_id,
                "pair": position.pair,
                "result": "TAKE_PROFIT",
                "pnl": profit_amount,
                "pnl_percentage": profit_percentage
            })
            
        logger.info(f"💰 Take profit: {position.pair} @ {position.take_profit} (+{profit_percentage:.2f}%)")
        
        return {
            "action": "TAKE_PROFIT",
            "price": position.take_profit,
            "pnl": profit_amount,
            "pnl_percentage": profit_percentage
        }
        
    async def _update_trailing_stop(self, position: PositionRisk):
        """Trailing stop yangilash"""
        if position.pnl_percentage <= 0:
            return  # Only trail in profit
            
        # Calculate new stop based on current price
        trail_distance = position.entry_price * (self._config.stop_loss_percentage / 100)
        new_stop = position.current_price - trail_distance
        
        # Only move stop up, never down
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
            logger.info(f"📈 Trailing stop updated: {position.pair} -> {new_stop:.2f}")
            
    async def _adjust_dynamic_risk(self):
        """Dinamik risk sozlash"""
        if not self._dynamic_risk_enabled:
            return
            
        # Get recent performance
        recent_trades = await self._get_recent_performance()
        
        if not recent_trades:
            return
            
        win_rate = recent_trades["win_rate"]
        avg_loss = recent_trades["avg_loss"]
        
        # Adjust risk based on performance
        if win_rate > 70 and avg_loss < 1.0:
            # Increase risk slightly
            self._current_risk_percentage = min(self._current_risk_percentage + 0.1, 1.0)
        elif win_rate < 40 or avg_loss > 2.0:
            # Decrease risk
            self._current_risk_percentage = max(self._current_risk_percentage - 0.1, 0.5)
            
        logger.info(f"🎯 Dynamic risk adjusted to {self._current_risk_percentage}%")
        
    async def _adjust_risk_after_stop(self):
        """Stop loss dan keyin riskni sozlash"""
        # Count recent stops
        recent_stops = sum(1 for event, time in self.risk_events 
                          if event == RiskEvent.STOP_LOSS_HIT and 
                          (TimeUtils.now_uzb() - time).total_seconds() < 86400)
                          
        if recent_stops >= 2:
            # Reduce risk after multiple stops
            self._current_risk_percentage = max(self._current_risk_percentage * 0.75, 0.5)
            logger.warning(f"⚠️ Risk reduced after {recent_stops} stops: {self._current_risk_percentage}%")
            
    async def _get_recent_performance(self) -> Dict[str, Any]:
        """Oxirgi natijalarni olish"""
        if not self.db_manager:
            return {}
            
        trades = await self.db_manager.get_recent_trades(hours=24)
        
        if not trades:
            return {}
            
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        total = len(trades)
        
        avg_loss = abs(sum(t.pnl for t in trades if t.pnl < 0) / max(losses, 1))
        
        return {
            "win_rate": (wins / total) * 100 if total > 0 else 0,
            "avg_loss": avg_loss,
            "total_trades": total
        }
        
    async def analyze_portfolio_risk(self) -> RiskAnalysis:
        """Portfolio riskini tahlil qilish"""
        try:
            # Calculate total risk exposure
            total_risk = sum(p.risk_amount for p in self.current_positions.values())
            
            # Get account balance (mock for now)
            balance = 10000  # This should come from exchange API
            risk_percentage = (total_risk / balance) * 100
            
            # Determine risk level
            if risk_percentage < 2:
                risk_level = RiskLevel.LOW
            elif risk_percentage < 5:
                risk_level = RiskLevel.MEDIUM
            elif risk_percentage < 10:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.EXTREME
                
            # Calculate other metrics
            today = str(TimeUtils.now_uzb().date())
            daily_loss = abs(self.daily_losses.get(today, 0))
            
            # Volatility score (average of all positions)
            volatility_scores = []
            for position in self.current_positions.values():
                vol = await self._check_volatility(position.pair)
                volatility_scores.append(vol)
                
            avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0
            
            # Correlation risk
            correlation_risk = await self._calculate_portfolio_correlation()
            
            # Generate recommendations
            recommendations = []
            warnings = []
            
            if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
                warnings.append("⚠️ Portfolio riski juda yuqori!")
                recommendations.append("Risk darajasini kamaytiring")
                
            if daily_loss > self._config.max_daily_loss * 0.8:
                warnings.append("📉 Kunlik zarar limitiga yaqinlashmoqda")
                recommendations.append("Yangi pozitsiya ochishni to'xtating")
                
            if avg_volatility > 3.0:
                warnings.append("🌊 Bozor juda volatil")
                recommendations.append("Pozitsiya hajmlarini kamaytiring")
                
            if correlation_risk > 0.7:
                warnings.append("🔗 Yuqori korrelyatsiya riski")
                recommendations.append("Portfolio diversifikatsiyasini oshiring")
                
            return RiskAnalysis(
                total_risk=total_risk,
                risk_level=risk_level,
                open_positions=len(self.current_positions),
                daily_loss=daily_loss,
                volatility_score=avg_volatility,
                correlation_risk=correlation_risk,
                recommendations=recommendations,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis xatosi: {e}")
            return RiskAnalysis(
                total_risk=0,
                risk_level=RiskLevel.LOW,
                open_positions=0,
                daily_loss=0,
                volatility_score=0,
                correlation_risk=0,
                warnings=[f"Risk tahlil xatosi: {str(e)}"]
            )
            
    async def _calculate_portfolio_correlation(self) -> float:
        """Portfolio korrelyatsiyasini hisoblash"""
        if len(self.current_positions) < 2:
            return 0.0
            
        # Group by base currency
        currency_groups = {}
        for position in self.current_positions.values():
            base = position.pair[:3]
            currency_groups[base] = currency_groups.get(base, 0) + 1
            
        # Calculate concentration
        max_concentration = max(currency_groups.values())
        correlation = max_concentration / len(self.current_positions)
        
        return correlation
        
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Risk metrikalarini olish"""
        analysis = await self.analyze_portfolio_risk()
        
        return {
            "current_risk_percentage": self._current_risk_percentage,
            "total_risk_amount": analysis.total_risk,
            "risk_level": analysis.risk_level.name,
            "open_positions": analysis.open_positions,
            "daily_pnl": -analysis.daily_loss,  # Negative because we track losses
            "volatility_score": f"{analysis.volatility_score:.2f}%",
            "correlation_risk": f"{analysis.correlation_risk:.2%}",
            "warnings": analysis.warnings,
            "recommendations": analysis.recommendations,
            "recent_stops": sum(1 for e, _ in self.risk_events if e == RiskEvent.STOP_LOSS_HIT),
            "risk_events": [(e.name, t.isoformat()) for e, t in self.risk_events[-10:]]
        }
        
    async def emergency_close_all(self) -> Dict[str, Any]:
        """Barcha pozitsiyalarni favqulodda yopish"""
        logger.warning("🚨 EMERGENCY CLOSE ALL POSITIONS")
        
        closed_positions = []
        total_pnl = 0.0
        
        for position_id, position in list(self.current_positions.items()):
            # Close at current price
            pnl_percentage = position.pnl_percentage
            pnl_amount = position.position_size * (pnl_percentage / 100)
            
            closed_positions.append({
                "pair": position.pair,
                "entry": position.entry_price,
                "exit": position.current_price,
                "pnl": pnl_amount,
                "pnl_percentage": pnl_percentage
            })
            
            total_pnl += pnl_percentage
            
            # Save to database
            if self.db_manager:
                await self.db_manager.save_trade_result({
                    "position_id": position_id,
                    "pair": position.pair,
                    "result": "EMERGENCY_CLOSE",
                    "pnl": pnl_amount,
                    "pnl_percentage": pnl_percentage
                })
                
        # Clear all positions
        self.current_positions.clear()
        
        # Record event
        self.risk_events.append((RiskEvent.DAILY_LOSS_LIMIT, TimeUtils.now_uzb()))
        
        return {
            "closed_count": len(closed_positions),
            "positions": closed_positions,
            "total_pnl": total_pnl,
            "timestamp": TimeUtils.now_uzb().isoformat()
        }
        
    async def _load_risk_metrics(self):
        """Saqlangan risk metrikalarini yuklash"""
        if not self.db_manager:
            return
            
        metrics = await self.db_manager.get_risk_metrics()
        if metrics:
            self._current_risk_percentage = metrics.current_risk_percentage
            self.daily_losses = metrics.daily_losses or {}
            
    async def save_risk_metrics(self):
        """Risk metrikalarini saqlash"""
        if not self.db_manager:
            return
            
        metrics = RiskMetrics(
            current_risk_percentage=self._current_risk_percentage,
            daily_losses=self.daily_losses,
            total_positions=len(self.current_positions),
            risk_events=[(e.name, t) for e, t in self.risk_events[-100:]],  # Keep last 100
            last_update=TimeUtils.now_uzb()
        )
        
        await self.db_manager.save_risk_metrics(metrics)
        
    def set_dynamic_risk(self, enabled: bool):
        """Dinamik risk yoqish/o'chirish"""
        self._dynamic_risk_enabled = enabled
        logger.info(f"Dynamic risk: {'ENABLED' if enabled else 'DISABLED'}")
        
    def set_risk_percentage(self, percentage: float):
        """Risk foizini o'rnatish"""
        if 0.1 <= percentage <= 2.0:
            self._current_risk_percentage = percentage
            logger.info(f"Risk percentage set to {percentage}%")
            return True
        else:
            logger.error(f"Invalid risk percentage: {percentage}")
            return False

# Global instance
risk_manager = RiskManager()
