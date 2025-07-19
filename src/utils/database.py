"""
Database Manager and Models
SQLAlchemy models, async database operations
"""
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, 
    Boolean, JSON, Text, ForeignKey, Index, UniqueConstraint,
    select, and_, or_, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.utils.logger import get_logger
from src.utils.helpers import TimeUtils

logger = get_logger(__name__)

# Base class for models
Base = declarative_base()

# Enums
class TradeStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

# Database Models
class BotState(Base):
    """Bot holati modeli"""
    __tablename__ = "bot_state"
    
    id = Column(Integer, primary_key=True)
    status = Column(String(50), nullable=False)
    signal_mode = Column(String(50), nullable=False)
    auto_trading = Column(Boolean, default=False)
    total_signals = Column(Integer, default=0)
    executed_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    failed_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    daily_stop_count = Column(Integer, default=0)
    last_update = Column(DateTime, default=TimeUtils.now_uzb)
    settings = Column(JSON, default={})

class TradingSignal(Base):
    """Trading signali modeli"""
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    type = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    position_size = Column(Float)
    risk_reward_ratio = Column(Float)
    confidence = Column(Float, nullable=False)
    sources = Column(JSON, default=[])
    analysis = Column(JSON, default={})
    reasoning = Column(JSON, default=[])
    created_at = Column(DateTime, default=TimeUtils.now_uzb, index=True)
    expires_at = Column(DateTime)
    executed = Column(Boolean, default=False)
    
    # Relationships
    trades = relationship("Trade", back_populates="signal")
    
    __table_args__ = (
        Index('idx_symbol_created', 'symbol', 'created_at'),
    )

class Trade(Base):
    """Savdo modeli"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), unique=True, nullable=False)
    signal_id = Column(String(100), ForeignKey('trading_signals.signal_id'))
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    position_size = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    status = Column(String(20), default=TradeStatus.PENDING.value, index=True)
    opened_at = Column(DateTime, default=TimeUtils.now_uzb)
    closed_at = Column(DateTime)
    duration_minutes = Column(Integer)
    exit_reason = Column(String(50))
    detailed_reason = Column(Text)
    market_conditions = Column(JSON, default={})
    
    # Relationships
    signal = relationship("TradingSignal", back_populates="trades")
    orders = relationship("Order", back_populates="trade")
    analysis = relationship("TradeAnalysis", back_populates="trade", uselist=False)
    
    __table_args__ = (
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_opened_at', 'opened_at'),
    )

class Order(Base):
    """Order modeli"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(100), unique=True, nullable=False)
    trade_id = Column(String(100), ForeignKey('trades.trade_id'))
    symbol = Column(String(20), nullable=False, index=True)
    type = Column(String(20), nullable=False)  # OrderType enum
    side = Column(String(10), nullable=False)
    price = Column(Float)
    stop_price = Column(Float)
    quantity = Column(Float, nullable=False)
    executed_qty = Column(Float, default=0.0)
    status = Column(String(20), nullable=False, index=True)
    commission = Column(Float, default=0.0)
    created_at = Column(DateTime, default=TimeUtils.now_uzb)
    updated_at = Column(DateTime, default=TimeUtils.now_uzb, onupdate=TimeUtils.now_uzb)
    
    # Relationships
    trade = relationship("Trade", back_populates="orders")

class TradeAnalysis(Base):
    """Savdo tahlili modeli"""
    __tablename__ = "trade_analysis"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id'), unique=True)
    symbol = Column(String(20), nullable=False)
    result = Column(String(20), nullable=False)  # WIN/LOSS/BREAKEVEN
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float, nullable=False)
    exit_reason = Column(String(50), nullable=False)
    detailed_reason = Column(Text)
    stop_loss_reason = Column(String(50))
    take_profit_reason = Column(String(50))
    duration_minutes = Column(Integer)
    market_conditions = Column(JSON, default={})
    performance_metrics = Column(JSON, default={})
    timestamp = Column(DateTime, default=TimeUtils.now_uzb)
    
    # Relationships
    trade = relationship("Trade", back_populates="analysis")

class MarketData(Base):
    """Bozor ma'lumotlari modeli"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    indicators = Column(JSON, default={})
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp'),
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )

class RiskMetrics(Base):
    """Risk metrikalari modeli"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    average_win = Column(Float, default=0.0)
    average_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    risk_score = Column(Float, default=50.0)
    daily_metrics = Column(JSON, default={})
    created_at = Column(DateTime, default=TimeUtils.now_uzb)

class DatabaseManager:
    """Database boshqaruv klassi"""
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or "sqlite+aiosqlite:///data/signalbot.db"
        self.engine = None
        self.async_session = None
        
    async def initialize(self):
        """Database ni sozlash"""
        try:
            logger.info("Database initialization boshlandi...")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                poolclass=NullPool
            )
            
            # Create session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info("✅ Database initialization tugadi")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization xatosi: {e}")
            return False
            
    async def close(self):
        """Database yopish"""
        if self.engine:
            await self.engine.dispose()
            
    # Bot State operations
    async def get_bot_state(self) -> Optional[BotState]:
        """Bot holatini olish"""
        async with self.async_session() as session:
            result = await session.execute(select(BotState).order_by(desc(BotState.id)).limit(1))
            return result.scalar_one_or_none()
            
    async def save_bot_state(self, state: BotState):
        """Bot holatini saqlash"""
        async with self.async_session() as session:
            session.add(state)
            await session.commit()
            
    # Signal operations
    async def save_signal(self, signal: TradingSignal):
        """Signalni saqlash"""
        async with self.async_session() as session:
            session.add(signal)
            await session.commit()
            
    async def get_active_signals(self) -> List[TradingSignal]:
        """Aktiv signallarni olish"""
        async with self.async_session() as session:
            now = TimeUtils.now_uzb()
            result = await session.execute(
                select(TradingSignal)
                .where(
                    and_(
                        TradingSignal.executed == False,
                        or_(
                            TradingSignal.expires_at == None,
                            TradingSignal.expires_at > now
                        )
                    )
                )
                .order_by(desc(TradingSignal.created_at))
            )
            return result.scalars().all()
            
    async def get_signals_by_symbol(self, symbol: str, limit: int = 50) -> List[TradingSignal]:
        """Symbol bo'yicha signallarni olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(TradingSignal)
                .where(TradingSignal.symbol == symbol)
                .order_by(desc(TradingSignal.created_at))
                .limit(limit)
            )
            return result.scalars().all()
            
    # Trade operations
    async def save_trade(self, trade: Trade):
        """Savdoni saqlash"""
        async with self.async_session() as session:
            session.add(trade)
            await session.commit()
            
    async def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """Savdoni yangilash"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.trade_id == trade_id)
            )
            trade = result.scalar_one_or_none()
            
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                await session.commit()
                
    async def get_open_trades(self) -> List[Trade]:
        """Ochiq savdolarni olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.status == TradeStatus.OPEN.value)
                .order_by(desc(Trade.opened_at))
            )
            return result.scalars().all()
            
    async def get_trades_by_symbol(self, symbol: str, status: Optional[str] = None) -> List[Trade]:
        """Symbol bo'yicha savdolarni olish"""
        async with self.async_session() as session:
            query = select(Trade).where(Trade.symbol == symbol)
            
            if status:
                query = query.where(Trade.status == status)
                
            query = query.order_by(desc(Trade.opened_at))
            
            result = await session.execute(query)
            return result.scalars().all()
            
    async def get_trades_after(self, after_date: datetime) -> List[Trade]:
        """Sanadan keyingi savdolarni olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.opened_at > after_date)
                .order_by(desc(Trade.opened_at))
            )
            return result.scalars().all()
            
    # Order operations
    async def save_order(self, order: Order):
        """Orderni saqlash"""
        async with self.async_session() as session:
            session.add(order)
            await session.commit()
            
    async def update_order(self, order_id: str, updates: Dict[str, Any]):
        """Orderni yangilash"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            order = result.scalar_one_or_none()
            
            if order:
                for key, value in updates.items():
                    setattr(order, key, value)
                order.updated_at = TimeUtils.now_uzb()
                await session.commit()
                
    async def get_orders_by_trade(self, trade_id: str) -> List[Order]:
        """Savdo bo'yicha orderlarni olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Order)
                .where(Order.trade_id == trade_id)
                .order_by(desc(Order.created_at))
            )
            return result.scalars().all()
            
    # Trade Analysis operations
    async def save_trade_analysis(self, analysis: TradeAnalysis):
        """Savdo tahlilini saqlash"""
        async with self.async_session() as session:
            session.add(analysis)
            await session.commit()
            
    async def get_trade_analysis(self, trade_id: str) -> Optional[TradeAnalysis]:
        """Savdo tahlilini olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(TradeAnalysis).where(TradeAnalysis.trade_id == trade_id)
            )
            return result.scalar_one_or_none()
            
    # Market Data operations
    async def save_market_data(self, data: MarketData):
        """Bozor ma'lumotlarini saqlash"""
        async with self.async_session() as session:
            session.add(data)
            await session.commit()
            
    async def save_market_data_bulk(self, data_list: List[MarketData]):
        """Ko'p bozor ma'lumotlarini saqlash"""
        async with self.async_session() as session:
            session.add_all(data_list)
            await session.commit()
            
    async def get_market_data(self, symbol: str, timeframe: str, 
                            start_time: datetime, end_time: datetime) -> List[MarketData]:
        """Bozor ma'lumotlarini olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(MarketData)
                .where(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.timeframe == timeframe,
                        MarketData.timestamp >= start_time,
                        MarketData.timestamp <= end_time
                    )
                )
                .order_by(asc(MarketData.timestamp))
            )
            return result.scalars().all()
            
    # Risk Metrics operations
    async def save_risk_metrics(self, metrics: RiskMetrics):
        """Risk metrikalarini saqlash"""
        async with self.async_session() as session:
            session.add(metrics)
            await session.commit()
            
    async def get_latest_risk_metrics(self) -> Optional[RiskMetrics]:
        """Oxirgi risk metrikalarini olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(RiskMetrics).order_by(desc(RiskMetrics.date)).limit(1)
            )
            return result.scalar_one_or_none()
            
    async def get_risk_metrics_range(self, start_date: datetime, end_date: datetime) -> List[RiskMetrics]:
        """Vaqt oralig'idagi risk metrikalarini olish"""
        async with self.async_session() as session:
            result = await session.execute(
                select(RiskMetrics)
                .where(
                    and_(
                        RiskMetrics.date >= start_date,
                        RiskMetrics.date <= end_date
                    )
                )
                .order_by(asc(RiskMetrics.date))
            )
            return result.scalars().all()
            
    # Statistics
    async def get_trading_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Trading statistikalarini olish"""
        async with self.async_session() as session:
            start_date = TimeUtils.now_uzb() - timedelta(days=days)
            
            # Get trades
            result = await session.execute(
                select(Trade)
                .where(
                    and_(
                        Trade.opened_at >= start_date,
                        Trade.status == TradeStatus.CLOSED.value
                    )
                )
            )
            trades = result.scalars().all()
            
            if not trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0
                }
                
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            
            return {
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": (len(winning_trades) / len(trades) * 100) if trades else 0,
                "total_pnl": sum(t.pnl for t in trades),
                "avg_win": (total_wins / len(winning_trades)) if winning_trades else 0,
                "avg_loss": (total_losses / len(losing_trades)) if losing_trades else 0,
                "profit_factor": (total_wins / total_losses) if total_losses > 0 else 0
            }
