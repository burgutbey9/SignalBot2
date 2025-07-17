"""
Database Manager and Crypto Trading Data Models
SQLAlchemy, async support, migrations, crypto-specific models
"""
import asyncio
from typing import Optional, List, Dict, Any, Union, Type
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index, JSON, DECIMAL, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.sql import text

Base = declarative_base()

class OrderStatus(str, Enum):
    """Order holatlari"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class SignalType(str, Enum):
    """Signal turlari"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"

class TradeResult(str, Enum):
    """Savdo natijalari"""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    PENDING = "PENDING"

# Database Models
class TradingPair(Base):
    """Crypto trading juftliklari"""
    __tablename__ = "trading_pairs"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    is_active = Column(Boolean, default=True)
    min_quantity = Column(DECIMAL(20, 8), default=0)
    max_quantity = Column(DECIMAL(20, 8), default=0)
    step_size = Column(DECIMAL(20, 8), default=0)
    tick_size = Column(DECIMAL(20, 8), default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signals = relationship("Signal", back_populates="pair", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="pair", cascade="all, delete-orphan")
    market_data = relationship("MarketData", back_populates="pair", cascade="all, delete-orphan")

class Signal(Base):
    """Trading signallari"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey("trading_pairs.id"), nullable=False)
    signal_type = Column(String(10), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)
    stop_loss = Column(DECIMAL(20, 8))
    take_profit = Column(DECIMAL(20, 8))
    confidence = Column(Float, default=0)
    risk_percentage = Column(Float, default=0.5)
    reason = Column(Text)
    
    # Analysis data
    ict_data = Column(JSON)  # ICT analysis ma'lumotlari
    smt_data = Column(JSON)  # SMT analysis ma'lumotlari
    order_flow_data = Column(JSON)  # Order flow ma'lumotlari
    sentiment_data = Column(JSON)  # Sentiment analysis
    
    is_executed = Column(Boolean, default=False)
    executed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    pair = relationship("TradingPair", back_populates="signals")
    orders = relationship("Order", back_populates="signal")
    
    __table_args__ = (
        Index("idx_signal_created", "created_at"),
        Index("idx_signal_pair_type", "pair_id", "signal_type"),
    )

class Order(Base):
    """Savdo buyurtmalari"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    pair_id = Column(Integer, ForeignKey("trading_pairs.id"), nullable=False)
    signal_id = Column(Integer, ForeignKey("signals.id"))
    
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT
    side = Column(String(10), nullable=False)  # BUY, SELL
    status = Column(String(20), default=OrderStatus.PENDING)
    
    quantity = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8))
    executed_quantity = Column(DECIMAL(20, 8), default=0)
    executed_price = Column(DECIMAL(20, 8))
    
    stop_loss = Column(DECIMAL(20, 8))
    take_profit = Column(DECIMAL(20, 8))
    
    commission = Column(DECIMAL(20, 8), default=0)
    commission_asset = Column(String(10))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    executed_at = Column(DateTime)
    
    # Relationships
    pair = relationship("TradingPair", back_populates="orders")
    signal = relationship("Signal", back_populates="orders")
    trades = relationship("Trade", back_populates="order", cascade="all, delete-orphan")

class Trade(Base):
    """Bajarilgan savdolar"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    
    entry_price = Column(DECIMAL(20, 8), nullable=False)
    exit_price = Column(DECIMAL(20, 8))
    quantity = Column(DECIMAL(20, 8), nullable=False)
    
    pnl = Column(DECIMAL(20, 8), default=0)
    pnl_percentage = Column(Float, default=0)
    result = Column(String(20), default=TradeResult.PENDING)
    
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    duration = Column(Integer)  # sekundlarda
    
    max_profit = Column(Float, default=0)  # savdo davomida maksimal profit
    max_loss = Column(Float, default=0)  # savdo davomida maksimal loss
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    order = relationship("Order", back_populates="trades")
    
    __table_args__ = (
        Index("idx_trade_result", "result"),
        Index("idx_trade_entry_time", "entry_time"),
    )

class MarketData(Base):
    """Bozor ma'lumotlari"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey("trading_pairs.id"), nullable=False)
    
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, etc
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, nullable=False)
    
    open = Column(DECIMAL(20, 8), nullable=False)
    high = Column(DECIMAL(20, 8), nullable=False)
    low = Column(DECIMAL(20, 8), nullable=False)
    close = Column(DECIMAL(20, 8), nullable=False)
    volume = Column(DECIMAL(20, 8), nullable=False)
    
    # Technical indicators
    indicators = Column(JSON)  # RSI, MACD, EMA, etc
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pair = relationship("TradingPair", back_populates="market_data")
    
    __table_args__ = (
        Index("idx_market_data_pair_time", "pair_id", "timeframe", "open_time"),
    )

class WhaleActivity(Base):
    """Whale harakatlari"""
    __tablename__ = "whale_activities"
    
    id = Column(Integer, primary_key=True)
    blockchain = Column(String(20), nullable=False)
    token = Column(String(20), nullable=False)
    
    from_address = Column(String(100))
    to_address = Column(String(100))
    amount = Column(DECIMAL(30, 8), nullable=False)
    usd_value = Column(DECIMAL(20, 2))
    
    transaction_hash = Column(String(100), unique=True)
    activity_type = Column(String(50))  # transfer, exchange_inflow, exchange_outflow, etc
    
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_whale_token_time", "token", "detected_at"),
    )

class BotState(Base):
    """Bot holati va konfiguratsiyasi"""
    __tablename__ = "bot_state"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseManager:
    """Async database manager"""
    _instance: Optional['DatabaseManager'] = None
    
    def __new__(cls, database_url: Optional[str] = None) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self, database_url: Optional[str] = None):
        if hasattr(self, '_initialized'): return
        self._initialized = True
        
        self.database_url = database_url or "sqlite+aiosqlite:///data/trading_bot.db"
        self.engine = None
        self.async_session = None
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Database initialization"""
        async with self._lock:
            if self.engine: return
            
            # Create data directory
            Path("data").mkdir(exist_ok=True)
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=20,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600
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
                
    async def get_session(self) -> AsyncSession:
        """Session olish"""
        if not self.engine: await self.initialize()
        return self.async_session()
        
    async def close(self) -> None:
        """Database yopish"""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            
    # CRUD Operations
    async def create(self, obj: Base) -> Base:
        """Yangi object yaratish"""
        async with self.async_session() as session:
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj
            
    async def bulk_create(self, objects: List[Base]) -> List[Base]:
        """Ko'plab objectlar yaratish"""
        async with self.async_session() as session:
            session.add_all(objects)
            await session.commit()
            return objects
            
    async def get(self, model: Type[Base], **filters) -> Optional[Base]:
        """Bitta object olish"""
        async with self.async_session() as session:
            query = select(model).filter_by(**filters)
            result = await session.execute(query)
            return result.scalar_one_or_none()
            
    async def get_all(self, model: Type[Base], limit: int = 100, offset: int = 0, **filters) -> List[Base]:
        """Ko'plab objectlar olish"""
        async with self.async_session() as session:
            query = select(model).filter_by(**filters).limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()
            
    async def update(self, obj: Base, **values) -> Base:
        """Object yangilash"""
        async with self.async_session() as session:
            for key, value in values.items():
                setattr(obj, key, value)
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj
            
    async def delete(self, obj: Base) -> bool:
        """Object o'chirish"""
        async with self.async_session() as session:
            await session.delete(obj)
            await session.commit()
            return True
            
    # Crypto-specific queries
    async def get_active_pairs(self) -> List[TradingPair]:
        """Aktiv trading juftliklarni olish"""
        return await self.get_all(TradingPair, is_active=True)
        
    async def get_recent_signals(self, hours: int = 24, limit: int = 50) -> List[Signal]:
        """So'nggi signallarni olish"""
        async with self.async_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            query = select(Signal).where(Signal.created_at >= cutoff).order_by(Signal.created_at.desc()).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()
            
    async def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Savdo statistikasini olish"""
        async with self.async_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Total trades
            total_query = select(func.count(Trade.id)).where(Trade.entry_time >= cutoff)
            total_result = await session.execute(total_query)
            total_trades = total_result.scalar()
            
            # Win/Loss stats
            stats_query = select(
                Trade.result,
                func.count(Trade.id),
                func.sum(Trade.pnl),
                func.avg(Trade.pnl_percentage)
            ).where(
                and_(Trade.entry_time >= cutoff, Trade.result != TradeResult.PENDING)
            ).group_by(Trade.result)
            
            stats_result = await session.execute(stats_query)
            stats = stats_result.all()
            
            return {
                "total_trades": total_trades,
                "win_rate": sum(s[1] for s in stats if s[0] == TradeResult.WIN) / max(sum(s[1] for s in stats), 1) * 100,
                "total_pnl": sum(s[2] or 0 for s in stats),
                "avg_win": next((s[3] for s in stats if s[0] == TradeResult.WIN), 0),
                "avg_loss": next((s[3] for s in stats if s[0] == TradeResult.LOSS), 0)
            }
            
    async def save_bot_state(self, key: str, value: Any) -> None:
        """Bot holatini saqlash"""
        state = await self.get(BotState, key=key)
        if state:
            await self.update(state, value=value)
        else:
            await self.create(BotState(key=key, value=value))
            
    async def get_bot_state(self, key: str) -> Any:
        """Bot holatini olish"""
        state = await self.get(BotState, key=key)
        return state.value if state else None

# Global instance
db_manager = DatabaseManager()
