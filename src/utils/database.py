"""
Advanced Database Manager with Connection Pooling and Performance Optimization
SQLite va PostgreSQL uchun async database operations
"""
import asyncio
import json
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import sqlite3
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, 
    async_sessionmaker, AsyncConnection
)
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, JSON, Index, event, select, update, delete,
    func, and_, or_, desc, asc
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import uuid

from utils.logger import get_logger
from utils.helpers import TimeUtils

logger = get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()

class SignalStatus(Enum):
    """Signal holati"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class TradeStatus(Enum):
    """Trade holati"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

# Database Models
class TradingSignal(Base):
    """Trading signal modeli"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    type = Column(String(20), nullable=False)  # BUY, SELL, STRONG_BUY, etc.
    action = Column(String(10), nullable=False)  # BUY, SELL
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    confidence = Column(Float, nullable=False)
    position_size = Column(Float)
    risk_reward_ratio = Column(Float)
    sources = Column(JSON)  # Analysis sources
    reasoning = Column(JSON)  # Reasoning list
    status = Column(String(20), default=SignalStatus.PENDING.value, index=True)
    timestamp = Column(DateTime, default=TimeUtils.now_uzb, index=True)
    expires_at = Column(DateTime)
    executed_at = Column(DateTime)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_signal_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_signal_status_timestamp', 'status', 'timestamp'),
        Index('idx_signal_expires', 'expires_at'),
    )

class Trade(Base):
    """Trade modeli"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    signal_id = Column(String(36), index=True)  # Foreign key to signal
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    status = Column(String(20), default=TradeStatus.OPEN.value, index=True)
    
    # Timestamps
    opened_at = Column(DateTime, default=TimeUtils.now_uzb, index=True)
    closed_at = Column(DateTime)
    
    # Trade metadata
    strategy = Column(String(50))  # ICT, SMT, etc.
    notes = Column(Text)
    metadata = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_trade_symbol_opened', 'symbol', 'opened_at'),
        Index('idx_trade_status_opened', 'status', 'opened_at'),
        Index('idx_trade_signal', 'signal_id'),
    )

class BotState(Base):
    """Bot holati modeli"""
    __tablename__ = 'bot_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String(20), nullable=False)
    signal_mode = Column(String(20), nullable=False)
    auto_trading = Column(Boolean, default=False)
    total_signals = Column(Integer, default=0)
    executed_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    failed_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    daily_stop_count = Column(Integer, default=0)
    last_update = Column(DateTime, default=TimeUtils.now_uzb)
    
    # Additional metrics
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

class PerformanceMetric(Base):
    """Performance metrics modeli"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=TimeUtils.now_uzb, index=True)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
    )

class APILog(Base):
    """API call logs"""
    __tablename__ = 'api_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(200))
    method = Column(String(10), default='GET')
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=TimeUtils.now_uzb, index=True)
    
    __table_args__ = (
        Index('idx_api_provider_timestamp', 'provider', 'timestamp'),
    )

class DatabaseManager:
    """Advanced Database Manager with connection pooling"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        self._connection_retries = 3
        self._pool_size = 10
        self._max_overflow = 20
        
        # Performance monitoring
        self._query_count = 0
        self._slow_query_threshold = 1.0  # seconds
        
    async def initialize(self, database_url: Optional[str] = None) -> bool:
        """Database ni sozlash"""
        try:
            if self._initialized:
                logger.warning("Database already initialized")
                return True
                
            # Get database URL
            if not database_url:
                database_url = self._get_database_url()
                
            logger.info(f"🔄 Database initializing: {database_url.split('://')[0]}://...")
            
            # Create engine with connection pooling
            self.engine = await self._create_engine(database_url)
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            await self._create_tables()
            
            # Setup event listeners
            self._setup_event_listeners()
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def _get_database_url(self) -> str:
        """Database URL ni olish"""
        import os
        
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
            
        # Default SQLite
        db_path = Path("data/signalbot.db")
        db_path.parent.mkdir(exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"
    
    async def _create_engine(self, database_url: str) -> AsyncEngine:
        """Engine yaratish with optimized settings"""
        
        if database_url.startswith('sqlite'):
            # SQLite configuration
            engine = create_async_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                },
                echo=False,  # Set to True for SQL debugging
                future=True
            )
            
            # SQLite optimization
            @event.listens_for(engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                # Performance optimizations
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL") 
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=memory")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()
                
        else:
            # PostgreSQL configuration
            engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                echo=False,
                future=True
            )
        
        return engine
    
    async def _create_tables(self):
        """Tabllarni yaratish"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified")
    
    def _setup_event_listeners(self):
        """Event listeners ni o'rnatish"""
        from sqlalchemy import event
        
        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = datetime.now()
            self._query_count += 1
        
        @event.listens_for(self.engine.sync_engine, "after_cursor_execute") 
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = (datetime.now() - context._query_start_time).total_seconds()
            if total > self._slow_query_threshold:
                logger.warning(f"Slow query detected: {total:.2f}s - {statement[:100]}...")
    
    async def _test_connection(self):
        """Connection ni test qilish"""
        async with self.get_session() as session:
            result = await session.execute(select(1))
            result.scalar()
        logger.info("✅ Database connection test passed")
    
    async def get_session(self) -> AsyncSession:
        """Async session olish"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        return self.async_session_factory()
    
    async def close(self):
        """Database connection ni yopish"""
        if self.engine:
            await self.engine.dispose()
            logger.info("✅ Database connections closed")
    
    # Signal Operations
    async def save_signal(self, signal: TradingSignal) -> bool:
        """Signal saqlash"""
        try:
            async with self.get_session() as session:
                session.add(signal)
                await session.commit()
                logger.debug(f"Signal saved: {signal.symbol} - {signal.action}")
                return True
        except IntegrityError:
            logger.warning(f"Signal already exists: {signal.signal_id}")
            return False
        except Exception as e:
            logger.error(f"Signal save error: {e}")
            return False
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """Aktiv signallarni olish"""
        try:
            async with self.get_session() as session:
                query = select(TradingSignal).where(
                    TradingSignal.status == SignalStatus.PENDING.value
                )
                
                if symbol:
                    query = query.where(TradingSignal.symbol == symbol)
                
                query = query.order_by(desc(TradingSignal.timestamp))
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Get active signals error: {e}")
            return []
    
    async def update_signal_status(self, signal_id: str, status: SignalStatus, 
                                 executed_at: Optional[datetime] = None) -> bool:
        """Signal statusini yangilash"""
        try:
            async with self.get_session() as session:
                query = update(TradingSignal).where(
                    TradingSignal.signal_id == signal_id
                ).values(
                    status=status.value,
                    executed_at=executed_at or TimeUtils.now_uzb()
                )
                
                await session.execute(query)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Update signal status error: {e}")
            return False
    
    # Trade Operations
    async def save_trade(self, trade: Trade) -> bool:
        """Trade saqlash"""
        try:
            async with self.get_session() as session:
                session.add(trade)
                await session.commit()
                logger.info(f"Trade saved: {trade.symbol} - {trade.side}")
                return True
        except Exception as e:
            logger.error(f"Trade save error: {e}")
            return False
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """Ochiq tradelarni olish"""
        try:
            async with self.get_session() as session:
                query = select(Trade).where(Trade.status == TradeStatus.OPEN.value)
                
                if symbol:
                    query = query.where(Trade.symbol == symbol)
                
                query = query.order_by(desc(Trade.opened_at))
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Get open trades error: {e}")
            return []
    
    async def close_trade(self, trade_id: str, exit_price: float, 
                         pnl: float, fees: float = 0.0) -> bool:
        """Trade yopish"""
        try:
            async with self.get_session() as session:
                # Calculate PnL percentage
                trade_query = select(Trade).where(Trade.trade_id == trade_id)
                result = await session.execute(trade_query)
                trade = result.scalar_one_or_none()
                
                if not trade:
                    logger.error(f"Trade not found: {trade_id}")
                    return False
                
                pnl_percentage = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                if trade.side == "SELL":
                    pnl_percentage = -pnl_percentage
                
                # Update trade
                update_query = update(Trade).where(
                    Trade.trade_id == trade_id
                ).values(
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    fees=fees,
                    status=TradeStatus.CLOSED.value,
                    closed_at=TimeUtils.now_uzb()
                )
                
                await session.execute(update_query)
                await session.commit()
                
                logger.info(f"Trade closed: {trade_id} - PnL: {pnl:+.2f}")
                return True
        except Exception as e:
            logger.error(f"Close trade error: {e}")
            return False
    
    # Bot State Operations
    async def save_bot_state(self, state: BotState) -> bool:
        """Bot holatini saqlash"""
        try:
            async with self.get_session() as session:
                # Delete old state (keep only latest)
                await session.execute(delete(BotState))
                
                # Save new state
                session.add(state)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Save bot state error: {e}")
            return False
    
    async def get_bot_state(self) -> Optional[BotState]:
        """Bot holatini olish"""
        try:
            async with self.get_session() as session:
                query = select(BotState).order_by(desc(BotState.last_update)).limit(1)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Get bot state error: {e}")
            return None
    
    # Performance Metrics
    async def save_metric(self, name: str, value: float, metadata: Optional[Dict] = None) -> bool:
        """Performance metric saqlash"""
        try:
            async with self.get_session() as session:
                metric = PerformanceMetric(
                    metric_name=name,
                    metric_value=value,
                    metadata=metadata
                )
                session.add(metric)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Save metric error: {e}")
            return False
    
    async def get_metrics(self, name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Metrikkalarni olish"""
        try:
            since = TimeUtils.now_uzb() - timedelta(hours=hours)
            
            async with self.get_session() as session:
                query = select(PerformanceMetric).where(
                    and_(
                        PerformanceMetric.metric_name == name,
                        PerformanceMetric.timestamp >= since
                    )
                ).order_by(PerformanceMetric.timestamp)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Get metrics error: {e}")
            return []
    
    # API Logging
    async def log_api_call(self, provider: str, endpoint: str, method: str = "GET",
                          status_code: Optional[int] = None, response_time_ms: Optional[float] = None,
                          error_message: Optional[str] = None) -> bool:
        """API call logini saqlash"""
        try:
            async with self.get_session() as session:
                api_log = APILog(
                    provider=provider,
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    error_message=error_message
                )
                session.add(api_log)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"API log save error: {e}")
            return False
    
    # Analytics and Reports
    async def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Trading statistikasi"""
        try:
            since = TimeUtils.now_uzb() - timedelta(days=days)
            
            async with self.get_session() as session:
                # Basic stats
                stats_query = select(
                    func.count(Trade.id).label('total_trades'),
                    func.count(Trade.id).filter(Trade.pnl > 0).label('winning_trades'),
                    func.sum(Trade.pnl).label('total_pnl'),
                    func.avg(Trade.pnl).label('avg_pnl'),
                    func.max(Trade.pnl).label('max_profit'),
                    func.min(Trade.pnl).label('max_loss'),
                    func.sum(Trade.fees).label('total_fees')
                ).where(
                    and_(
                        Trade.status == TradeStatus.CLOSED.value,
                        Trade.closed_at >= since
                    )
                )
                
                result = await session.execute(stats_query)
                stats = result.first()
                
                if not stats or stats.total_trades == 0:
                    return {"message": "No trades found"}
                
                win_rate = (stats.winning_trades / stats.total_trades) * 100
                
                # Symbol breakdown
                symbol_query = select(
                    Trade.symbol,
                    func.count(Trade.id).label('count'),
                    func.sum(Trade.pnl).label('pnl')
                ).where(
                    and_(
                        Trade.status == TradeStatus.CLOSED.value,
                        Trade.closed_at >= since
                    )
                ).group_by(Trade.symbol).order_by(desc('pnl'))
                
                symbol_result = await session.execute(symbol_query)
                symbols = [
                    {"symbol": row.symbol, "trades": row.count, "pnl": float(row.pnl)}
                    for row in symbol_result
                ]
                
                return {
                    "period_days": days,
                    "total_trades": stats.total_trades,
                    "winning_trades": stats.winning_trades,
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(float(stats.total_pnl), 2),
                    "avg_pnl": round(float(stats.avg_pnl), 2),
                    "max_profit": round(float(stats.max_profit), 2),
                    "max_loss": round(float(stats.max_loss), 2),
                    "total_fees": round(float(stats.total_fees), 2),
                    "symbols": symbols
                }
                
        except Exception as e:
            logger.error(f"Get trade statistics error: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days: int = 90) -> bool:
        """Eski ma'lumotlarni tozalash"""
        try:
            cutoff_date = TimeUtils.now_uzb() - timedelta(days=days)
            
            async with self.get_session() as session:
                # Clean old signals
                await session.execute(
                    delete(TradingSignal).where(TradingSignal.timestamp < cutoff_date)
                )
                
                # Clean old metrics
                await session.execute(
                    delete(PerformanceMetric).where(PerformanceMetric.timestamp < cutoff_date)
                )
                
                # Clean old API logs
                await session.execute(
                    delete(APILog).where(APILog.timestamp < cutoff_date)
                )
                
                await session.commit()
                logger.info(f"✅ Cleaned data older than {days} days")
                return True
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Database statistikasi"""
        return {
            "initialized": self._initialized,
            "query_count": self._query_count,
            "pool_size": self._pool_size,
            "max_overflow": self._max_overflow,
            "connection_retries": self._connection_retries
        }

# Global instance
database_manager = DatabaseManager()
