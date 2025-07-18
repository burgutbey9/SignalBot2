"""
Bot Lifecycle Manager and Signal Controller
Bot ishga tushirish, to'xtatish, signal boshqaruv
"""
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import signal
import sys

from config.config import config_manager, TradingMode
from utils.logger import get_logger
from utils.helpers import TimeUtils, PerformanceMonitor
from api.telegram_client import telegram_notifier
from utils.database import DatabaseManager, BotState, TradingSignal

logger = get_logger(__name__)

class BotStatus(Enum):
    """Bot holati"""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    ERROR = auto()

class SignalMode(Enum):
    """Signal rejimi"""
    ACTIVE = auto()
    PAUSED = auto()
    MANUAL_ONLY = auto()

@dataclass
class BotStatistics:
    """Bot statistikasi"""
    start_time: datetime
    total_signals: int = 0
    executed_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    uptime_seconds: float = 0.0
    errors_count: int = 0
    last_signal_time: Optional[datetime] = None
    
    @property
    def win_rate(self) -> float:
        """G'alaba foizi"""
        if self.executed_trades == 0:
            return 0.0
        return (self.successful_trades / self.executed_trades) * 100
        
    @property
    def average_pnl(self) -> float:
        """O'rtacha PnL"""
        if self.executed_trades == 0:
            return 0.0
        return self.total_pnl / self.executed_trades

class BotManager:
    """Asosiy bot manager"""
    def __init__(self):
        self.status = BotStatus.STOPPED
        self.signal_mode = SignalMode.PAUSED
        self.trading_mode = TradingMode.SIGNAL_ONLY
        self.statistics = BotStatistics(start_time=TimeUtils.now_uzb())
        self.performance_monitor = PerformanceMonitor()
        self.db_manager: Optional[DatabaseManager] = None
        
        # Components
        self._components: Dict[str, Any] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_handlers: List[Callable] = []
        
        # Control flags
        self._running = False
        self._auto_trading_enabled = False
        self._daily_stop_count = 0
        self._last_stop_reset = TimeUtils.now_uzb().date()
        
    async def initialize(self) -> bool:
        """Bot komponentlarini sozlash"""
        try:
            logger.info("🚀 Bot initialization boshlandi...")
            self.status = BotStatus.STARTING
            
            # Load configuration
            if not await config_manager.load_config():
                raise Exception("Konfiguratsiya yuklanmadi")
                
            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Load saved state
            await self._load_bot_state()
            
            # Trading mode
            self.trading_mode = config_manager.trading.mode
            
            # Register shutdown handlers
            self._register_shutdown_handlers()
            
            logger.info("✅ Bot initialization tugadi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization xatosi: {e}")
            self.status = BotStatus.ERROR
            return False
            
    def _register_shutdown_handlers(self):
        """Shutdown handlerlarni ro'yxatdan o'tkazish"""
        def signal_handler(sig, frame):
            logger.info("🛑 Shutdown signal qabul qilindi")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def start(self) -> bool:
        """Botni ishga tushirish"""
        if self.status == BotStatus.RUNNING:
            logger.warning("Bot allaqachon ishlamoqda")
            return True
            
        try:
            logger.info("🟢 Bot ishga tushmoqda...")
            self.status = BotStatus.STARTING
            
            # Initialize components
            if not await self.initialize():
                return False
                
            # Start components
            await self._start_components()
            
            # Start monitoring
            self._tasks["monitor"] = asyncio.create_task(self._monitor_loop())
            self._tasks["health_check"] = asyncio.create_task(self._health_check_loop())
            
            self._running = True
            self.status = BotStatus.RUNNING
            self.statistics.start_time = TimeUtils.now_uzb()
            
            # Notify
            await telegram_notifier.notify_market_update({
                "mood": "NEUTRAL",
                "message": "🤖 Bot muvaffaqiyatli ishga tushdi!",
                "timestamp": TimeUtils.now_uzb()
            })
            
            logger.info("✅ Bot to'liq ishga tushdi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot start xatosi: {e}")
            self.status = BotStatus.ERROR
            self.statistics.errors_count += 1
            return False
            
    async def _start_components(self):
        """Komponentlarni ishga tushirish"""
        # Import here to avoid circular imports
        from analysis.ict_analyzer import ICTAnalyzer
        from analysis.smt_analyzer import SMTAnalyzer
        from analysis.order_flow import OrderFlowAnalyzer
        from analysis.sentiment import SentimentAnalyzer
        from trading.signal_generator import SignalGenerator
        from trading.execution_engine import ExecutionEngine
        from api.ai_clients import ai_orchestrator
        
        # Initialize analyzers
        self._components["ict"] = ICTAnalyzer()
        self._components["smt"] = SMTAnalyzer()
        self._components["order_flow"] = OrderFlowAnalyzer()
        self._components["sentiment"] = SentimentAnalyzer()
        
        # Initialize trading components
        self._components["signal_generator"] = SignalGenerator()
        self._components["execution_engine"] = ExecutionEngine()
        
        # Initialize AI
        await ai_orchestrator.initialize()
        
        # Start component tasks
        for name, component in self._components.items():
            if hasattr(component, "start"):
                self._tasks[name] = asyncio.create_task(component.start())
                
        logger.info(f"✅ {len(self._components)} ta komponent ishga tushdi")
        
    async def stop(self) -> bool:
        """Botni to'xtatish"""
        if self.status == BotStatus.STOPPED:
            logger.warning("Bot allaqachon to'xtatilgan")
            return True
            
        try:
            logger.info("🔴 Bot to'xtatilmoqda...")
            self.status = BotStatus.STOPPING
            self._running = False
            
            # Stop all tasks
            for name, task in self._tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
            # Stop components
            await self._stop_components()
            
            # Save state
            await self._save_bot_state()
            
            # Calculate uptime
            uptime = (TimeUtils.now_uzb() - self.statistics.start_time).total_seconds()
            self.statistics.uptime_seconds = uptime
            
            self.status = BotStatus.STOPPED
            
            logger.info(f"✅ Bot to'xtatildi. Uptime: {uptime/3600:.1f} soat")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot stop xatosi: {e}")
            self.statistics.errors_count += 1
            return False
            
    async def _stop_components(self):
        """Komponentlarni to'xtatish"""
        for name, component in self._components.items():
            if hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception as e:
                    logger.error(f"Component {name} stop xatosi: {e}")
                    
    async def pause(self) -> bool:
        """Botni pause qilish"""
        if self.status != BotStatus.RUNNING:
            logger.warning("Bot ishlamayapti, pause qilib bo'lmaydi")
            return False
            
        self.status = BotStatus.PAUSED
        self.signal_mode = SignalMode.PAUSED
        
        logger.info("⏸️ Bot pause qilindi")
        
        await telegram_notifier.notify_market_update({
            "mood": "NEUTRAL",
            "message": "⏸️ Bot vaqtincha to'xtatildi",
            "timestamp": TimeUtils.now_uzb()
        })
        
        return True
        
    async def resume(self) -> bool:
        """Botni davom ettirish"""
        if self.status != BotStatus.PAUSED:
            logger.warning("Bot pause holatida emas")
            return False
            
        self.status = BotStatus.RUNNING
        self.signal_mode = SignalMode.ACTIVE
        
        logger.info("▶️ Bot davom ettirildi")
        
        await telegram_notifier.notify_market_update({
            "mood": "NEUTRAL",
            "message": "▶️ Bot yana ishga tushdi",
            "timestamp": TimeUtils.now_uzb()
        })
        
        return True
        
    async def shutdown(self):
        """Botni to'liq o'chirish"""
        logger.info("🛑 Bot shutdown boshlandi...")
        
        # Run shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                await handler() if asyncio.iscoroutinefunction(handler) else handler()
            except Exception as e:
                logger.error(f"Shutdown handler xatosi: {e}")
                
        # Stop bot
        await self.stop()
        
        # Close database
        if self.db_manager:
            await self.db_manager.close()
            
        logger.info("✅ Bot shutdown tugadi")
        
    def enable_auto_trading(self) -> bool:
        """Auto trading yoqish"""
        if self.trading_mode == TradingMode.SIGNAL_ONLY:
            logger.warning("Bot SIGNAL_ONLY rejimida, auto trading yoqib bo'lmaydi")
            return False
            
        self._auto_trading_enabled = True
        logger.info("✅ Auto trading yoqildi")
        return True
        
    def disable_auto_trading(self) -> bool:
        """Auto trading o'chirish"""
        self._auto_trading_enabled = False
        logger.info("🔴 Auto trading o'chirildi")
        return True
        
    async def process_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Signalni qayta ishlash"""
        if self.signal_mode == SignalMode.PAUSED:
            logger.info("Signal mode paused, signal o'tkazib yuborildi")
            return False
            
        try:
            # Update statistics
            self.statistics.total_signals += 1
            self.statistics.last_signal_time = TimeUtils.now_uzb()
            
            # Check daily stop limit
            if self._check_daily_stop_limit():
                logger.warning("Kunlik stop limit, signal o'tkazib yuborildi")
                return False
                
            # Save signal to database
            if self.db_manager:
                await self.db_manager.save_signal(TradingSignal(**signal_data))
                
            # Send notification
            await telegram_notifier.notify_signal(signal_data)
            
            # Execute trade if auto trading enabled
            if self._auto_trading_enabled and self.trading_mode in [TradingMode.LIVE, TradingMode.PAPER]:
                execution_result = await self._execute_trade(signal_data)
                
                if execution_result:
                    self.statistics.executed_trades += 1
                    
            return True
            
        except Exception as e:
            logger.error(f"Signal process xatosi: {e}")
            self.statistics.errors_count += 1
            return False
            
    async def _execute_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Savdoni bajarish"""
        try:
            execution_engine = self._components.get("execution_engine")
            if not execution_engine:
                logger.error("Execution engine topilmadi")
                return False
                
            result = await execution_engine.execute_signal(signal_data)
            
            if result.get("success"):
                self.statistics.successful_trades += 1
            else:
                self.statistics.failed_trades += 1
                
                # Check stop loss
                if result.get("reason") == "stop_loss":
                    self._daily_stop_count += 1
                    
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Trade execution xatosi: {e}")
            return False
            
    def _check_daily_stop_limit(self) -> bool:
        """Kunlik stop limitni tekshirish"""
        # Reset daily counter
        today = TimeUtils.now_uzb().date()
        if today > self._last_stop_reset:
            self._daily_stop_count = 0
            self._last_stop_reset = today
            
        # Check limit (2-3 stop = pause)
        if self._daily_stop_count >= 2:
            return True
            
        return False
        
    async def _monitor_loop(self):
        """Monitoring loop"""
        while self._running:
            try:
                # Performance metrics
                self.performance_monitor.record("bot_uptime", 
                    (TimeUtils.now_uzb() - self.statistics.start_time).total_seconds())
                self.performance_monitor.record("total_signals", self.statistics.total_signals)
                self.performance_monitor.record("win_rate", self.statistics.win_rate)
                
                # Check components health
                for name, component in self._components.items():
                    if hasattr(component, "is_healthy"):
                        if not await component.is_healthy():
                            logger.warning(f"Component {name} unhealthy")
                            
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Monitor loop xatosi: {e}")
                await asyncio.sleep(60)
                
    async def _health_check_loop(self):
        """Health check loop"""
        while self._running:
            try:
                # System health check
                health_status = await self._check_system_health()
                
                if not health_status["healthy"]:
                    logger.warning(f"System health issue: {health_status}")
                    
                    # Critical issues
                    if health_status.get("critical"):
                        await telegram_notifier.notify_error(
                            f"🚨 CRITICAL: {health_status.get('message')}",
                            critical=True
                        )
                        
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Health check xatosi: {e}")
                await asyncio.sleep(300)
                
    async def _check_system_health(self) -> Dict[str, Any]:
        """System salomatligini tekshirish"""
        health = {
            "healthy": True,
            "components": {},
            "metrics": {}
        }
        
        # Check each component
        for name, component in self._components.items():
            if hasattr(component, "get_health_status"):
                status = await component.get_health_status()
                health["components"][name] = status
                if not status.get("healthy", True):
                    health["healthy"] = False
                    
        # Check metrics
        health["metrics"]["error_rate"] = self.statistics.errors_count / max(self.statistics.total_signals, 1)
        health["metrics"]["win_rate"] = self.statistics.win_rate
        
        # Critical thresholds
        if health["metrics"]["error_rate"] > 0.1:  # 10% error rate
            health["critical"] = True
            health["message"] = "Xatolik darajasi juda yuqori!"
            
        return health
        
    async def _load_bot_state(self):
        """Bot holatini yuklash"""
        if not self.db_manager:
            return
            
        state = await self.db_manager.get_bot_state()
        if state:
            self.statistics.total_signals = state.total_signals
            self.statistics.executed_trades = state.executed_trades
            self.statistics.successful_trades = state.successful_trades
            self.statistics.failed_trades = state.failed_trades
            self.statistics.total_pnl = state.total_pnl
            self._daily_stop_count = state.daily_stop_count
            
    async def _save_bot_state(self):
        """Bot holatini saqlash"""
        if not self.db_manager:
            return
            
        state = BotState(
            status=self.status.name,
            signal_mode=self.signal_mode.name,
            auto_trading=self._auto_trading_enabled,
            total_signals=self.statistics.total_signals,
            executed_trades=self.statistics.executed_trades,
            successful_trades=self.statistics.successful_trades,
            failed_trades=self.statistics.failed_trades,
            total_pnl=self.statistics.total_pnl,
            daily_stop_count=self._daily_stop_count,
            last_update=TimeUtils.now_uzb()
        )
        
        await self.db_manager.save_bot_state(state)
        
    def get_status(self) -> Dict[str, Any]:
        """Bot holatini olish"""
        return {
            "status": self.status.name,
            "signal_mode": self.signal_mode.name,
            "trading_mode": self.trading_mode.name,
            "auto_trading": self._auto_trading_enabled,
            "statistics": {
                "start_time": self.statistics.start_time.isoformat(),
                "total_signals": self.statistics.total_signals,
                "executed_trades": self.statistics.executed_trades,
                "win_rate": f"{self.statistics.win_rate:.1f}%",
                "total_pnl": f"{self.statistics.total_pnl:+.2f}%",
                "errors": self.statistics.errors_count
            },
            "components": list(self._components.keys()),
            "health": "HEALTHY" if self.status == BotStatus.RUNNING else "UNHEALTHY"
        }
        
    def register_shutdown_handler(self, handler: Callable):
        """Shutdown handler qo'shish"""
        self._shutdown_handlers.append(handler)

# Global instance
bot_manager = BotManager()
