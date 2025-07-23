"""
Bot Lifecycle Manager and Signal Controller - FIXED VERSION
Bot ishga tushirish, to'xtatish, signal boshqaruv
All import issues resolved
"""
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import signal
import sys

# ✅ FIXED: Correct imports without circular dependencies
from config.config import config_manager, TradingMode
from utils.logger import get_logger
from utils.helpers import TimeUtils, PerformanceMonitor
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
    """Asosiy bot manager - Fixed imports"""
    def __init__(self):
        self.status = BotStatus.STOPPED
        self.signal_mode = SignalMode.PAUSED
        self.trading_mode = TradingMode.SIGNAL_ONLY
        self.statistics = BotStatistics(start_time=TimeUtils.now_uzb())
        self.performance_monitor = PerformanceMonitor()
        self.db_manager: Optional[DatabaseManager] = None
        
        # Components - ✅ FIXED: Initialize as empty dict
        self._components: Dict[str, Any] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_handlers: List[Callable] = []
        
        # Control flags
        self._running = False
        self._auto_trading_enabled = False
        self._daily_stop_count = 0
        self._last_stop_reset = TimeUtils.now_uzb().date()
        
    async def initialize(self) -> bool:
        """Bot komponentlarini sozlash - Fixed version"""
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
        """Botni ishga tushirish - Fixed version"""
        if self.status == BotStatus.RUNNING:
            logger.warning("Bot allaqachon ishlamoqda")
            return True
            
        try:
            logger.info("🟢 Bot ishga tushmoqda...")
            self.status = BotStatus.STARTING
            
            # Initialize components if not done
            if not await self.initialize():
                return False
                
            # ✅ FIXED: Load components inside function to avoid circular imports
            await self._load_components()
            
            # Start components
            await self._start_components()
            
            # Start monitoring
            self._tasks["monitor"] = asyncio.create_task(self._monitor_loop())
            self._tasks["health_check"] = asyncio.create_task(self._health_check_loop())
            
            self._running = True
            self.status = BotStatus.RUNNING
            self.statistics.start_time = TimeUtils.now_uzb()
            
            # ✅ FIXED: Use message handler for notifications
            await self._send_startup_notification()
            
            logger.info("✅ Bot to'liq ishga tushdi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot start xatosi: {e}")
            self.status = BotStatus.ERROR
            self.statistics.errors_count += 1
            return False
    
    async def _load_components(self):
        """Komponentlarni yuklash - Fixed imports"""
        try:
            # ✅ FIXED: Import inside function to avoid circular imports
            logger.info("📦 Komponentlar yuklanmoqda...")
            
            # Import and store analyzers
            from analysis.ict_analyzer import ict_analyzer
            self._components["ict_analyzer"] = ict_analyzer
            
            from analysis.smt_analyzer import smt_analyzer
            self._components["smt_analyzer"] = smt_analyzer
            
            from analysis.order_flow import order_flow_analyzer
            self._components["order_flow_analyzer"] = order_flow_analyzer
            
            from analysis.sentiment import sentiment_analyzer
            self._components["sentiment_analyzer"] = sentiment_analyzer
            
            # Import trading components
            from trading.signal_generator import signal_generator
            self._components["signal_generator"] = signal_generator
            
            from trading.execution_engine import execution_engine
            self._components["execution_engine"] = execution_engine
            
            # Import AI orchestrator
            from api.ai_clients import ai_orchestrator
            self._components["ai_orchestrator"] = ai_orchestrator
            
            # Import message handler
            from telegram.message_handler import notification_manager
            self._components["notification_manager"] = notification_manager
            
            logger.info(f"✅ {len(self._components)} komponent yuklandi")
            
        except Exception as e:
            logger.error(f"❌ Komponentlarni yuklashda xato: {e}")
            raise
            
    async def _start_components(self):
        """Komponentlarni ishga tushirish - Fixed version"""
        try:
            # Start AI orchestrator first
            ai_orchestrator = self._components.get("ai_orchestrator")
            if ai_orchestrator and hasattr(ai_orchestrator, "initialize"):
                await ai_orchestrator.initialize()
                logger.info("✅ AI Orchestrator ishga tushdi")
            
            # Start analyzers
            analyzer_names = ["ict_analyzer", "smt_analyzer", "order_flow_analyzer", "sentiment_analyzer"]
            for name in analyzer_names:
                component = self._components.get(name)
                if component:
                    # ✅ FIXED: Check if component has start method
                    if hasattr(component, "start"):
                        self._tasks[name] = asyncio.create_task(component.start())
                    elif hasattr(component, "initialize"):
                        await component.initialize()
                    logger.info(f"✅ {name} ishga tushdi")
            
            # Start trading components
            trading_names = ["signal_generator", "execution_engine"]
            for name in trading_names:
                component = self._components.get(name)
                if component:
                    if hasattr(component, "start"):
                        self._tasks[name] = asyncio.create_task(component.start())
                    elif hasattr(component, "initialize"):
                        await component.initialize()
                    logger.info(f"✅ {name} ishga tushdi")
                
            logger.info(f"✅ Barcha komponentlar ishga tushdi")
            
        except Exception as e:
            logger.error(f"❌ Komponentlarni ishga tushirishda xato: {e}")
            raise
            
    async def _send_startup_notification(self):
        """Startup notification yuborish"""
        try:
            notification_manager = self._components.get("notification_manager")
            if notification_manager:
                await notification_manager.notify_system_info({
                    "status": "STARTED",
                    "message": f"🤖 Bot muvaffaqiyatli ishga tushdi!\n\n"
                              f"📊 Mode: {self.trading_mode.value}\n"
                              f"⚖️ Risk: {config_manager.trading.risk_percentage}%\n"
                              f"💱 Symbols: {len(config_manager.trading.pairs)}\n\n"
                              f"Bot tayyor!",
                    "component": "BotManager"
                })
        except Exception as e:
            logger.error(f"Startup notification xatosi: {e}")
            
    async def stop(self) -> bool:
        """Botni to'xtatish - Fixed version"""
        if self.status == BotStatus.STOPPED:
            logger.warning("Bot allaqachon to'xtatilgan")
            return True
            
        try:
            logger.info("🔴 Bot to'xtatilmoqda...")
            self.status = BotStatus.STOPPING
            self._running = False
            
            # Send shutdown notification
            await self._send_shutdown_notification()
            
            # Stop all tasks
            for name, task in self._tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info(f"Task {name} cancelled")
                        
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
            
    async def _send_shutdown_notification(self):
        """Shutdown notification yuborish"""
        try:
            notification_manager = self._components.get("notification_manager")
            if notification_manager:
                await notification_manager.notify_system_info({
                    "status": "STOPPED",
                    "message": "🛑 Bot to'xtatilmoqda...\n\n"
                              "Barcha pozitsiyalar saqlanadi.\n"
                              "Bot tez orada qayta ishga tushadi.",
                    "component": "BotManager"
                })
        except Exception as e:
            logger.error(f"Shutdown notification xatosi: {e}")
            
    async def _stop_components(self):
        """Komponentlarni to'xtatish - Fixed version"""
        for name, component in self._components.items():
            try:
                if hasattr(component, "stop"):
                    await component.stop()
                    logger.info(f"✅ {name} to'xtatildi")
            except Exception as e:
                logger.error(f"❌ Component {name} stop xatosi: {e}")
                
    async def pause(self) -> bool:
        """Botni pause qilish"""
        if self.status != BotStatus.RUNNING:
            logger.warning("Bot ishlamayapti, pause qilib bo'lmaydi")
            return False
            
        self.status = BotStatus.PAUSED
        self.signal_mode = SignalMode.PAUSED
        
        logger.info("⏸️ Bot pause qilindi")
        
        # Send notification
        notification_manager = self._components.get("notification_manager")
        if notification_manager:
            await notification_manager.notify_system_info({
                "status": "PAUSED",
                "message": "⏸️ Bot vaqtincha to'xtatildi",
                "component": "BotManager"
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
        
        # Send notification
        notification_manager = self._components.get("notification_manager")
        if notification_manager:
            await notification_manager.notify_system_info({
                "status": "RESUMED",
                "message": "▶️ Bot yana ishga tushdi",
                "component": "BotManager"
            })
        
        return True
        
    async def shutdown(self):
        """Botni to'liq o'chirish"""
        logger.info("🛑 Bot shutdown boshlandi...")
        
        # Run shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
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
        """Signalni qayta ishlash - Fixed version"""
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
            notification_manager = self._components.get("notification_manager")
            if notification_manager:
                await notification_manager.notify_signal(signal_data)
            
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
                
            # ✅ FIXED: Check if method exists
            if hasattr(execution_engine, "execute_signal"):
                result = await execution_engine.execute_signal(signal_data)
            else:
                logger.warning("execute_signal method mavjud emas")
                return False
            
            if result and result.get("success"):
                self.statistics.successful_trades += 1
            else:
                self.statistics.failed_trades += 1
                
                # Check stop loss
                if result and result.get("reason") == "stop_loss":
                    self._daily_stop_count += 1
                    
            return result.get("success", False) if result else False
            
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
                        try:
                            if not await component.is_healthy():
                                logger.warning(f"Component {name} unhealthy")
                        except Exception as e:
                            logger.error(f"Health check xatosi {name}: {e}")
                            
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
                        notification_manager = self._components.get("notification_manager")
                        if notification_manager:
                            await notification_manager.notify_error({
                                "message": health_status.get('message', 'Unknown critical error'),
                                "critical": True,
                                "component": "HealthCheck"
                            })
                        
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
                try:
                    status = await component.get_health_status()
                    health["components"][name] = status
                    if not status.get("healthy", True):
                        health["healthy"] = False
                except Exception as e:
                    health["components"][name] = {"healthy": False, "error": str(e)}
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
            
        try:
            state = await self.db_manager.get_bot_state()
            if state:
                self.statistics.total_signals = state.total_signals
                self.statistics.executed_trades = state.executed_trades
                self.statistics.successful_trades = state.successful_trades
                self.statistics.failed_trades = state.failed_trades
                self.statistics.total_pnl = state.total_pnl
                self._daily_stop_count = state.daily_stop_count
                logger.info("✅ Bot state yuklandi")
        except Exception as e:
            logger.error(f"Bot state yuklashda xato: {e}")
            
    async def _save_bot_state(self):
        """Bot holatini saqlash"""
        if not self.db_manager:
            return
            
        try:
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
            logger.info("✅ Bot state saqlandi")
        except Exception as e:
            logger.error(f"Bot state saqlashda xato: {e}")
        
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
        
    async def get_performance_report(self) -> Dict[str, Any]:
        """Performance hisoboti"""
        try:
            uptime = (TimeUtils.now_uzb() - self.statistics.start_time).total_seconds()
            
            return {
                "uptime_hours": uptime / 3600,
                "total_signals": self.statistics.total_signals,
                "executed_trades": self.statistics.executed_trades,
                "successful_trades": self.statistics.successful_trades,
                "failed_trades": self.statistics.failed_trades,
                "win_rate": self.statistics.win_rate,
                "total_pnl": self.statistics.total_pnl,
                "average_pnl": self.statistics.average_pnl,
                "daily_stop_count": self._daily_stop_count,
                "errors_count": self.statistics.errors_count,
                "performance_metrics": self.performance_monitor.get_summary()
            }
        except Exception as e:
            logger.error(f"Performance report xatosi: {e}")
            return {}

# ✅ FIXED: Global instance
bot_manager = BotManager()
