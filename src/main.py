"""
SignalBot - Professional AI Crypto Trading Bot
Asosiy ishga tushirish fayli with advanced error handling
"""
import asyncio
import sys
import signal
import traceback
from typing import Optional, Dict, Any, List
import argparse
import platform
import logging
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import gc

# Custom exceptions
class SignalBotError(Exception):
    """Base SignalBot exception"""
    pass

class ConfigurationError(SignalBotError):
    """Configuration related errors"""
    pass

class ComponentInitializationError(SignalBotError):
    """Component initialization errors"""
    pass

class DatabaseConnectionError(SignalBotError):
    """Database connection errors"""
    pass

class APIConnectionError(SignalBotError):
    """API connection errors"""
    pass

class TelegramError(SignalBotError):
    """Telegram related errors"""
    pass

class TradingError(SignalBotError):
    """Trading operation errors"""
    pass

# Import with specific error handling
try:
    from config.config import config_manager, TradingMode, ConfigValidationError
except ImportError as e:
    print(f"❌ Critical: Config module import failed: {e}")
    sys.exit(1)

try:
    from utils.logger import get_logger, setup_logging
    from utils.helpers import TimeUtils, PerformanceMonitor
    from utils.database import database_manager, DatabaseManager
except ImportError as e:
    print(f"❌ Critical: Utils module import failed: {e}")
    sys.exit(1)

# ASCII Art for startup
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      ░██████╗██╗░██████╗░███╗░░██╗░█████╗░██╗░░░░░  ██████╗░░█████╗░████████╗ ║
║      ██╔════╝██║██╔════╝░████╗░██║██╔══██╗██║░░░░░  ██╔══██╗██╔══██╗╚══██╔══╝ ║
║      ╚█████╗░██║██║░░██╗░██╔██╗██║███████║██║░░░░░  ██████╦╝██║░░██║░░░██║░░░ ║
║      ░╚═══██╗██║██║░░╚██╗██║╚████║██╔══██║██║░░░░░  ██╔══██╗██║░░██║░░░██║░░░ ║
║      ██████╔╝██║╚██████╔╝██║░╚███║██║░░██║███████╗  ██████╦╝╚█████╔╝░░░██║░░░ ║
║      ╚═════╝░╚═╝░╚═════╝░╚═╝░░╚══╝╚═╝░░╚═╝╚══════╝  ╚═════╝░░╚════╝░░░░╚═╝░░░ ║
║                                                                               ║
║                     Professional AI Crypto Trading Bot v2.0                   ║
║                  Powered by ICT, SMT, Order Flow & AI Analysis                ║
║                            Made in Uzbekistan 🇺🇿                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# Global variables
logger: Optional[logging.Logger] = None
performance_monitor: Optional[PerformanceMonitor] = None

class SystemHealth:
    """System health monitoring"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.error_count = 0
        self.warning_count = 0
        self.last_error_time: Optional[datetime] = None
        
    def record_error(self, error: Exception):
        """Error kayit qilish"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
    def record_warning(self):
        """Warning kayit qilish"""
        self.warning_count += 1
        
    def get_uptime(self) -> timedelta:
        """Uptime olish"""
        return datetime.now() - self.start_time
        
    def get_health_status(self) -> Dict[str, Any]:
        """Health status"""
        uptime = self.get_uptime()
        error_rate = self.error_count / max(uptime.total_seconds() / 3600, 1)  # errors per hour
        
        return {
            "uptime_hours": round(uptime.total_seconds() / 3600, 2),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "error_rate_per_hour": round(error_rate, 2),
            "last_error": self.last_error_time.isoformat() if self.last_error_time else None,
            "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": psutil.Process().cpu_percent()
        }

system_health = SystemHealth()

class SignalBot:
    """Asosiy bot klassi with comprehensive error handling"""
    
    def __init__(self):
        self.running = False
        self.components_started = False
        self.shutdown_requested = False
        self.restart_requested = False
        
        # Component references
        self.bot_manager = None
        self.telegram_interface = None
        
        # Error handling
        self.max_consecutive_errors = 5
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        self.error_cooldown_seconds = 60
        
    async def start(self) -> bool:
        """Botni ishga tushirish with error handling"""
        global logger, performance_monitor
        
        startup_start_time = datetime.now()
        
        try:
            # Print banner
            print(BANNER)
            print(f"🕐 Vaqt: {TimeUtils.now_uzb().strftime('%Y-%m-%d %H:%M:%S')} (UZB)")
            print(f"💻 Platform: {platform.system()} {platform.release()}")
            print(f"🐍 Python: {platform.python_version()}")
            print(f"💾 Memory: {psutil.virtual_memory().available // 1024 // 1024} MB available")
            print("-" * 65)
            
            # Setup logging with error handling
            try:
                setup_logging()
                logger = get_logger(__name__)
                logger.info("🚀 SignalBot startup initiated...")
            except Exception as e:
                print(f"❌ Logging setup failed: {e}")
                return False
                
            # Initialize performance monitor
            try:
                performance_monitor = PerformanceMonitor()
                performance_monitor.start_timer("startup")
            except Exception as e:
                logger.warning(f"Performance monitor init failed: {e}")
                
            # Load and validate configuration
            print("📋 Konfiguratsiya yuklanmoqda...")
            try:
                if not await self._load_configuration():
                    raise ConfigurationError("Configuration loading failed")
                logger.info(f"✅ Configuration loaded. Trading mode: {config_manager.trading.mode.name}")
            except ConfigValidationError as e:
                logger.error(f"❌ Configuration validation failed: {e}")
                return False
            except Exception as e:
                logger.error(f"❌ Configuration error: {e}")
                return False
                
            # Initialize database
            print("🗄️ Database sozlanmoqda...")
            try:
                if not await self._initialize_database():
                    raise DatabaseConnectionError("Database initialization failed")
            except DatabaseConnectionError as e:
                logger.error(f"❌ Database error: {e}")
                return False
            except Exception as e:
                logger.error(f"❌ Unexpected database error: {e}")
                return False
                
            # Initialize components
            print("🔧 Komponentlar sozlanmoqda...")
            try:
                if not await self._initialize_components():
                    raise ComponentInitializationError("Component initialization failed")
            except ComponentInitializationError as e:
                logger.error(f"❌ Component error: {e}")
                return False
            except ImportError as e:
                logger.error(f"❌ Component import error: {e}")
                return False
            except Exception as e:
                logger.error(f"❌ Unexpected component error: {e}")
                return False
                
            # Start bot manager
            print("🤖 Bot Manager ishga tushmoqda...")
            try:
                if not await self._start_bot_manager():
                    raise ComponentInitializationError("Bot Manager failed to start")
            except Exception as e:
                logger.error(f"❌ Bot Manager error: {e}")
                return False
                
            # Mark as successfully started
            self.running = True
            self.components_started = True
            self.consecutive_errors = 0
            
            # Calculate startup time
            startup_time = (datetime.now() - startup_start_time).total_seconds()
            if performance_monitor:
                performance_monitor.stop_timer("startup")
                
            # Success message
            print("\n" + "="*65)
            print("✅ SignalBot muvaffaqiyatli ishga tushdi!")
            print("="*65)
            print(f"\n📊 Trading Mode: {config_manager.trading.mode.name}")
            print(f"💱 Symbols: {', '.join(config_manager.trading.symbols[:5])}{'...' if len(config_manager.trading.symbols) > 5 else ''}")
            print(f"⚖️ Risk: {config_manager.trading.risk_percentage}%")
            print(f"👥 Authorized Users: {len(config_manager.telegram.admin_ids)}")
            print(f"⏱️ Startup Time: {startup_time:.2f}s")
            print("\n💡 Bot to'xtatish uchun Ctrl+C bosing")
            print("-" * 65 + "\n")
            
            # Send startup notification
            await self._send_startup_notification(startup_time)
            
            # Keep running with error handling
            await self._run_forever()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("❌ Startup interrupted by user")
            return False
        except SignalBotError as e:
            logger.error(f"❌ SignalBot specific error: {e}")
            system_health.record_error(e)
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected startup error: {e}", exc_info=True)
            system_health.record_error(e)
            return False
            
    async def _load_configuration(self) -> bool:
        """Konfiguratsiya yuklash with validation"""
        try:
            success = await config_manager.load_config()
            if not success:
                validation_errors = config_manager.get_validation_errors()
                for error in validation_errors:
                    logger.error(f"Config validation: {error}")
                return False
                
            # Log configuration summary
            logger.info("Configuration Summary:")
            logger.info(f"  - Trading Mode: {config_manager.trading.mode.name}")
            logger.info(f"  - Symbols: {len(config_manager.trading.symbols)}")
            logger.info(f"  - Risk: {config_manager.trading.risk_percentage}%")
            logger.info(f"  - API Providers: {len(config_manager._api_configs)}")
            
            return True
            
        except ConfigValidationError:
            raise  # Re-raise validation errors
        except FileNotFoundError as e:
            raise ConfigurationError(f"Configuration file not found: {e}")
        except PermissionError as e:
            raise ConfigurationError(f"Permission denied accessing config: {e}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected configuration error: {e}")
            
    async def _initialize_database(self) -> bool:
        """Database ni sozlash"""
        try:
            success = await database_manager.initialize()
            if not success:
                return False
                
            # Test database connection
            test_data = await database_manager.get_bot_state()
            logger.info(f"✅ Database connected. Bot state: {'Found' if test_data else 'New'}")
            
            return True
            
        except ConnectionError as e:
            raise DatabaseConnectionError(f"Database connection failed: {e}")
        except PermissionError as e:
            raise DatabaseConnectionError(f"Database permission denied: {e}")
        except Exception as e:
            raise DatabaseConnectionError(f"Database initialization error: {e}")
            
    async def _initialize_components(self) -> bool:
        """Komponentlarni sozlash with specific error handling"""
        components = [
            ("Risk Manager", self._init_risk_manager),
            ("Analyzers", self._init_analyzers),
            ("Trading Components", self._init_trading),
            ("Telegram Interface", self._init_telegram),
        ]
        
        initialized_components = []
        
        try:
            for name, init_func in components:
                logger.info(f"Initializing {name}...")
                try:
                    if not await init_func():
                        raise ComponentInitializationError(f"{name} initialization failed")
                    initialized_components.append(name)
                    logger.info(f"✅ {name} initialized successfully")
                except ImportError as e:
                    raise ComponentInitializationError(f"{name} import error: {e}")
                except AttributeError as e:
                    raise ComponentInitializationError(f"{name} attribute error: {e}")
                except Exception as e:
                    raise ComponentInitializationError(f"{name} unexpected error: {e}")
                    
            logger.info(f"✅ All {len(initialized_components)} components initialized")
            return True
            
        except ComponentInitializationError:
            # Cleanup any partially initialized components
            logger.error(f"❌ Component initialization failed. Initialized: {initialized_components}")
            await self._cleanup_components(initialized_components)
            raise
            
    async def _init_risk_manager(self) -> bool:
        """Risk manager ni sozlash"""
        try:
            from core.risk_manager import risk_manager
            await risk_manager.initialize()
            return True
        except ImportError as e:
            logger.error(f"Risk manager import error: {e}")
            return False
        except AttributeError as e:
            logger.error(f"Risk manager attribute error: {e}")
            return False
        except Exception as e:
            logger.error(f"Risk manager initialization error: {e}")
            return False
            
    async def _init_analyzers(self) -> bool:
        """Tahlilchilarni sozlash"""
        analyzers = [
            ("ICT Analyzer", "analysis.ict_analyzer", "ict_analyzer"),
            ("SMT Analyzer", "analysis.smt_analyzer", "smt_analyzer"),
            ("Order Flow", "analysis.order_flow", "order_flow_analyzer"),
            ("Sentiment", "analysis.sentiment", "sentiment_analyzer")
        ]
        
        for name, module_name, analyzer_name in analyzers:
            try:
                module = __import__(module_name, fromlist=[analyzer_name])
                analyzer = getattr(module, analyzer_name)
                await analyzer.start()
                logger.debug(f"✅ {name} started")
            except ImportError as e:
                logger.error(f"{name} import error: {e}")
                return False
            except AttributeError as e:
                logger.error(f"{name} attribute error: {e}")
                return False
            except Exception as e:
                logger.error(f"{name} start error: {e}")
                return False
                
        return True
        
    async def _init_trading(self) -> bool:
        """Trading komponentlarini sozlash"""
        trading_components = [
            ("Signal Generator", "trading.signal_generator", "signal_generator"),
            ("Trade Analyzer", "trading.trade_analyzer", "trade_analyzer"),
            ("Execution Engine", "trading.execution_engine", "execution_engine")
        ]
        
        for name, module_name, component_name in trading_components:
            try:
                module = __import__(module_name, fromlist=[component_name])
                component = getattr(module, component_name)
                await component.start()
                logger.debug(f"✅ {name} started")
            except ImportError as e:
                logger.error(f"{name} import error: {e}")
                return False
            except AttributeError as e:
                logger.error(f"{name} attribute error: {e}")
                return False
            except Exception as e:
                logger.error(f"{name} start error: {e}")
                return False
                
        return True
        
    async def _init_telegram(self) -> bool:
        """Telegram interfaceini sozlash"""
        try:
            from telegram.bot_interface import telegram_interface
            self.telegram_interface = telegram_interface
            await telegram_interface.start()
            return True
        except ImportError as e:
            raise TelegramError(f"Telegram import error: {e}")
        except Exception as e:
            raise TelegramError(f"Telegram initialization error: {e}")
            
    async def _start_bot_manager(self) -> bool:
        """Bot manager ni ishga tushirish"""
        try:
            from core.bot_manager import bot_manager
            self.bot_manager = bot_manager
            success = await bot_manager.start()
            return success
        except ImportError as e:
            logger.error(f"Bot manager import error: {e}")
            return False
        except Exception as e:
            logger.error(f"Bot manager start error: {e}")
            return False
            
    async def _send_startup_notification(self, startup_time: float):
        """Startup notification yuborish"""
        try:
            if self.telegram_interface:
                health = system_health.get_health_status()
                message = (
                    f"🚀 <b>SignalBot muvaffaqiyatli ishga tushdi!</b>\n\n"
                    f"📊 Mode: {config_manager.trading.mode.name}\n"
                    f"⚖️ Risk: {config_manager.trading.risk_percentage}%\n"
                    f"💱 Symbols: {len(config_manager.trading.symbols)}\n"
                    f"⏱️ Startup: {startup_time:.2f}s\n"
                    f"💾 Memory: {health['memory_usage_mb']} MB\n\n"
                    f"Bot tayyor! /help - yordam uchun"
                )
                await self.telegram_interface.broadcast_message(message)
        except Exception as e:
            logger.warning(f"Startup notification failed: {e}")
            
    async def _run_forever(self):
        """Doimiy ishlash with error recovery"""
        try:
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitoring_loop())
            
            # Main loop
            while self.running and not self.shutdown_requested:
                try:
                    # Check for restart request
                    if self.restart_requested:
                        logger.info("🔄 Restart requested")
                        break
                        
                    # Reset consecutive errors if enough time passed
                    if (self.last_error_time and 
                        datetime.now() - self.last_error_time > timedelta(seconds=self.error_cooldown_seconds)):
                        self.consecutive_errors = 0
                        
                    # Memory cleanup
                    if datetime.now().minute % 15 == 0:  # Every 15 minutes
                        gc.collect()
                        
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled")
                    break
                except Exception as e:
                    await self._handle_runtime_error(e)
                    
            # Cleanup
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logger.error(f"❌ Fatal error in main loop: {e}", exc_info=True)
            system_health.record_error(e)
            
    async def _handle_runtime_error(self, error: Exception):
        """Runtime error handling"""
        self.consecutive_errors += 1
        self.last_error_time = datetime.now()
        system_health.record_error(error)
        
        logger.error(f"❌ Runtime error ({self.consecutive_errors}/{self.max_consecutive_errors}): {error}")
        
        # If too many consecutive errors, shut down
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical(f"❌ Too many consecutive errors ({self.consecutive_errors}). Shutting down...")
            
            try:
                if self.telegram_interface:
                    await self.telegram_interface.broadcast_message(
                        f"🚨 <b>CRITICAL ERROR</b>\n\n"
                        f"Too many consecutive errors: {self.consecutive_errors}\n"
                        f"Bot shutting down for safety.\n\n"
                        f"Last error: {str(error)[:100]}..."
                    )
            except:
                pass  # Don't let notification failure prevent shutdown
                
            await self.stop()
        else:
            # Wait before continuing
            await asyncio.sleep(min(self.consecutive_errors * 5, 30))
            
    async def _health_monitoring_loop(self):
        """System health monitoring"""
        while self.running:
            try:
                health = system_health.get_health_status()
                
                # Check memory usage
                if health['memory_usage_mb'] > 500:  # 500MB threshold
                    logger.warning(f"High memory usage: {health['memory_usage_mb']} MB")
                    gc.collect()
                    
                # Check error rate
                if health['error_rate_per_hour'] > 10:  # 10 errors per hour
                    logger.warning(f"High error rate: {health['error_rate_per_hour']} errors/hour")
                    
                # Log health status every hour
                if datetime.now().minute == 0:
                    logger.info(f"Health Status: {health}")
                    
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(300)
                
    async def _cleanup_components(self, initialized_components: List[str]):
        """Komponentlarni tozalash"""
        for component_name in reversed(initialized_components):
            try:
                logger.info(f"Cleaning up {component_name}...")
                # Component-specific cleanup logic here
                await asyncio.sleep(0.1)  # Small delay
            except Exception as e:
                logger.error(f"Cleanup error for {component_name}: {e}")
                
    async def stop(self) -> bool:
        """Botni to'xtatish with graceful shutdown"""
        logger.info("🛑 SignalBot shutdown initiated...")
        
        self.running = False
        self.shutdown_requested = True
        
        # Send shutdown notification
        try:
            if self.telegram_interface:
                await self.telegram_interface.broadcast_message(
                    "🛑 <b>SignalBot to'xtatilmoqda...</b>\n\n"
                    "Barcha pozitsiyalar saqlanadi.\n"
                    "Bot tez orada qayta ishga tushadi."
                )
        except Exception as e:
            logger.warning(f"Shutdown notification failed: {e}")
            
        # Stop components gracefully
        if self.components_started and self.bot_manager:
            try:
                await self.bot_manager.shutdown()
                logger.info("✅ Bot manager stopped")
            except Exception as e:
                logger.error(f"Bot manager shutdown error: {e}")
                
        # Close database connections
        try:
            await database_manager.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"Database close error: {e}")
            
        # Final health report
        final_health = system_health.get_health_status()
        logger.info(f"Final Health Report: {final_health}")
        
        logger.info("✅ SignalBot shutdown completed")
        return True
        
    def setup_signal_handlers(self):
        """Signal handlerlarni o'rnatish"""
        def signal_handler(sig, frame):
            signal_name = signal.Signals(sig).name
            logger.info(f"Signal {signal_name} received")
            
            if sig == signal.SIGTERM:
                self.shutdown_requested = True
            elif sig == signal.SIGINT:
                self.shutdown_requested = True
            elif sig == signal.SIGUSR1:  # Custom restart signal
                self.restart_requested = True
                
            # Create task for graceful shutdown
            asyncio.create_task(self.stop())
            
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # SIGUSR1 for restart (Unix only)
        if platform.system() != 'Windows':
            signal.signal(signal.SIGUSR1, signal_handler)

def parse_arguments():
    """Command line argumentlarni parsing qilish"""
    parser = argparse.ArgumentParser(
        description="SignalBot - Professional AI Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode signal              # Signal-only mode
  python main.py --mode paper --debug       # Paper trading with debug
  python main.py --mode live --symbols BTCUSDT,ETHUSDT  # Live trading specific pairs
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "signal", "backtest"],
        default="signal",
        help="Trading mode (default: signal)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.json",
        help="Config file path (default: config/settings.json)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)"
    )
    
    parser.add_argument(
        "--risk",
        type=float,
        help="Risk percentage per trade (0.1-2.0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual trading (paper mode)"
    )
    
    return parser.parse_args()

async def main():
    """Asosiy funksiya with comprehensive error handling"""
    exit_code = 0
    bot = None
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if args.risk and (args.risk < 0.1 or args.risk > 2.0):
            print("❌ Risk must be between 0.1 and 2.0")
            sys.exit(1)
            
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            if not all(s.endswith('USDT') for s in symbols):
                print("❌ All symbols must end with USDT")
                sys.exit(1)
                
        # Set trading mode
        if args.mode:
            mode_map = {
                "live": TradingMode.LIVE,
                "paper": TradingMode.PAPER,
                "signal": TradingMode.SIGNAL_ONLY,
                "backtest": TradingMode.BACKTEST
            }
            # Will be set after config loading
            
        # Create and configure bot
        bot = SignalBot()
        bot.setup_signal_handlers()
        
        # Override config with command line args (after config loading)
        # This will be handled in the start method
        
        # Start bot with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = await bot.start()
                if success:
                    break
                else:
                    logger.error(f"Bot start failed, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)  # Wait before retry
            except KeyboardInterrupt:
                logger.info("Startup interrupted by user")
                break
            except Exception as e:
                logger.error(f"Startup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)  # Wait longer before retry
        else:
            logger.error("❌ Bot failed to start after all attempts")
            exit_code = 1
            
    except KeyboardInterrupt:
        logger.info("❌ Interrupted by user during argument parsing")
        exit_code = 1
    except SystemExit as e:
        exit_code = e.code
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        if logger:
            logger.critical(f"Critical error in main: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Cleanup
        if bot:
            try:
                await bot.stop()
            except Exception as e:
                if logger:
                    logger.error(f"Final cleanup error: {e}")
                else:
                    print(f"Final cleanup error: {e}")
                    
        # Final message
        if exit_code == 0:
            print("\n👋 SignalBot to'xtatildi. Xayr!")
        else:
            print(f"\n❌ SignalBot error bilan to'xtatildi (exit code: {exit_code})")
            
        sys.exit(exit_code)

if __name__ == "__main__":
    # Windows event loop policy
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # Set process priority for better performance
    try:
        import os
        if hasattr(os, 'nice'):
            os.nice(-5)  # Higher priority on Unix systems
    except:
        pass
        
    # Run bot with proper error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 SignalBot to'xtatildi. Xayr!")
    except Exception as e:
        print(f"\n❌ Kritik xatolik: {e}")
        sys.exit(1)
