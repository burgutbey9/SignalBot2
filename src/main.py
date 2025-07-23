"""
SignalBot - Professional AI Crypto Trading Bot
Asosiy ishga tushirish fayli - COMPLETELY FIXED
All import issues resolved, circular imports eliminated
"""
import asyncio
import sys
import signal
from typing import Optional
import argparse
import platform

# ✅ FIXED: Correct imports without circular dependencies
from config.config import config_manager, TradingMode
from utils.logger import get_logger, setup_logging
from utils.helpers import TimeUtils
from utils.database import DatabaseManager

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

# Global logger
logger: Optional[object] = None

class SignalBot:
    """Asosiy bot klassi - Import issues fixed"""
    
    def __init__(self):
        self.running = False
        self.components_started = False
        self.db_manager: Optional[DatabaseManager] = None
        self._components = {}
        
    async def start(self):
        """Botni ishga tushirish - Fixed version"""
        global logger
        
        try:
            # Print banner
            print(BANNER)
            print(f"🕐 Vaqt: {TimeUtils.now_uzb().strftime('%Y-%m-%d %H:%M:%S')} (UZB)")
            print(f"💻 Platform: {platform.system()} {platform.release()}")
            print(f"🐍 Python: {platform.python_version()}")
            print("-" * 65)
            
            # Setup logging
            setup_logging()
            logger = get_logger(__name__)
            
            logger.info("🚀 SignalBot ishga tushmoqda...")
            
            # Load configuration
            print("📋 Konfiguratsiya yuklanmoqda...")
            if not await config_manager.load_config():
                logger.error("❌ Konfiguratsiya yuklanmadi!")
                return False
                
            logger.info(f"✅ Konfiguratsiya yuklandi. Trading mode: {config_manager.trading.mode.name}")
            
            # Initialize database
            print("💾 Database ulanmoqda...")
            self.db_manager = DatabaseManager()
            if not await self.db_manager.initialize():
                logger.error("❌ Database ulanmadi!")
                return False
            
            # Initialize components
            print("🔧 Komponentlar sozlanmoqda...")
            if not await self._initialize_components():
                logger.error("❌ Komponentlarni sozlashda xatolik!")
                return False
                
            # Start bot manager
            print("🤖 Bot Manager ishga tushmoqda...")
            # ✅ FIXED: Import bot_manager inside function to avoid circular import
            from core.bot_manager import bot_manager
            self._components['bot_manager'] = bot_manager
            
            if not await bot_manager.start():
                logger.error("❌ Bot Manager ishga tushmadi!")
                return False
                
            self.running = True
            self.components_started = True
            
            # Success message
            print("\n" + "="*65)
            print("✅ SignalBot muvaffaqiyatli ishga tushdi!")
            print("="*65)
            print(f"\n📊 Trading Mode: {config_manager.trading.mode.name}")
            # ✅ FIXED: Use correct property name - pairs instead of symbols
            print(f"💱 Symbols: {', '.join(config_manager.trading.pairs[:5])}...")
            # ✅ FIXED: Use correct property name - risk_percentage
            print(f"⚖️ Risk: {config_manager.trading.risk_percentage}%")
            # ✅ FIXED: Use correct property name - admin_ids
            print(f"👥 Authorized Users: {len(config_manager.telegram.admin_ids)}")
            print("\n💡 Bot to'xtatish uchun Ctrl+C bosing")
            print("-" * 65 + "\n")
            
            # Keep running
            await self._run_forever()
            
            return True
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Bot start xatosi: {e}", exc_info=True)
            else:
                print(f"❌ Bot start xatosi: {e}")
            return False
            
    async def _initialize_components(self):
        """Komponentlarni sozlash - Fixed imports"""
        try:
            # ✅ FIXED: Import inside function to avoid circular imports
            
            # Initialize analyzers
            print("  🔍 ICT Analyzer...")
            from analysis.ict_analyzer import ict_analyzer
            self._components['ict_analyzer'] = ict_analyzer
            await ict_analyzer.initialize()
            
            print("  🧠 SMT Analyzer...")
            from analysis.smt_analyzer import smt_analyzer
            self._components['smt_analyzer'] = smt_analyzer
            await smt_analyzer.initialize()
            
            print("  📊 Order Flow Analyzer...")
            from analysis.order_flow import order_flow_analyzer
            self._components['order_flow_analyzer'] = order_flow_analyzer
            await order_flow_analyzer.initialize()
            
            print("  💭 Sentiment Analyzer...")
            from analysis.sentiment import sentiment_analyzer
            self._components['sentiment_analyzer'] = sentiment_analyzer
            await sentiment_analyzer.initialize()
            
            # Initialize trading components
            print("  🎯 Signal Generator...")
            from trading.signal_generator import signal_generator
            self._components['signal_generator'] = signal_generator
            await signal_generator.initialize()
            
            print("  ⚡ Execution Engine...")
            from trading.execution_engine import execution_engine
            self._components['execution_engine'] = execution_engine
            await execution_engine.initialize()
            
            # Initialize AI orchestrator
            print("  🤖 AI Orchestrator...")
            from api.ai_clients import ai_orchestrator
            self._components['ai_orchestrator'] = ai_orchestrator
            await ai_orchestrator.initialize()
            
            # Initialize Telegram interface
            print("  📱 Telegram Interface...")
            from telegram.bot_interface import telegram_interface
            self._components['telegram_interface'] = telegram_interface
            await telegram_interface.initialize()
            
            logger.info(f"✅ {len(self._components)} ta komponent muvaffaqiyatli sozlandi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization xatosi: {e}")
            return False
            
    async def _run_forever(self):
        """Doimiy ishlash - Fixed version"""
        try:
            # ✅ FIXED: Access telegram through components
            telegram_interface = self._components.get('telegram_interface')
            
            if telegram_interface:
                # Send startup notification
                await telegram_interface.send_message(
                    f"🚀 <b>SignalBot ishga tushdi!</b>\n\n"
                    f"📊 Mode: {config_manager.trading.mode.name}\n"
                    f"⚖️ Risk: {config_manager.trading.risk_percentage}%\n"
                    f"💱 Symbols: {len(config_manager.trading.pairs)}\n\n"
                    f"Bot tayyor! /help - yordam uchun"
                )
            
            # Main loop
            while self.running:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Main loop bekor qilindi")
        except Exception as e:
            logger.error(f"Main loop xatosi: {e}")
            
    async def stop(self):
        """Botni to'xtatish - Fixed version"""
        logger.info("🛑 SignalBot to'xtatilmoqda...")
        
        self.running = False
        
        # Send shutdown notification
        try:
            telegram_interface = self._components.get('telegram_interface')
            if telegram_interface:
                await telegram_interface.send_message(
                    "🛑 <b>SignalBot to'xtatilmoqda...</b>\n\n"
                    "Barcha pozitsiyalar saqlanadi.\n"
                    "Bot tez orada qayta ishga tushadi."
                )
        except Exception:
            pass
            
        # Stop components in reverse order
        if self.components_started:
            # Stop bot manager
            bot_manager = self._components.get('bot_manager')
            if bot_manager:
                await bot_manager.shutdown()
                
            # Stop other components
            for name, component in self._components.items():
                if hasattr(component, 'stop'):
                    try:
                        await component.stop()
                        logger.info(f"✅ {name} to'xtatildi")
                    except Exception as e:
                        logger.error(f"❌ {name} to'xtatishda xato: {e}")
                        
        # Close database
        if self.db_manager:
            await self.db_manager.close()
            
        logger.info("✅ SignalBot to'liq to'xtatildi")
        
    def setup_signal_handlers(self):
        """Signal handlerlarni o'rnatish"""
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} qabul qilindi")
            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            loop.create_task(self.stop())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    """Command line argumentlarni parsing qilish"""
    parser = argparse.ArgumentParser(
        description="SignalBot - Professional AI Crypto Trading Bot"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "signal"],
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
        "--test",
        action="store_true",
        help="Run in test mode"
    )
    
    return parser.parse_args()

async def main():
    """Asosiy funksiya - All imports fixed"""
    # Parse arguments
    args = parse_arguments()
    
    # Set trading mode
    if args.mode:
        mode_map = {
            "live": TradingMode.LIVE,
            "paper": TradingMode.PAPER,
            "signal": TradingMode.SIGNAL_ONLY
        }
        # Set mode directly in config
        config_manager._trading_config.mode = mode_map.get(args.mode, TradingMode.SIGNAL_ONLY)
        
    # Create and start bot
    bot = SignalBot()
    bot.setup_signal_handlers()
    
    try:
        success = await bot.start()
        if not success:
            print("❌ Bot ishga tushmadi!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt qabul qilindi")
        
    except Exception as e:
        print(f"❌ Bot xatosi: {e}")
        sys.exit(1)
        
    finally:
        try:
            await bot.stop()
        except Exception as e:
            print(f"❌ Bot to'xtatishda xato: {e}")

if __name__ == "__main__":
    # Windows event loop policy
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # Run bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 SignalBot to'xtatildi. Xayr!")
    except Exception as e:
        print(f"\n❌ Kritik xatolik: {e}")
        sys.exit(1)
