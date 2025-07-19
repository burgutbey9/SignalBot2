"""
SignalBot - Professional AI Crypto Trading Bot
Asosiy ishga tushirish fayli
"""
import asyncio
import sys
import signal
from typing import Optional
import argparse
import platform

from config.config import config_manager, TradingMode
from utils.logger import get_logger, setup_logging
from utils.helpers import TimeUtils
from core.bot_manager import bot_manager
from telegram.bot_interface import telegram_interface

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

logger: Optional[get_logger] = None

class SignalBot:
    """Asosiy bot klassi"""
    
    def __init__(self):
        self.running = False
        self.components_started = False
        
    async def start(self):
        """Botni ishga tushirish"""
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
            
            # Initialize components
            print("🔧 Komponentlar sozlanmoqda...")
            if not await self._initialize_components():
                logger.error("❌ Komponentlarni sozlashda xatolik!")
                return False
                
            # Start bot manager
            print("🤖 Bot Manager ishga tushmoqda...")
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
            print(f"💱 Symbols: {', '.join(config_manager.trading.symbols[:5])}...")
            print(f"⚖️ Risk: {config_manager.risk_management.base_risk_percent}%")
            print(f"👥 Authorized Users: {len(config_manager.telegram.authorized_users)}")
            print("\n💡 Bot to'xtatish uchun Ctrl+C bosing")
            print("-" * 65 + "\n")
            
            # Keep running
            await self._run_forever()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot start xatosi: {e}", exc_info=True)
            return False
            
    async def _initialize_components(self):
        """Komponentlarni sozlash"""
        try:
            # Initialize in correct order
            components = [
                ("Risk Manager", self._init_risk_manager),
                ("Analyzers", self._init_analyzers),
                ("Trading Components", self._init_trading),
                ("Telegram Interface", self._init_telegram),
            ]
            
            for name, init_func in components:
                logger.info(f"Initializing {name}...")
                if not await init_func():
                    logger.error(f"Failed to initialize {name}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Component initialization xatosi: {e}")
            return False
            
    async def _init_risk_manager(self):
        """Risk manager ni sozlash"""
        try:
            from core.risk_manager import risk_manager
            await risk_manager.initialize()
            logger.info("✅ Risk Manager tayyor")
            return True
        except Exception as e:
            logger.error(f"Risk Manager init xatosi: {e}")
            return False
            
    async def _init_analyzers(self):
        """Tahlilchilarni sozlash"""
        try:
            from analysis.ict_analyzer import ict_analyzer
            from analysis.smt_analyzer import smt_analyzer
            from analysis.order_flow import order_flow_analyzer
            from analysis.sentiment import sentiment_analyzer
            
            # Start analyzers
            await ict_analyzer.start()
            await smt_analyzer.start()
            await order_flow_analyzer.start()
            await sentiment_analyzer.start()
            
            logger.info("✅ Barcha tahlilchilar tayyor")
            return True
            
        except Exception as e:
            logger.error(f"Analyzers init xatosi: {e}")
            return False
            
    async def _init_trading(self):
        """Trading komponentlarini sozlash"""
        try:
            from trading.signal_generator import signal_generator
            from trading.trade_analyzer import trade_analyzer
            from trading.execution_engine import execution_engine
            
            # Start trading components
            await signal_generator.start()
            await trade_analyzer.start()
            await execution_engine.start()
            
            logger.info("✅ Trading komponentlari tayyor")
            return True
            
        except Exception as e:
            logger.error(f"Trading components init xatosi: {e}")
            return False
            
    async def _init_telegram(self):
        """Telegram interfaceini sozlash"""
        try:
            await telegram_interface.start()
            logger.info("✅ Telegram interface tayyor")
            return True
            
        except Exception as e:
            logger.error(f"Telegram interface init xatosi: {e}")
            return False
            
    async def _run_forever(self):
        """Doimiy ishlash"""
        try:
            # Send startup notification
            await telegram_interface.broadcast_message(
                f"🚀 <b>SignalBot ishga tushdi!</b>\n\n"
                f"📊 Mode: {config_manager.trading.mode.name}\n"
                f"⚖️ Risk: {config_manager.risk_management.base_risk_percent}%\n"
                f"💱 Symbols: {len(config_manager.trading.symbols)}\n\n"
                f"Bot tayyor! /help - yordam uchun"
            )
            
            # Main loop
            while self.running:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Main loop bekor qilindi")
            
    async def stop(self):
        """Botni to'xtatish"""
        logger.info("🛑 SignalBot to'xtatilmoqda...")
        
        self.running = False
        
        # Send shutdown notification
        try:
            await telegram_interface.broadcast_message(
                "🛑 <b>SignalBot to'xtatilmoqda...</b>\n\n"
                "Barcha pozitsiyalar saqlanadi.\n"
                "Bot tez orada qayta ishga tushadi."
            )
        except:
            pass
            
        # Stop components in reverse order
        if self.components_started:
            await bot_manager.shutdown()
            
        logger.info("✅ SignalBot to'xtatildi")
        
    def setup_signal_handlers(self):
        """Signal handlerlarni o'rnatish"""
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} qabul qilindi")
            asyncio.create_task(self.stop())
            
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
    """Asosiy funksiya"""
    # Parse arguments
    args = parse_arguments()
    
    # Set trading mode
    if args.mode:
        mode_map = {
            "live": TradingMode.LIVE,
            "paper": TradingMode.PAPER,
            "signal": TradingMode.SIGNAL_ONLY
        }
        config_manager.trading.mode = mode_map.get(args.mode, TradingMode.SIGNAL_ONLY)
        
    # Create and start bot
    bot = SignalBot()
    bot.setup_signal_handlers()
    
    try:
        success = await bot.start()
        if not success:
            logger.error("Bot ishga tushmadi!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt qabul qilindi")
        
    except Exception as e:
        logger.error(f"Bot xatosi: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        await bot.stop()

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
