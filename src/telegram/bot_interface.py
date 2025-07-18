"""
Telegram Bot Interface
Bot tugmalari, komandalar, foydalanuvchi bilan aloqa
"""
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils
from core.bot_manager import bot_manager
from telegram.message_handler import message_handler

logger = get_logger(__name__)

class UserRole(Enum):
    """Foydalanuvchi rollari"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"

class BotStates(StatesGroup):
    """Bot holatlari"""
    waiting_for_confirmation = State()
    waiting_for_risk_input = State()
    waiting_for_symbol_add = State()
    waiting_for_symbol_remove = State()

@dataclass
class UserSession:
    """Foydalanuvchi sessiyasi"""
    user_id: int
    username: str
    role: UserRole
    last_activity: datetime
    active_signals: List[str] = None
    settings: Dict[str, Any] = None

class TelegramBotInterface:
    """Telegram bot interfeysi"""
    def __init__(self):
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.sessions: Dict[int, UserSession] = {}
        self.authorized_users: List[int] = []
        
    async def start(self):
        """Bot interfaceini ishga tushirish"""
        try:
            logger.info("🤖 Telegram Bot Interface ishga tushmoqda...")
            
            # Initialize bot
            token = config_manager.telegram.bot_token
            self.bot = Bot(token=token, parse_mode="HTML")
            self.dp = Dispatcher()
            
            # Load authorized users
            self.authorized_users = config_manager.telegram.authorized_users
            
            # Register handlers
            self._register_handlers()
            
            # Start polling
            asyncio.create_task(self._start_polling())
            
            logger.info("✅ Telegram Bot Interface tayyor")
            
        except Exception as e:
            logger.error(f"❌ Bot Interface start xatosi: {e}")
            
    def _register_handlers(self):
        """Handlerlarni ro'yxatdan o'tkazish"""
        # Command handlers
        self.dp.message.register(self._handle_start, Command("start"))
        self.dp.message.register(self._handle_help, Command("help"))
        self.dp.message.register(self._handle_status, Command("status"))
        self.dp.message.register(self._handle_stats, Command("stats"))
        self.dp.message.register(self._handle_positions, Command("positions"))
        self.dp.message.register(self._handle_signals, Command("signals"))
        self.dp.message.register(self._handle_risk, Command("risk"))
        self.dp.message.register(self._handle_symbols, Command("symbols"))
        
        # Callback handlers
        self.dp.callback_query.register(self._handle_bot_control, F.data.in_(["start_bot", "stop_bot", "pause_bot"]))
        self.dp.callback_query.register(self._handle_trading_control, F.data.in_(["enable_trading", "disable_trading"]))
        self.dp.callback_query.register(self._handle_signal_action, F.data.startswith("signal_"))
        self.dp.callback_query.register(self._handle_position_action, F.data.startswith("position_"))
        self.dp.callback_query.register(self._handle_risk_action, F.data.startswith("risk_"))
        
        # State handlers
        self.dp.message.register(self._handle_risk_input, StateFilter(BotStates.waiting_for_risk_input))
        self.dp.message.register(self._handle_symbol_add, StateFilter(BotStates.waiting_for_symbol_add))
        self.dp.message.register(self._handle_symbol_remove, StateFilter(BotStates.waiting_for_symbol_remove))
        
        # Text message handler
        self.dp.message.register(self._handle_text_message)
        
    async def _start_polling(self):
        """Polling boshlash"""
        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Polling xatosi: {e}")
            
    async def _handle_start(self, message: Message):
        """Start komandasi"""
        user_id = message.from_user.id
        
        if not self._is_authorized(user_id):
            await message.answer("❌ Sizda botdan foydalanish huquqi yo'q!")
            return
            
        # Create session
        session = UserSession(
            user_id=user_id,
            username=message.from_user.username or "Unknown",
            role=self._get_user_role(user_id),
            last_activity=TimeUtils.now_uzb()
        )
        self.sessions[user_id] = session
        
        # Send welcome message with main menu
        keyboard = self._get_main_keyboard()
        welcome_text = f"""
🤖 <b>SignalBot - Professional Crypto Trading</b>

Assalomu alaykum, {message.from_user.first_name}!

Bot holati: {self._get_bot_status_emoji()} {bot_manager.status.name}
Trading rejimi: {config_manager.trading.mode.name}

Asosiy komandalar:
/status - Bot holati
/signals - Aktiv signallar
/positions - Ochiq pozitsiyalar
/stats - Statistika
/help - Yordam

Quyidagi tugmalardan foydalaning 👇
"""
        
        await message.answer(welcome_text, reply_markup=keyboard)
        
    async def _handle_help(self, message: Message):
        """Yordam komandasi"""
        if not self._is_authorized(message.from_user.id):
            return
            
        help_text = """
📚 <b>Bot Komandalar</b>

<b>Asosiy:</b>
/start - Botni ishga tushirish
/status - Bot holati va sozlamalari
/help - Ushbu yordam

<b>Trading:</b>
/signals - Aktiv signallar ro'yxati
/positions - Ochiq pozitsiyalar
/stats - Trading statistikasi
/risk - Risk sozlamalari

<b>Boshqaruv:</b>
/symbols - Trading juftliklari
/settings - Bot sozlamalari

<b>Tugmalar:</b>
🟢 Bot Ishga Tushirish - Botni aktivlashtirish
🔴 Bot To'xtatish - Botni to'xtatish
⏸️ Pauza - Vaqtincha to'xtatish

💰 Avto Savdo - Trading yoqish/o'chirish
📊 Risk - Risk foizini o'zgartirish

<b>Signal Tugmalari:</b>
✅ Avto Savdo - Signalni avtomatik bajarish
❌ Bekor Qilish - Signalni rad etish

❓ Savollar bo'lsa @admin ga murojaat qiling
"""
        
        await message.answer(help_text)
        
    async def _handle_status(self, message: Message):
        """Status komandasi"""
        if not self._is_authorized(message.from_user.id):
            return
            
        status = bot_manager.get_status()
        
        status_text = f"""
📊 <b>Bot Status</b>

🤖 Bot: {self._get_bot_status_emoji()} {status['status']}
📡 Signal Mode: {status['signal_mode']}
💱 Trading Mode: {status['trading_mode']}
💰 Auto Trading: {'✅ ON' if status['auto_trading'] else '❌ OFF'}

<b>Statistika:</b>
📈 Jami Signallar: {status['statistics']['total_signals']}
✅ Bajarilgan: {status['statistics']['executed_trades']}
📊 Win Rate: {status['statistics']['win_rate']}
💵 Total PnL: {status['statistics']['total_pnl']}
❌ Xatolar: {status['statistics']['errors']}

<b>Komponentlar:</b>
{', '.join(status['components'])}

🏥 Salomatlik: {status['health']}
⏰ Vaqt: {TimeUtils.now_uzb().strftime('%H:%M:%S')} (UZB)
"""
            
            keyboard = self._get_control_keyboard()
            await message.edit_text(status_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Status update xatosi: {e}")
            
    async def _update_risk_message(self, message: Message):
        """Risk xabarini yangilash"""
        try:
            from core.risk_manager import risk_manager
            
            risk_text = f"""
⚖️ <b>Risk Management</b>

<b>Hozirgi sozlamalar:</b>
📊 Base Risk: {risk_manager.params.base_risk}%
📈 Max Risk: {risk_manager.params.max_risk}%
🔻 Max Daily Loss: {risk_manager.params.max_daily_loss}%
📍 Max Positions: {risk_manager.params.max_positions}

<b>Dinamik Risk:</b>
🎯 Current Risk: {risk_manager._current_risk_percent:.1f}%
📊 Risk Level: {risk_manager._risk_level.name}

Risk foizini o'zgartirish uchun tugmalardan foydalaning:
"""
            
            keyboard = self._get_risk_keyboard()
            await message.edit_text(risk_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Risk update xatosi: {e}")
            
    async def send_notification(self, user_id: int, message: str, 
                              keyboard: Optional[InlineKeyboardMarkup] = None):
        """Foydalanuvchiga xabar yuborish"""
        try:
            if self.bot:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    reply_markup=keyboard,
                    parse_mode="HTML"
                )
        except Exception as e:
            logger.error(f"Notification yuborish xatosi: {e}")
            
    async def broadcast_message(self, message: str, 
                              keyboard: Optional[InlineKeyboardMarkup] = None):
        """Barcha foydalanuvchilarga xabar yuborish"""
        for user_id in self.authorized_users:
            await self.send_notification(user_id, message, keyboard)
            await asyncio.sleep(0.1)  # Rate limiting
            
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Aktiv sessiyalarni olish"""
        active = []
        cutoff_time = TimeUtils.now_uzb() - timedelta(minutes=30)
        
        for user_id, session in self.sessions.items():
            if session.last_activity > cutoff_time:
                active.append({
                    "user_id": user_id,
                    "username": session.username,
                    "role": session.role.value,
                    "last_activity": session.last_activity.isoformat()
                })
                
        return active

# Global instance
telegram_interface = TelegramBotInterface()

🏥 Salomatlik: {status['health']}
⏰ Vaqt: {TimeUtils.now_uzb().strftime('%H:%M:%S')} (UZB)
"""
        
        keyboard = self._get_control_keyboard()
        await message.answer(status_text, reply_markup=keyboard)
        
    async def _handle_stats(self, message: Message):
        """Statistika komandasi"""
        if not self._is_authorized(message.from_user.id):
            return
            
        from trading.signal_generator import signal_generator
        from trading.trade_analyzer import trade_analyzer
        
        # Get statistics
        signal_stats = signal_generator.get_signal_metrics()
        trade_stats = trade_analyzer.get_performance_stats()
        
        stats_text = f"""
📊 <b>Trading Statistika</b>

<b>Signallar:</b>
📡 Jami: {signal_stats['total_signals']}
✅ Muvaffaqiyatli: {signal_stats['successful_signals']}
❌ Muvaffaqiyatsiz: {signal_stats['failed_signals']}
📈 Win Rate: {signal_stats['win_rate']:.1f}%
🎯 Aktiv: {signal_stats['active_signals']}

<b>Trading Performance:</b>
💹 Win Rate: {trade_stats['win_rate']}
📈 O'rtacha Yutuq: {trade_stats['avg_win']}
📉 O'rtacha Zarar: {trade_stats['avg_loss']}
💰 Profit Factor: {trade_stats['profit_factor']}
📊 Sharpe Ratio: {trade_stats['sharpe_ratio']}
🔻 Max Drawdown: {trade_stats['max_drawdown']}

<b>Stop Loss Sabablari:</b>
{self._format_stop_loss_reasons(trade_stats.get('stop_loss_reasons', {}))}

<b>Take Profit Sabablari:</b>
{self._format_take_profit_reasons(trade_stats.get('take_profit_reasons', {}))}
"""
        
        await message.answer(stats_text)
        
    async def _handle_positions(self, message: Message):
        """Ochiq pozitsiyalar"""
        if not self._is_authorized(message.from_user.id):
            return
            
        from trading.execution_engine import execution_engine
        positions = execution_engine.get_open_positions()
        
        if not positions:
            await message.answer("📊 Hozirda ochiq pozitsiyalar yo'q")
            return
            
        for pos in positions:
            pos_text = message_handler.format_position_message(pos)
            keyboard = self._get_position_keyboard(pos['symbol'])
            await message.answer(pos_text, reply_markup=keyboard)
            
    async def _handle_signals(self, message: Message):
        """Aktiv signallar"""
        if not self._is_authorized(message.from_user.id):
            return
            
        from trading.signal_generator import signal_generator
        signals = signal_generator.get_active_signals()
        
        if not signals:
            await message.answer("📡 Hozirda aktiv signallar yo'q")
            return
            
        for signal in signals[:5]:  # Last 5 signals
            signal_text = message_handler.format_signal_message(signal)
            keyboard = self._get_signal_keyboard(signal['symbol'])
            await message.answer(signal_text, reply_markup=keyboard)
            
    async def _handle_risk(self, message: Message, state: FSMContext):
        """Risk sozlamalari"""
        if not self._is_authorized(message.from_user.id):
            return
            
        from core.risk_manager import risk_manager
        
        risk_text = f"""
⚖️ <b>Risk Management</b>

<b>Hozirgi sozlamalar:</b>
📊 Base Risk: {risk_manager.params.base_risk}%
📈 Max Risk: {risk_manager.params.max_risk}%
🔻 Max Daily Loss: {risk_manager.params.max_daily_loss}%
📍 Max Positions: {risk_manager.params.max_positions}

<b>Dinamik Risk:</b>
🎯 Current Risk: {risk_manager._current_risk_percent:.1f}%
📊 Risk Level: {risk_manager._risk_level.name}

Risk foizini o'zgartirish uchun tugmalardan foydalaning:
"""
        
        keyboard = self._get_risk_keyboard()
        await message.answer(risk_text, reply_markup=keyboard)
        
    async def _handle_symbols(self, message: Message, state: FSMContext):
        """Trading juftliklari"""
        if not self._is_authorized(message.from_user.id):
            return
            
        symbols = config_manager.trading.symbols
        
        symbols_text = f"""
💱 <b>Trading Juftliklari</b>

<b>Aktiv ({len(symbols)}):</b>
{chr(10).join(f'• {s}' for s in symbols)}

Juftlik qo'shish: /add_symbol
Juftlik o'chirish: /remove_symbol
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="➕ Qo'shish", callback_data="add_symbol"),
                InlineKeyboardButton(text="➖ O'chirish", callback_data="remove_symbol")
            ]
        ])
        
        await message.answer(symbols_text, reply_markup=keyboard)
        
    async def _handle_bot_control(self, callback: CallbackQuery):
        """Bot boshqaruv tugmalari"""
        user_id = callback.from_user.id
        if not self._is_authorized(user_id):
            await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
            return
            
        action = callback.data
        
        if action == "start_bot":
            success = await bot_manager.start()
            if success:
                await callback.answer("✅ Bot ishga tushdi!")
                await self._update_status_message(callback.message)
            else:
                await callback.answer("❌ Xatolik yuz berdi!", show_alert=True)
                
        elif action == "stop_bot":
            success = await bot_manager.stop()
            if success:
                await callback.answer("🔴 Bot to'xtatildi!")
                await self._update_status_message(callback.message)
            else:
                await callback.answer("❌ Xatolik yuz berdi!", show_alert=True)
                
        elif action == "pause_bot":
            success = await bot_manager.pause()
            if success:
                await callback.answer("⏸️ Bot pause qilindi!")
                await self._update_status_message(callback.message)
            else:
                await callback.answer("❌ Xatolik yuz berdi!", show_alert=True)
                
    async def _handle_trading_control(self, callback: CallbackQuery):
        """Trading boshqaruv tugmalari"""
        user_id = callback.from_user.id
        if not self._is_authorized(user_id):
            await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
            return
            
        action = callback.data
        
        if action == "enable_trading":
            success = bot_manager.enable_auto_trading()
            if success:
                await callback.answer("✅ Auto trading yoqildi!")
            else:
                await callback.answer("❌ Trading yoqib bo'lmadi!", show_alert=True)
                
        elif action == "disable_trading":
            success = bot_manager.disable_auto_trading()
            if success:
                await callback.answer("🔴 Auto trading o'chirildi!")
            else:
                await callback.answer("❌ Xatolik yuz berdi!", show_alert=True)
                
        await self._update_status_message(callback.message)
        
    async def _handle_signal_action(self, callback: CallbackQuery):
        """Signal action tugmalari"""
        user_id = callback.from_user.id
        if not self._is_authorized(user_id):
            await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
            return
            
        data_parts = callback.data.split("_")
        action = data_parts[1]
        symbol = data_parts[2] if len(data_parts) > 2 else None
        
        if action == "execute" and symbol:
            # Execute signal
            from trading.execution_engine import execution_engine
            from trading.signal_generator import signal_generator
            
            # Get signal data
            signals = signal_generator.get_active_signals()
            signal = next((s for s in signals if s['symbol'] == symbol), None)
            
            if signal:
                result = await execution_engine.execute_signal(signal)
                if result['success']:
                    await callback.answer("✅ Signal bajarildi!")
                    await callback.message.edit_text(
                        callback.message.text + "\n\n✅ <b>BAJARILDI</b>"
                    )
                else:
                    await callback.answer(f"❌ Xato: {result['reason']}", show_alert=True)
            else:
                await callback.answer("❌ Signal topilmadi!", show_alert=True)
                
        elif action == "cancel" and symbol:
            await callback.answer("❌ Signal bekor qilindi")
            await callback.message.edit_text(
                callback.message.text + "\n\n❌ <b>BEKOR QILINDI</b>"
            )
            
    async def _handle_position_action(self, callback: CallbackQuery):
        """Position action tugmalari"""
        user_id = callback.from_user.id
        if not self._is_authorized(user_id):
            await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
            return
            
        data_parts = callback.data.split("_")
        action = data_parts[1]
        symbol = data_parts[2] if len(data_parts) > 2 else None
        
        if action == "close" and symbol:
            from trading.execution_engine import execution_engine
            result = await execution_engine.close_position(symbol, reason="manual_close")
            
            if result['success']:
                pnl_text = f"{'🟢' if result['pnl'] > 0 else '🔴'} PnL: {result['pnl']:+.2f}"
                await callback.answer(f"✅ Pozitsiya yopildi! {pnl_text}")
                await callback.message.edit_text(
                    callback.message.text + f"\n\n✅ <b>YOPILDI</b>\n{pnl_text}"
                )
            else:
                await callback.answer(f"❌ Xato: {result['reason']}", show_alert=True)
                
        elif action == "trail" and symbol:
            from trading.execution_engine import execution_engine
            success = await execution_engine.enable_trailing_stop(symbol)
            
            if success:
                await callback.answer("✅ Trailing stop yoqildi!")
            else:
                await callback.answer("❌ Xatolik yuz berdi!", show_alert=True)
                
    async def _handle_risk_action(self, callback: CallbackQuery, state: FSMContext):
        """Risk action tugmalari"""
        user_id = callback.from_user.id
        if not self._is_authorized(user_id):
            await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
            return
            
        action = callback.data
        from core.risk_manager import risk_manager
        
        if action == "risk_05":
            risk_manager._current_risk_percent = 0.5
            await callback.answer("✅ Risk 0.5% ga o'rnatildi")
            
        elif action == "risk_075":
            risk_manager._current_risk_percent = 0.75
            await callback.answer("✅ Risk 0.75% ga o'rnatildi")
            
        elif action == "risk_10":
            risk_manager._current_risk_percent = 1.0
            await callback.answer("✅ Risk 1.0% ga o'rnatildi")
            
        elif action == "risk_custom":
            await callback.answer("Risk foizini kiriting (0.1-2.0):")
            await state.set_state(BotStates.waiting_for_risk_input)
            
        await self._update_risk_message(callback.message)
        
    async def _handle_risk_input(self, message: Message, state: FSMContext):
        """Custom risk input"""
        try:
            risk_value = float(message.text)
            
            if 0.1 <= risk_value <= 2.0:
                from core.risk_manager import risk_manager
                risk_manager._current_risk_percent = risk_value
                
                await message.answer(f"✅ Risk {risk_value}% ga o'rnatildi")
                await state.clear()
            else:
                await message.answer("❌ Risk 0.1% dan 2.0% gacha bo'lishi kerak!")
                
        except ValueError:
            await message.answer("❌ Noto'g'ri format! Raqam kiriting (masalan: 0.75)")
            
    async def _handle_symbol_add(self, message: Message, state: FSMContext):
        """Symbol qo'shish"""
        symbol = message.text.upper()
        
        if not re.match(r'^[A-Z]+USDT$', symbol):
            await message.answer("❌ Noto'g'ri format! Masalan: BTCUSDT")
            return
            
        if symbol not in config_manager.trading.symbols:
            config_manager.trading.symbols.append(symbol)
            await config_manager.save_config()
            await message.answer(f"✅ {symbol} qo'shildi!")
        else:
            await message.answer(f"❌ {symbol} allaqachon mavjud!")
            
        await state.clear()
        
    async def _handle_symbol_remove(self, message: Message, state: FSMContext):
        """Symbol o'chirish"""
        symbol = message.text.upper()
        
        if symbol in config_manager.trading.symbols:
            config_manager.trading.symbols.remove(symbol)
            await config_manager.save_config()
            await message.answer(f"✅ {symbol} o'chirildi!")
        else:
            await message.answer(f"❌ {symbol} topilmadi!")
            
        await state.clear()
        
    async def _handle_text_message(self, message: Message):
        """Oddiy text xabarlar"""
        if not self._is_authorized(message.from_user.id):
            return
            
        # Update session activity
        if message.from_user.id in self.sessions:
            self.sessions[message.from_user.id].last_activity = TimeUtils.now_uzb()
            
        # Echo or help
        await message.answer("Komandalar ro'yxati uchun /help yozing")
        
    def _is_authorized(self, user_id: int) -> bool:
        """Foydalanuvchi ruxsatini tekshirish"""
        return user_id in self.authorized_users
        
    def _get_user_role(self, user_id: int) -> UserRole:
        """Foydalanuvchi rolini olish"""
        if user_id == self.authorized_users[0]:  # First user is admin
            return UserRole.ADMIN
        return UserRole.TRADER
        
    def _get_bot_status_emoji(self) -> str:
        """Bot status emojisi"""
        status_map = {
            "RUNNING": "🟢",
            "STOPPED": "🔴",
            "PAUSED": "⏸️",
            "ERROR": "❌",
            "STARTING": "🔄",
            "STOPPING": "🔄"
        }
        return status_map.get(bot_manager.status.name, "❓")
        
    def _get_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Asosiy keyboard"""
        keyboard = [
            [KeyboardButton(text="📊 Status"), KeyboardButton(text="📈 Signallar")],
            [KeyboardButton(text="💼 Pozitsiyalar"), KeyboardButton(text="📊 Statistika")],
            [KeyboardButton(text="⚖️ Risk"), KeyboardButton(text="💱 Juftliklar")]
        ]
        
        return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)
        
    def _get_control_keyboard(self) -> InlineKeyboardMarkup:
        """Bot control keyboard"""
        keyboard = []
        
        # Bot control buttons
        if bot_manager.status.name == "STOPPED":
            keyboard.append([InlineKeyboardButton(text="🟢 Bot Ishga Tushirish", callback_data="start_bot")])
        elif bot_manager.status.name == "RUNNING":
            keyboard.append([
                InlineKeyboardButton(text="⏸️ Pauza", callback_data="pause_bot"),
                InlineKeyboardButton(text="🔴 To'xtatish", callback_data="stop_bot")
            ])
        elif bot_manager.status.name == "PAUSED":
            keyboard.append([
                InlineKeyboardButton(text="▶️ Davom Ettirish", callback_data="start_bot"),
                InlineKeyboardButton(text="🔴 To'xtatish", callback_data="stop_bot")
            ])
            
        # Trading control
        if bot_manager._auto_trading_enabled:
            keyboard.append([InlineKeyboardButton(text="💰 Auto Trading: ✅", callback_data="disable_trading")])
        else:
            keyboard.append([InlineKeyboardButton(text="💰 Auto Trading: ❌", callback_data="enable_trading")])
            
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
        
    def _get_signal_keyboard(self, symbol: str) -> InlineKeyboardMarkup:
        """Signal action keyboard"""
        keyboard = [
            [
                InlineKeyboardButton(text="✅ Avto Savdo", callback_data=f"signal_execute_{symbol}"),
                InlineKeyboardButton(text="❌ Bekor Qilish", callback_data=f"signal_cancel_{symbol}")
            ]
        ]
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
        
    def _get_position_keyboard(self, symbol: str) -> InlineKeyboardMarkup:
        """Position action keyboard"""
        keyboard = [
            [
                InlineKeyboardButton(text="🔴 Yopish", callback_data=f"position_close_{symbol}"),
                InlineKeyboardButton(text="🎯 Trailing Stop", callback_data=f"position_trail_{symbol}")
            ]
        ]
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
        
    def _get_risk_keyboard(self) -> InlineKeyboardMarkup:
        """Risk management keyboard"""
        keyboard = [
            [
                InlineKeyboardButton(text="0.5%", callback_data="risk_05"),
                InlineKeyboardButton(text="0.75%", callback_data="risk_075"),
                InlineKeyboardButton(text="1.0%", callback_data="risk_10")
            ],
            [InlineKeyboardButton(text="⚙️ Custom", callback_data="risk_custom")]
        ]
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
        
    def _format_stop_loss_reasons(self, reasons: Dict[str, int]) -> str:
        """Stop loss sabablarini formatlash"""
        if not reasons:
            return "Ma'lumot yo'q"
            
        formatted = []
        for reason, count in reasons.items():
            formatted.append(f"• {reason}: {count}")
            
        return "\n".join(formatted)
        
    def _format_take_profit_reasons(self, reasons: Dict[str, int]) -> str:
        """Take profit sabablarini formatlash"""
        if not reasons:
            return "Ma'lumot yo'q"
            
        formatted = []
        for reason, count in reasons.items():
            formatted.append(f"• {reason}: {count}")
            
        return "\n".join(formatted)
        
    async def _update_status_message(self, message: Message):
        """Status xabarini yangilash"""
        try:
            status = bot_manager.get_status()
            
            status_text = f"""
📊 <b>Bot Status</b>

🤖 Bot: {self._get_bot_status_emoji()} {status['status']}
📡 Signal Mode: {status['signal_mode']}
💱 Trading Mode: {status['trading_mode']}
💰 Auto Trading: {'✅ ON' if status['auto_trading'] else '❌ OFF'}

<b>Statistika:</b>
📈 Jami Signallar: {status['statistics']['total_signals']}
✅ Bajarilgan: {status['statistics']['executed_trades']}
📊 Win Rate: {status['statistics']['win_rate']}
💵 Total PnL: {status['statistics']['total_pnl']}
❌ Xatolar: {status['statistics']['errors']}

<b>Komponentlar:</b>
{', '.join(status['components'])}
