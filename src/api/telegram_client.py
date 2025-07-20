"""
Telegram Bot API Client
Signal yuborish, keyboard yaratish, user boshqaruv
"""
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
from enum import Enum, auto
import json

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import RateLimiter, retry_manager, TimeUtils

logger = get_logger(__name__)

class MessageType(Enum):
    """Xabar turlari"""
    SIGNAL = auto()
    ALERT = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    REPORT = auto()

class ButtonType(Enum):
    """Tugma turlari"""
    CALLBACK = auto()
    URL = auto()
    SWITCH_INLINE = auto()

@dataclass
class TelegramButton:
    """Telegram tugma"""
    text: str
    type: ButtonType = ButtonType.CALLBACK
    callback_data: Optional[str] = None
    url: Optional[str] = None
    
@dataclass
class TelegramKeyboard:
    """Telegram klaviatura"""
    buttons: List[List[TelegramButton]] = field(default_factory=list)
    inline: bool = True
    resize: bool = False
    one_time: bool = False
    
@dataclass
class TelegramMessage:
    """Telegram xabar"""
    chat_id: Union[str, int]
    text: str
    type: MessageType = MessageType.INFO
    keyboard: Optional[TelegramKeyboard] = None
    parse_mode: str = "HTML"
    disable_notification: bool = False
    reply_to: Optional[int] = None
    photo: Optional[str] = None

class TelegramClient:
    """Telegram Bot API client"""
    def __init__(self):
        self.bot_token = config_manager.telegram.bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.channel_id = config_manager.telegram.channel_id
        self.admin_ids = config_manager.telegram.admin_ids
        self.rate_limiter = RateLimiter(calls=30, period=1)  # 30 msg/sec
        self.session: Optional[aiohttp.ClientSession] = None
        self._message_handlers: Dict[str, Callable] = {}
        self._callback_handlers: Dict[str, Callable] = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def send_message(self, message: TelegramMessage) -> Optional[Dict[str, Any]]:
        """Xabar yuborish"""
        if not self.bot_token:
            logger.error("Telegram bot token topilmadi")
            return None
            
        try:
            await self.rate_limiter.acquire()
            
            data = {
                "chat_id": message.chat_id,
                "text": self._format_message(message),
                "parse_mode": message.parse_mode,
                "disable_notification": message.disable_notification
            }
            
            if message.reply_to:
                data["reply_to_message_id"] = message.reply_to
                
            if message.keyboard:
                data["reply_markup"] = self._create_keyboard(message.keyboard)
                
            # Photo bilan yuborish
            if message.photo:
                return await self._send_photo(data, message.photo)
            else:
                return await self._send_text(data)
                
        except Exception as e:
            logger.error(f"Telegram xabar yuborishda xato: {e}")
            return None
            
    async def _send_text(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Matnli xabar yuborish"""
        url = f"{self.base_url}/sendMessage"
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("ok"):
                    return result.get("result")
                else:
                    logger.error(f"Telegram API xatosi: {result.get('description')}")
            else:
                logger.error(f"Telegram HTTP xatosi: {response.status}")
                
        return None
        
    async def _send_photo(self, data: Dict[str, Any], photo_url: str) -> Optional[Dict[str, Any]]:
        """Rasmli xabar yuborish"""
        url = f"{self.base_url}/sendPhoto"
        data["photo"] = photo_url
        data["caption"] = data.pop("text", "")
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("ok"):
                    return result.get("result")
                    
        return None
        
    def _format_message(self, message: TelegramMessage) -> str:
        """Xabarni formatlash"""
        # Message type emoji
        type_emojis = {
            MessageType.SIGNAL: "📊",
            MessageType.ALERT: "🚨",
            MessageType.INFO: "ℹ️",
            MessageType.WARNING: "⚠️",
            MessageType.ERROR: "❌",
            MessageType.REPORT: "📈"
        }
        
        emoji = type_emojis.get(message.type, "")
        
        # Add timestamp for signals
        if message.type == MessageType.SIGNAL:
            timestamp = TimeUtils.format_uzb(fmt="%H:%M")
            return f"{emoji} {message.text}\n\n⏰ <i>Vaqt: {timestamp} (UZB)</i>"
        else:
            return f"{emoji} {message.text}"
            
    def _create_keyboard(self, keyboard: TelegramKeyboard) -> Dict[str, Any]:
        """Klaviatura yaratish"""
        if keyboard.inline:
            buttons = []
            for row in keyboard.buttons:
                button_row = []
                for button in row:
                    btn_data = {"text": button.text}
                    
                    if button.type == ButtonType.CALLBACK:
                        btn_data["callback_data"] = button.callback_data or button.text
                    elif button.type == ButtonType.URL:
                        btn_data["url"] = button.url
                        
                    button_row.append(btn_data)
                buttons.append(button_row)
                
            return {
                "keyboard": buttons,
                "resize_keyboard": keyboard.resize,
                "one_time_keyboard": keyboard.one_time
            }
            
    async def send_signal(self, signal_text: str, keyboard: Optional[TelegramKeyboard] = None) -> bool:
        """Trading signal yuborish"""
        message = TelegramMessage(
            chat_id=self.channel_id,
            text=signal_text,
            type=MessageType.SIGNAL,
            keyboard=keyboard or self._create_signal_keyboard()
        )
        
        result = await self.send_message(message)
        return result is not None
        
    def _create_signal_keyboard(self) -> TelegramKeyboard:
        """Signal uchun default keyboard"""
        return TelegramKeyboard(buttons=[
            [
                TelegramButton("🟢 AVTO SAVDO", callback_data="auto_trade"),
                TelegramButton("🔴 BEKOR QILISH", callback_data="cancel_signal")
            ],
            [
                TelegramButton("📊 Statistika", callback_data="stats"),
                TelegramButton("⚙️ Sozlamalar", callback_data="settings")
            ]
        ])
        
    async def send_alert(self, alert_text: str, important: bool = False) -> bool:
        """Alert xabar yuborish"""
        message = TelegramMessage(
            chat_id=self.channel_id,
            text=alert_text,
            type=MessageType.ALERT,
            disable_notification=not important
        )
        
        result = await self.send_message(message)
        return result is not None
        
    async def send_report(self, report_data: Dict[str, Any]) -> bool:
        """Hisobot yuborish"""
        report_text = self._format_report(report_data)
        
        message = TelegramMessage(
            chat_id=self.channel_id,
            text=report_text,
            type=MessageType.REPORT,
            parse_mode="HTML"
        )
        
        result = await self.send_message(message)
        return result is not None
        
    def _format_report(self, data: Dict[str, Any]) -> str:
        """Hisobotni formatlash"""
        period = data.get("period", "Kunlik")
        trades = data.get("total_trades", 0)
        profit_trades = data.get("profit_trades", 0)
        loss_trades = data.get("loss_trades", 0)
        total_pnl = data.get("total_pnl", 0)
        win_rate = data.get("win_rate", 0)
        
        report = f"""📈 <b>{period} HISOBOT</b>
════════════════════
📊 Jami savdolar: {trades}
✅ Foydali: {profit_trades}
❌ Zararli: {loss_trades}
💰 Umumiy PnL: {total_pnl:+.2f}%
🎯 Win Rate: {win_rate:.1f}%
════════════════════"""
        
        # Top performing pairs
        if "top_pairs" in data:
            report += "\n\n🏆 <b>TOP JUFTLIKLAR:</b>"
            for i, (pair, pnl) in enumerate(data["top_pairs"][:3], 1):
                report += f"\n{i}. {pair}: {pnl:+.2f}%"
                
        return report
        
    async def edit_message(self, chat_id: Union[str, int], message_id: int, new_text: str, 
                          keyboard: Optional[TelegramKeyboard] = None) -> bool:
        """Xabarni tahrirlash"""
        try:
            url = f"{self.base_url}/editMessageText"
            
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": new_text,
                "parse_mode": "HTML"
            }
            
            if keyboard:
                data["reply_markup"] = self._create_keyboard(keyboard)
                
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("ok", False)
                    
        except Exception as e:
            logger.error(f"Xabar tahrirlashda xato: {e}")
            
        return False
        
    async def delete_message(self, chat_id: Union[str, int], message_id: int) -> bool:
        """Xabarni o'chirish"""
        try:
            url = f"{self.base_url}/deleteMessage"
            data = {"chat_id": chat_id, "message_id": message_id}
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("ok", False)
                    
        except Exception as e:
            logger.error(f"Xabar o'chirishda xato: {e}")
            
        return False
        
    async def answer_callback_query(self, callback_id: str, text: Optional[str] = None, 
                                  show_alert: bool = False) -> bool:
        """Callback query javob berish"""
        try:
            url = f"{self.base_url}/answerCallbackQuery"
            data = {"callback_query_id": callback_id, "show_alert": show_alert}
            
            if text:
                data["text"] = text
                
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("ok", False)
                    
        except Exception as e:
            logger.error(f"Callback javobda xato: {e}")
            
        return False
        
    def register_handler(self, handler_type: str, pattern: str, handler: Callable):
        """Handler ro'yxatdan o'tkazish"""
        if handler_type == "message":
            self._message_handlers[pattern] = handler
        elif handler_type == "callback":
            self._callback_handlers[pattern] = handler
            
    async def process_update(self, update: Dict[str, Any]):
        """Telegram update ni qayta ishlash"""
        try:
            # Message handler
            if "message" in update:
                message = update["message"]
                text = message.get("text", "")
                
                for pattern, handler in self._message_handlers.items():
                    if pattern in text or pattern == "*":
                        await handler(message)
                        break
                        
            # Callback query handler
            elif "callback_query" in update:
                callback = update["callback_query"]
                data = callback.get("data", "")
                
                for pattern, handler in self._callback_handlers.items():
                    if pattern in data or pattern == "*":
                        await handler(callback)
                        await self.answer_callback_query(callback["id"])
                        break
                        
        except Exception as e:
            logger.error(f"Update qayta ishlashda xato: {e}")
            
    async def set_webhook(self, webhook_url: str) -> bool:
        """Webhook o'rnatish"""
        try:
            url = f"{self.base_url}/setWebhook"
            data = {"url": webhook_url}
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("ok", False)
                    
        except Exception as e:
            logger.error(f"Webhook o'rnatishda xato: {e}")
            
        return False
        
    async def get_updates(self, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Updates olish (polling uchun)"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {"timeout": 30}
            
            if offset:
                params["offset"] = offset
                
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        return result.get("result", [])
                        
        except Exception as e:
            logger.error(f"Updates olishda xato: {e}")
            
        return []

class TelegramNotifier:
    """Telegram orqali notification yuborish"""
    def __init__(self):
        self.client = TelegramClient()
        self._notification_queue: List[TelegramMessage] = []
        self._batch_size = 10
        self._batch_interval = 1.0  # sekund
        self._running = False
        
    async def start(self):
        """Notifier ishga tushirish"""
        self._running = True
        asyncio.create_task(self._process_queue())
        logger.info("✅ Telegram Notifier ishga tushdi")
        
    async def stop(self):
        """Notifier to'xtatish"""
        self._running = False
        await self._flush_queue()
        
    async def notify_signal(self, signal_data: Dict[str, Any]):
        """Signal haqida xabar berish"""
        text = self._format_signal(signal_data)
        
        keyboard = TelegramKeyboard(buttons=[
            [
                TelegramButton("🟢 AVTO SAVDO", callback_data=f"auto_{signal_data.get('id')}"),
                TelegramButton("🔴 BEKOR", callback_data=f"cancel_{signal_data.get('id')}")
            ]
        ])
        
        message = TelegramMessage(
            chat_id=self.client.channel_id,
            text=text,
            type=MessageType.SIGNAL,
            keyboard=keyboard
        )
        
        await self._add_to_queue(message)
        
    def _format_signal(self, data: Dict[str, Any]) -> str:
        """Signal ma'lumotlarini formatlash"""
        return f"""📊 <b>SIGNAL KELDI</b>
════════════════
📈 Savdo: <b>{data.get('action')} {data.get('pair')}</b>
💰 Narx: <b>${data.get('price')}</b>
📊 Lot: <b>{data.get('lot')} {data.get('base_currency')}</b>
🛡️ Stop Loss: <b>${data.get('stop_loss')} ({data.get('sl_percentage')}%)</b>
🎯 Take Profit: <b>${data.get('take_profit')} ({data.get('tp_percentage')}%)</b>
⚡ Ishonch: <b>{data.get('confidence')}%</b>
🔥 Risk: <b>{data.get('risk')}%</b>
════════════════
📝 Sabab: {data.get('reason')}
🐋 On-chain: {data.get('onchain_info', 'N/A')}"""
        
    async def notify_error(self, error_text: str, critical: bool = False):
        """Xatolik haqida xabar"""
        message = TelegramMessage(
            chat_id=self.client.channel_id,
            text=f"❌ <b>XATOLIK</b>\n\n{error_text}",
            type=MessageType.ERROR,
            disable_notification=not critical
        )
        
        if critical:
            # Critical xatolar darhol yuboriladi
            async with self.client:
                await self.client.send_message(message)
        else:
            await self._add_to_queue(message)
            
    async def notify_market_update(self, update_data: Dict[str, Any]):
        """Bozor yangiligi"""
        mood = update_data.get("mood", "NEUTRAL")
        mood_emoji = {
            "VERY_BULLISH": "🚀",
            "BULLISH": "📈",
            "NEUTRAL": "➡️",
            "BEARISH": "📉",
            "VERY_BEARISH": "💥"
        }.get(mood, "➡️")
        
        text = f"""{mood_emoji} <b>BOZOR YANGILIGI</b>

Kayfiyat: <b>{mood}</b>
Bullish: {update_data.get('bullish_ratio', 0):.1%}
Bearish: {update_data.get('bearish_ratio', 0):.1%}

🔥 Trend mavzular:"""
        
        for topic in update_data.get("trending_topics", [])[:3]:
            text += f"\n• {topic['topic']}: {topic['count']} ({topic['percentage']:.1f}%)"
            
        message = TelegramMessage(
            chat_id=self.client.channel_id,
            text=text,
            type=MessageType.INFO
        )
        
        await self._add_to_queue(message)
        
    async def _add_to_queue(self, message: TelegramMessage):
        """Xabarni navbatga qo'shish"""
        self._notification_queue.append(message)
        
        if len(self._notification_queue) >= self._batch_size:
            await self._flush_queue()
            
    async def _process_queue(self):
        """Navbatni qayta ishlash"""
        while self._running:
            try:
                await asyncio.sleep(self._batch_interval)
                
                if self._notification_queue:
                    await self._flush_queue()
                    
            except Exception as e:
                logger.error(f"Queue process xatosi: {e}")
                
    async def _flush_queue(self):
        """Navbatdagi xabarlarni yuborish"""
        if not self._notification_queue:
            return
            
        messages_to_send = self._notification_queue[:self._batch_size]
        self._notification_queue = self._notification_queue[self._batch_size:]
        
        async with self.client:
            for message in messages_to_send:
                try:
                    await self.client.send_message(message)
                    await asyncio.sleep(0.1)  # Rate limit uchun
                except Exception as e:
                    logger.error(f"Xabar yuborishda xato: {e}")

# Global instances
telegram_client = TelegramClient()
telegram_notifier = TelegramNotifier()
        else:
            # Reply keyboard
            buttons = [[btn.text for btn in row] for row in keyboard.buttons]
            return {
                "
