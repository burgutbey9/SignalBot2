"""
Message Handler - Xabarlarni formatlash
Signal, pozitsiya, alert va boshqa xabarlarni O'zbekcha formatlash
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from utils.logger import get_logger
from utils.helpers import TimeUtils

logger = get_logger(__name__)

class MessageType(Enum):
    """Xabar turlari"""
    SIGNAL = "signal"
    POSITION = "position"
    ALERT = "alert"
    ERROR = "error"
    INFO = "info"
    TRADE_RESULT = "trade_result"
    MARKET_UPDATE = "market_update"

class MessagePriority(Enum):
    """Xabar muhimligi"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class MessageHandler:
    """Xabarlarni formatlash handleri"""
    
    def __init__(self):
        # Emoji mappings
        self.signal_emojis = {
            "STRONG_BUY": "🚀",
            "BUY": "📈",
            "NEUTRAL": "⚖️",
            "SELL": "📉",
            "STRONG_SELL": "🔻"
        }
        
        self.result_emojis = {
            "WIN": "✅",
            "LOSS": "❌",
            "BREAKEVEN": "➖",
            "ACTIVE": "🔄"
        }
        
        self.mood_emojis = {
            "BULLISH": "🐂",
            "BEARISH": "🐻",
            "NEUTRAL": "😐",
            "EXTREME_FEAR": "😱",
            "EXTREME_GREED": "🤑"
        }
        
    def format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Signal xabarini formatlash"""
        try:
            # Get emoji
            signal_emoji = self.signal_emojis.get(signal.get('type', 'NEUTRAL'), '📊')
            
            # Format confidence
            confidence = signal.get('confidence', 0)
            confidence_bar = self._get_confidence_bar(confidence)
            
            # Format risk/reward
            rr_ratio = signal.get('risk_reward_ratio', 0)
            
            # Format take profits
            take_profits = signal.get('take_profit', [])
            tp_text = self._format_take_profits(take_profits)
            
            # Format reasoning
            reasoning = signal.get('reasoning', [])
            reasoning_text = self._format_reasoning(reasoning)
            
            # Build message
            message = f"""
{signal_emoji} <b>SIGNAL KELDI</b>
════════════════════════

📊 <b>Savdo:</b> {signal.get('action', 'HOLD')} {signal.get('symbol', 'UNKNOWN')}
💰 <b>Kirish narxi:</b> ${signal.get('entry_price', 0):,.2f}
🛡️ <b>Stop Loss:</b> ${signal.get('stop_loss', 0):,.2f}
🎯 <b>Take Profit:</b>
{tp_text}

📊 <b>Lot hajmi:</b> {signal.get('position_size', 0):.4f}
📈 <b>Risk/Reward:</b> 1:{rr_ratio:.1f}
⚡ <b>Ishonch darajasi:</b> {confidence:.0f}%
{confidence_bar}

📝 <b>Sabab:</b>
{reasoning_text}

⏰ <b>Vaqt:</b> {TimeUtils.now_uzb().strftime('%H:%M:%S')} (UZB)

<i>Signal {signal.get('expires_at', 'cheksiz')} gacha amal qiladi</i>
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Signal formatting xatosi: {e}")
            return "❌ Signal formatlashda xatolik"
            
    def format_position_message(self, position: Dict[str, Any]) -> str:
        """Pozitsiya xabarini formatlash"""
        try:
            # Calculate PnL percentage
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            
            if position.get('side') == 'BUY':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
                
            # PnL emoji
            pnl_emoji = "🟢" if pnl_percent > 0 else "🔴" if pnl_percent < 0 else "⚪"
            
            message = f"""
💼 <b>OCHIQ POZITSIYA</b>
════════════════════════

📊 <b>Juftlik:</b> {position.get('symbol', 'UNKNOWN')}
📈 <b>Yo'nalish:</b> {position.get('side', 'UNKNOWN')}
💰 <b>Kirish narxi:</b> ${entry_price:,.2f}
📊 <b>Hozirgi narx:</b> ${current_price:,.2f}
📏 <b>Hajm:</b> {position.get('quantity', 0):.4f}

{pnl_emoji} <b>PnL:</b> ${position.get('unrealized_pnl', 0):+,.2f} ({pnl_percent:+.2f}%)

⏱️ <b>Davomiyligi:</b> {self._format_duration(position.get('opened_at'))}
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Position formatting xatosi: {e}")
            return "❌ Pozitsiya formatlashda xatolik"
            
    def format_trade_result_message(self, trade: Dict[str, Any]) -> str:
        """Savdo natijasi xabarini formatlash"""
        try:
            # Get result emoji
            result = trade.get('result', 'UNKNOWN')
            result_emoji = self.result_emojis.get(result, '❓')
            
            # Format PnL
            pnl = trade.get('pnl', 0)
            pnl_percent = trade.get('pnl_percent', 0)
            pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            # Exit reason
            exit_reason = trade.get('exit_reason', 'UNKNOWN')
            detailed_reason = trade.get('detailed_reason', '')
            
            message = f"""
{result_emoji} <b>SAVDO YAKUNLANDI</b>
════════════════════════

📊 <b>Juftlik:</b> {trade.get('symbol', 'UNKNOWN')}
📈 <b>Yo'nalish:</b> {trade.get('side', 'UNKNOWN')}

💵 <b>Kirish:</b> ${trade.get('entry_price', 0):,.2f}
💸 <b>Chiqish:</b> ${trade.get('exit_price', 0):,.2f}

{pnl_color} <b>Natija:</b> ${pnl:+,.2f} ({pnl_percent:+.2f}%)

📝 <b>Chiqish sababi:</b> {exit_reason}
{f'💡 <i>{detailed_reason}</i>' if detailed_reason else ''}

⏱️ <b>Davomiyligi:</b> {trade.get('duration', 'N/A')}

<b>Bozor holati:</b>
• Narx: ${trade.get('market_conditions', {}).get('price', 0):,.2f}
• Hajm 24s: ${trade.get('market_conditions', {}).get('volume_24h', 0):,.0f}
• O'zgarish 24s: {trade.get('market_conditions', {}).get('price_change_24h', 0):+.2f}%
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Trade result formatting xatosi: {e}")
            return "❌ Savdo natijasi formatlashda xatolik"
            
    def format_alert_message(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Alert xabarini formatlash"""
        try:
            if alert_type == "whale_alert":
                return self._format_whale_alert(data)
            elif alert_type == "risk_alert":
                return self._format_risk_alert(data)
            elif alert_type == "market_alert":
                return self._format_market_alert(data)
            elif alert_type == "news_alert":
                return self._format_news_alert(data)
            else:
                return self._format_generic_alert(alert_type, data)
                
        except Exception as e:
            logger.error(f"Alert formatting xatosi: {e}")
            return "❌ Alert formatlashda xatolik"
            
    def _format_whale_alert(self, data: Dict[str, Any]) -> str:
        """Whale alert formatlash"""
        amount = data.get('amount', 0)
        value_usd = data.get('value_usd', 0)
        alert_type = data.get('type', 'TRANSFER')
        
        emoji = "🐋" if value_usd > 10000000 else "🐳"  # $10M+
        
        message = f"""
{emoji} <b>WHALE ALERT</b>

💰 <b>Miqdor:</b> {amount:,.2f} {data.get('symbol', '')}
💵 <b>Qiymat:</b> ${value_usd:,.0f}
🔄 <b>Turi:</b> {alert_type}

{self._get_whale_interpretation(alert_type, value_usd)}
"""
        
        return message.strip()
        
    def _format_risk_alert(self, data: Dict[str, Any]) -> str:
        """Risk alert formatlash"""
        risk_type = data.get('type', 'UNKNOWN')
        severity = data.get('severity', 'MEDIUM')
        
        emoji_map = {
            'LOW': '📊',
            'MEDIUM': '⚠️',
            'HIGH': '🚨',
            'CRITICAL': '🚨🚨🚨'
        }
        
        emoji = emoji_map.get(severity, '⚠️')
        
        message = f"""
{emoji} <b>RISK ALERT</b>

📊 <b>Turi:</b> {risk_type}
⚡ <b>Darajasi:</b> {severity}
📝 <b>Tavsif:</b> {data.get('description', 'N/A')}

💡 <b>Tavsiya:</b> {data.get('recommendation', 'Ehtiyot bo\'ling')}
"""
        
        return message.strip()
        
    def _format_market_alert(self, data: Dict[str, Any]) -> str:
        """Market alert formatlash"""
        alert_type = data.get('type', 'UNKNOWN')
        symbol = data.get('symbol', 'MARKET')
        
        message = f"""
📊 <b>BOZOR ALERT</b>

💱 <b>Juftlik:</b> {symbol}
📈 <b>Holat:</b> {alert_type}

{data.get('description', '')}

<b>Ma'lumotlar:</b>
• Narx: ${data.get('price', 0):,.2f}
• O'zgarish: {data.get('change_percent', 0):+.2f}%
• Hajm: ${data.get('volume', 0):,.0f}
"""
        
        return message.strip()
        
    def _format_news_alert(self, data: Dict[str, Any]) -> str:
        """News alert formatlash"""
        impact = data.get('impact', 'MEDIUM')
        
        emoji_map = {
            'LOW': '📰',
            'MEDIUM': '📢',
            'HIGH': '📣',
            'CRITICAL': '🚨'
        }
        
        emoji = emoji_map.get(impact, '📰')
        
        message = f"""
{emoji} <b>YANGILIK</b>

📌 <b>Sarlavha:</b> {data.get('title', 'N/A')}
📊 <b>Ta'sir:</b> {impact}
🏷️ <b>Teglar:</b> {', '.join(data.get('tags', []))}

📝 {data.get('summary', '')}

🔗 <a href="{data.get('url', '#')}">To'liq o'qish</a>
"""
        
        return message.strip()
        
    def _format_generic_alert(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Umumiy alert formatlash"""
        message = f"""
⚠️ <b>{alert_type.upper()}</b>

{data.get('message', 'Alert ma\'lumotlari mavjud emas')}

⏰ {TimeUtils.now_uzb().strftime('%H:%M:%S')}
"""
        
        return message.strip()
        
    def format_error_message(self, error_type: str, details: str) -> str:
        """Xatolik xabarini formatlash"""
        message = f"""
❌ <b>XATOLIK</b>

🔴 <b>Turi:</b> {error_type}
📝 <b>Tafsilotlar:</b> {details}

💡 <i>Agar muammo davom etsa, @admin ga murojaat qiling</i>
"""
        
        return message.strip()
        
    def format_market_update(self, data: Dict[str, Any]) -> str:
        """Bozor yangilanishi xabarini formatlash"""
        mood = data.get('mood', 'NEUTRAL')
        mood_emoji = self.mood_emojis.get(mood, '📊')
        
        message = f"""
{mood_emoji} <b>BOZOR YANGILANISHI</b>

{data.get('message', '')}

<b>Asosiy ko'rsatkichlar:</b>
• Fear & Greed: {data.get('fear_greed', 50)}
• BTC Dominance: {data.get('btc_dominance', 0):.1f}%
• Total Market Cap: ${data.get('market_cap', 0)/1e9:.1f}B

⏰ {data.get('timestamp', TimeUtils.now_uzb()).strftime('%H:%M')} (UZB)
"""
        
        return message.strip()
        
    def format_performance_summary(self, period: str, stats: Dict[str, Any]) -> str:
        """Performance xulosasini formatlash"""
        win_rate = stats.get('win_rate', 0)
        total_pnl = stats.get('total_pnl', 0)
        
        # Win rate emoji
        if win_rate >= 70:
            wr_emoji = "🔥"
        elif win_rate >= 50:
            wr_emoji = "✅"
        else:
            wr_emoji = "📉"
            
        # PnL emoji
        pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
        
        message = f"""
📊 <b>{period.upper()} HISOBOT</b>
════════════════════════

<b>Savdolar:</b>
• Jami: {stats.get('total_trades', 0)}
• Yutuqlar: {stats.get('wins', 0)}
• Yutqazishlar: {stats.get('losses', 0)}

{wr_emoji} <b>Win Rate:</b> {win_rate:.1f}%
{pnl_emoji} <b>Umumiy PnL:</b> {total_pnl:+.2f}%

<b>O'rtacha ko'rsatkichlar:</b>
• O'rtacha yutuq: +{stats.get('avg_win', 0):.2f}%
• O'rtacha zarar: -{stats.get('avg_loss', 0):.2f}%
• Profit Factor: {stats.get('profit_factor', 0):.2f}

<b>Risk ko'rsatkichlari:</b>
• Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
• Max Drawdown: -{stats.get('max_drawdown', 0):.2f}%

💡 <i>{self._get_performance_advice(win_rate, total_pnl)}</i>
"""
        
        return message.strip()
        
    def _get_confidence_bar(self, confidence: float) -> str:
        """Ishonch darajasi progress bar"""
        filled = int(confidence / 10)
        empty = 10 - filled
        
        bar = "🟩" * filled + "⬜" * empty
        
        return f"[{bar}]"
        
    def _format_take_profits(self, take_profits: List[float]) -> str:
        """Take profitlarni formatlash"""
        if not take_profits:
            return "   • Belgilanmagan"
            
        tp_lines = []
        for i, tp in enumerate(take_profits[:3], 1):
            tp_lines.append(f"   • TP{i}: ${tp:,.2f}")
            
        return "\n".join(tp_lines)
        
    def _format_reasoning(self, reasoning: List[str]) -> str:
        """Sabablarni formatlash"""
        if not reasoning:
            return "Ma'lumot yo'q"
            
        # Take first 5 reasons
        formatted = []
        for reason in reasoning[:5]:
            formatted.append(f"• {reason}")
            
        return "\n".join(formatted)
        
    def _format_duration(self, opened_at: Any) -> str:
        """Davomiylikni formatlash"""
        try:
            if isinstance(opened_at, str):
                opened_at = datetime.fromisoformat(opened_at)
            elif not isinstance(opened_at, datetime):
                return "N/A"
                
            duration = TimeUtils.now_uzb() - opened_at
            
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            
            if hours > 0:
                return f"{hours}s {minutes}d"
            else:
                return f"{minutes} daqiqa"
                
        except:
            return "N/A"
            
    def _get_whale_interpretation(self, alert_type: str, value_usd: float) -> str:
        """Whale harakati talqini"""
        if alert_type == "EXCHANGE_IN":
            if value_usd > 50000000:  # $50M+
                return "🔴 <b>KATTA SELL PRESSURE kutilmoqda!</b>"
            else:
                return "📉 Sell pressure kuchayishi mumkin"
                
        elif alert_type == "EXCHANGE_OUT":
            if value_usd > 50000000:  # $50M+
                return "🟢 <b>KUCHLI ACCUMULATION!</b>"
            else:
                return "📈 Accumulation davom etmoqda"
                
        elif alert_type == "LARGE_TRANSFER":
            return "🔄 Katta hajmdagi transfer amalga oshirildi"
            
        return "📊 Whale faoliyati aniqlandi"
        
    def _get_performance_advice(self, win_rate: float, total_pnl: float) -> str:
        """Performance bo'yicha maslahat"""
        if win_rate >= 70 and total_pnl > 10:
            return "Ajoyib natija! Strategiya yaxshi ishlayapti 🎯"
        elif win_rate >= 50 and total_pnl > 0:
            return "Yaxshi natija. Risk managementga e'tibor bering 📊"
        elif win_rate < 50 and total_pnl < 0:
            return "Strategiyani qayta ko'rib chiqing. Stop loss qoidalariga amal qiling ⚠️"
        else:
            return "Barqaror savdo qiling, hissiyotlarga berilmang 🧘"
            
    def get_message_priority(self, message_type: MessageType, data: Dict[str, Any]) -> MessagePriority:
        """Xabar muhimligini aniqlash"""
        if message_type == MessageType.ERROR:
            return MessagePriority.HIGH
            
        elif message_type == MessageType.SIGNAL:
            confidence = data.get('confidence', 0)
            if confidence >= 85:
                return MessagePriority.HIGH
            elif confidence >= 70:
                return MessagePriority.MEDIUM
            else:
                return MessagePriority.LOW
                
        elif message_type == MessageType.ALERT:
            severity = data.get('severity', 'MEDIUM')
            if severity == 'CRITICAL':
                return MessagePriority.CRITICAL
            elif severity == 'HIGH':
                return MessagePriority.HIGH
            else:
                return MessagePriority.MEDIUM
                
        else:
            return MessagePriority.LOW

# Global instance
message_handler = MessageHandler()
