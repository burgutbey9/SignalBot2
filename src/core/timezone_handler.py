"""
Timezone Handler for Uzbekistan Time
O'zbekiston vaqt zonasi boshqaruvi
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, time, date, timezone
from dataclasses import dataclass
from enum import Enum, auto
import pytz
from zoneinfo import ZoneInfo

from config.config import config_manager
from utils.logger import get_logger

logger = get_logger(__name__)

class TradingSession(Enum):
    """Trading sessiyalari"""
    ASIA = auto()      # 00:00 - 08:00 UTC (05:00 - 13:00 UZB)
    EUROPE = auto()    # 08:00 - 16:00 UTC (13:00 - 21:00 UZB)
    US = auto()        # 13:00 - 22:00 UTC (18:00 - 03:00 UZB)
    CRYPTO = auto()    # 24/7 trading

class MarketHours(Enum):
    """Bozor soatlari (UTC)"""
    ASIA_OPEN = time(0, 0)      # Tokyo open
    ASIA_CLOSE = time(8, 0)     
    EUROPE_OPEN = time(8, 0)    # London open
    EUROPE_CLOSE = time(16, 0)
    US_OPEN = time(13, 0)       # New York open  
    US_CLOSE = time(22, 0)

@dataclass
class SessionInfo:
    """Sessiya ma'lumotlari"""
    session: TradingSession
    start_utc: time
    end_utc: time
    start_uzb: time
    end_uzb: time
    is_active: bool = False
    volatility_level: str = "NORMAL"  # LOW, NORMAL, HIGH
    
@dataclass
class KillZone:
    """ICT Kill Zone vaqtlari"""
    name: str
    start_time: time
    end_time: time
    session: TradingSession
    importance: str  # HIGH, MEDIUM, LOW

class TimezoneHandler:
    """Vaqt zonasi handler"""
    def __init__(self):
        self.uzb_tz = pytz.timezone('Asia/Tashkent')
        self.utc_tz = pytz.UTC
        self.sessions: Dict[TradingSession, SessionInfo] = {}
        self.kill_zones: List[KillZone] = []
        self._initialize_sessions()
        self._initialize_kill_zones()
        
    def _initialize_sessions(self):
        """Trading sessiyalarini sozlash"""
        # Asia session (05:00 - 13:00 UZB)
        self.sessions[TradingSession.ASIA] = SessionInfo(
            session=TradingSession.ASIA,
            start_utc=MarketHours.ASIA_OPEN.value,
            end_utc=MarketHours.ASIA_CLOSE.value,
            start_uzb=self._utc_to_uzb_time(MarketHours.ASIA_OPEN.value),
            end_uzb=self._utc_to_uzb_time(MarketHours.ASIA_CLOSE.value)
        )
        
        # Europe session (13:00 - 21:00 UZB)
        self.sessions[TradingSession.EUROPE] = SessionInfo(
            session=TradingSession.EUROPE,
            start_utc=MarketHours.EUROPE_OPEN.value,
            end_utc=MarketHours.EUROPE_CLOSE.value,
            start_uzb=self._utc_to_uzb_time(MarketHours.EUROPE_OPEN.value),
            end_uzb=self._utc_to_uzb_time(MarketHours.EUROPE_CLOSE.value)
        )
        
        # US session (18:00 - 03:00 UZB)
        self.sessions[TradingSession.US] = SessionInfo(
            session=TradingSession.US,
            start_utc=MarketHours.US_OPEN.value,
            end_utc=MarketHours.US_CLOSE.value,
            start_uzb=self._utc_to_uzb_time(MarketHours.US_OPEN.value),
            end_uzb=self._utc_to_uzb_time(MarketHours.US_CLOSE.value)
        )
        
        # Crypto 24/7
        self.sessions[TradingSession.CRYPTO] = SessionInfo(
            session=TradingSession.CRYPTO,
            start_utc=time(0, 0),
            end_utc=time(23, 59),
            start_uzb=time(5, 0),
            end_uzb=time(4, 59),
            is_active=True  # Har doim aktiv
        )
        
    def _initialize_kill_zones(self):
        """ICT Kill Zone vaqtlarini sozlash"""
        # Asian Kill Zone (07:00 - 09:00 UTC = 12:00 - 14:00 UZB)
        self.kill_zones.append(KillZone(
            name="Asian Kill Zone",
            start_time=time(7, 0),
            end_time=time(9, 0),
            session=TradingSession.ASIA,
            importance="MEDIUM"
        ))
        
        # London Open Kill Zone (08:00 - 10:00 UTC = 13:00 - 15:00 UZB)
        self.kill_zones.append(KillZone(
            name="London Open Kill Zone",
            start_time=time(8, 0),
            end_time=time(10, 0),
            session=TradingSession.EUROPE,
            importance="HIGH"
        ))
        
        # New York Open Kill Zone (13:00 - 15:00 UTC = 18:00 - 20:00 UZB)
        self.kill_zones.append(KillZone(
            name="New York Open Kill Zone",
            start_time=time(13, 0),
            end_time=time(15, 0),
            session=TradingSession.US,
            importance="HIGH"
        ))
        
        # London Close Kill Zone (15:00 - 17:00 UTC = 20:00 - 22:00 UZB)
        self.kill_zones.append(KillZone(
            name="London Close Kill Zone",
            start_time=time(15, 0),
            end_time=time(17, 0),
            session=TradingSession.EUROPE,
            importance="MEDIUM"
        ))
        
    def now_utc(self) -> datetime:
        """Hozirgi UTC vaqt"""
        return datetime.now(self.utc_tz)
        
    def now_uzb(self) -> datetime:
        """Hozirgi O'zbekiston vaqti"""
        return datetime.now(self.uzb_tz)
        
    def utc_to_uzb(self, dt: datetime) -> datetime:
        """UTC dan UZB vaqtiga o'tkazish"""
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        return dt.astimezone(self.uzb_tz)
        
    def uzb_to_utc(self, dt: datetime) -> datetime:
        """UZB dan UTC vaqtiga o'tkazish"""
        if dt.tzinfo is None:
            dt = self.uzb_tz.localize(dt)
        return dt.astimezone(self.utc_tz)
        
    def _utc_to_uzb_time(self, utc_time: time) -> time:
        """UTC time dan UZB time ga o'tkazish"""
        # Create datetime for conversion
        dt = datetime.combine(date.today(), utc_time)
        dt = self.utc_tz.localize(dt)
        uzb_dt = dt.astimezone(self.uzb_tz)
        return uzb_dt.time()
        
    def get_current_session(self) -> Optional[TradingSession]:
        """Hozirgi trading sessiyani olish"""
        current_time = self.now_utc().time()
        
        for session, info in self.sessions.items():
            if session == TradingSession.CRYPTO:
                continue  # Skip crypto, it's always active
                
            # Handle sessions that cross midnight
            if info.start_utc <= info.end_utc:
                if info.start_utc <= current_time <= info.end_utc:
                    return session
            else:  # US session crosses midnight
                if current_time >= info.start_utc or current_time <= info.end_utc:
                    return session
                    
        return TradingSession.CRYPTO  # Default to crypto if no session active
        
    def is_kill_zone_active(self) -> Tuple[bool, Optional[KillZone]]:
        """Kill zone aktivligini tekshirish"""
        current_time = self.now_utc().time()
        
        for kz in self.kill_zones:
            if kz.start_time <= current_time <= kz.end_time:
                return True, kz
                
        return False, None
        
    def get_session_volatility(self, session: TradingSession) -> str:
        """Sessiya volatilligini olish"""
        # Session overlap = higher volatility
        current_sessions = self.get_active_sessions()
        
        if len(current_sessions) >= 2:
            return "HIGH"
        elif self.is_kill_zone_active()[0]:
            return "HIGH"
        else:
            return "NORMAL"
            
    def get_active_sessions(self) -> List[TradingSession]:
        """Aktiv sessiyalarni olish"""
        active = []
        current_time = self.now_utc().time()
        
        for session, info in self.sessions.items():
            if session == TradingSession.CRYPTO:
                active.append(session)
                continue
                
            # Check if session is active
            if info.start_utc <= info.end_utc:
                if info.start_utc <= current_time <= info.end_utc:
                    active.append(session)
            else:  # Handle midnight crossing
                if current_time >= info.start_utc or current_time <= info.end_utc:
                    active.append(session)
                    
        return active
        
    def get_next_kill_zone(self) -> Optional[Dict[str, Any]]:
        """Keyingi kill zone ni olish"""
        current_time = self.now_utc().time()
        next_kz = None
        min_minutes = float('inf')
        
        for kz in self.kill_zones:
            # Calculate minutes until kill zone
            kz_start_minutes = kz.start_time.hour * 60 + kz.start_time.minute
            current_minutes = current_time.hour * 60 + current_time.minute
            
            if kz_start_minutes > current_minutes:
                diff = kz_start_minutes - current_minutes
            else:
                # Next day
                diff = (24 * 60) - current_minutes + kz_start_minutes
                
            if diff < min_minutes:
                min_minutes = diff
                next_kz = kz
                
        if next_kz:
            return {
                "kill_zone": next_kz,
                "minutes_until": min_minutes,
                "time_uzb": self._utc_to_uzb_time(next_kz.start_time)
            }
            
        return None
        
    def is_trading_hours(self, symbol: str = "BTCUSDT") -> bool:
        """Trading soatlari ichidami tekshirish"""
        # Crypto is always tradeable
        if "USDT" in symbol or "BUSD" in symbol:
            return True
            
        # For other assets, check session
        current_session = self.get_current_session()
        return current_session is not None
        
    def get_market_schedule(self) -> Dict[str, Any]:
        """Bozor jadvalini olish"""
        schedule = {
            "current_time_utc": self.now_utc().strftime("%H:%M:%S"),
            "current_time_uzb": self.now_uzb().strftime("%H:%M:%S"),
            "active_sessions": [],
            "next_session": None,
            "current_kill_zone": None,
            "next_kill_zone": None
        }
        
        # Active sessions
        for session in self.get_active_sessions():
            info = self.sessions[session]
            schedule["active_sessions"].append({
                "session": session.name,
                "start_uzb": info.start_uzb.strftime("%H:%M"),
                "end_uzb": info.end_uzb.strftime("%H:%M"),
                "volatility": self.get_session_volatility(session)
            })
            
        # Current kill zone
        is_kz, kz = self.is_kill_zone_active()
        if is_kz:
            schedule["current_kill_zone"] = {
                "name": kz.name,
                "end_time_uzb": self._utc_to_uzb_time(kz.end_time).strftime("%H:%M"),
                "importance": kz.importance
            }
            
        # Next kill zone
        next_kz = self.get_next_kill_zone()
        if next_kz:
            schedule["next_kill_zone"] = {
                "name": next_kz["kill_zone"].name,
                "start_time_uzb": next_kz["time_uzb"].strftime("%H:%M"),
                "minutes_until": next_kz["minutes_until"],
                "importance": next_kz["kill_zone"].importance
            }
            
        return schedule
        
    def format_timestamp(self, dt: datetime, include_date: bool = False) -> str:
        """Vaqtni formatlash"""
        if dt.tzinfo is None:
            dt = self.uzb_tz.localize(dt)
        else:
            dt = dt.astimezone(self.uzb_tz)
            
        if include_date:
            return dt.strftime("%d.%m.%Y %H:%M:%S")
        else:
            return dt.strftime("%H:%M:%S")
            
    def get_trading_day_summary(self) -> Dict[str, Any]:
        """Trading kun xulosasi"""
        today = self.now_uzb().date()
        
        summary = {
            "date": today.strftime("%d.%m.%Y"),
            "day_of_week": today.strftime("%A"),
            "sessions": [],
            "kill_zones": [],
            "best_trading_hours": []
        }
        
        # Session times
        for session, info in self.sessions.items():
            if session != TradingSession.CRYPTO:
                summary["sessions"].append({
                    "name": session.name,
                    "start": info.start_uzb.strftime("%H:%M"),
                    "end": info.end_uzb.strftime("%H:%M")
                })
                
        # Kill zones
        for kz in self.kill_zones:
            summary["kill_zones"].append({
                "name": kz.name,
                "start": self._utc_to_uzb_time(kz.start_time).strftime("%H:%M"),
                "end": self._utc_to_uzb_time(kz.end_time).strftime("%H:%M"),
                "importance": kz.importance
            })
            
        # Best trading hours (session overlaps)
        summary["best_trading_hours"] = [
            "13:00 - 15:00 (London Open)",
            "18:00 - 20:00 (New York Open)",
            "20:00 - 22:00 (London Close)"
        ]
        
        return summary
        
    async def wait_for_kill_zone(self):
        """Keyingi kill zone ni kutish"""
        next_kz = self.get_next_kill_zone()
        
        if next_kz:
            minutes = next_kz["minutes_until"]
            kz_name = next_kz["kill_zone"].name
            
            logger.info(f"⏳ {kz_name} gacha {minutes} daqiqa kutilmoqda...")
            
            if minutes > 60:
                # Long wait, check every 30 minutes
                while minutes > 60:
                    await asyncio.sleep(1800)  # 30 minutes
                    next_kz = self.get_next_kill_zone()
                    minutes = next_kz["minutes_until"] if next_kz else 0
                    
            # Final wait
            await asyncio.sleep(minutes * 60)
            logger.info(f"🎯 {kz_name} boshlandi!")
            
    def is_weekend(self) -> bool:
        """Dam olish kunlarini tekshirish"""
        current_day = self.now_uzb().weekday()
        # 5 = Saturday, 6 = Sunday
        return current_day in [5, 6]
        
    def get_time_until_market_open(self) -> Optional[timedelta]:
        """Bozor ochilishigacha vaqt"""
        if not self.is_weekend():
            return None
            
        # Find Monday 00:00 UTC (05:00 UZB)
        now = self.now_utc()
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:  # Already Monday
            return None
            
        monday = now + timedelta(days=days_until_monday)
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return monday - now

# Global instance
timezone_handler = TimezoneHandler()
