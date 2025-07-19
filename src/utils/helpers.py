"""
Helper Functions and Utilities
Yordamchi funksiyalar, vaqt, rate limiting, performance monitoring
"""
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from functools import wraps
import pytz
import numpy as np
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import aiohttp
from collections import deque, defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TimeUtils:
    """Vaqt bilan ishlash uchun yordamchi klass"""
    UZB_TZ = pytz.timezone('Asia/Tashkent')
    
    @staticmethod
    def now_utc() -> datetime:
        """Hozirgi UTC vaqt"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def now_uzb() -> datetime:
        """Hozirgi O'zbekiston vaqti"""
        return datetime.now(TimeUtils.UZB_TZ)
    
    @staticmethod
    def utc_to_uzb(dt: datetime) -> datetime:
        """UTC dan O'zbekiston vaqtiga o'tkazish"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(TimeUtils.UZB_TZ)
    
    @staticmethod
    def uzb_to_utc(dt: datetime) -> datetime:
        """O'zbekiston vaqtidan UTC ga o'tkazish"""
        if dt.tzinfo is None:
            dt = TimeUtils.UZB_TZ.localize(dt)
        return dt.astimezone(pytz.UTC)
    
    @staticmethod
    def timestamp_to_uzb(timestamp: Union[int, float]) -> datetime:
        """Unix timestamp dan O'zbekiston vaqtiga"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.astimezone(TimeUtils.UZB_TZ)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Davomiylikni formatlash"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

@dataclass
class RateLimitConfig:
    """Rate limit konfiguratsiyasi"""
    calls: int = 60
    period: float = 60.0  # seconds
    burst_allowed: bool = True
    burst_factor: float = 1.5

class RateLimiter:
    """API rate limiting"""
    def __init__(self, calls_per_minute: int = 60, burst_allowed: bool = True):
        self.config = RateLimitConfig(
            calls=calls_per_minute,
            period=60.0,
            burst_allowed=burst_allowed
        )
        self.calls = deque()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Rate limit tekshirish va kutish"""
        async with self._lock:
            now = time.time()
            
            # Eski qo'ng'iroqlarni tozalash
            cutoff = now - self.config.period
            while self.calls and self.calls[0] < cutoff:
                self.calls.popleft()
            
            # Limit tekshirish
            max_calls = self.config.calls
            if self.config.burst_allowed:
                max_calls = int(max_calls * self.config.burst_factor)
                
            if len(self.calls) >= max_calls:
                # Kutish vaqtini hisoblash
                sleep_time = self.calls[0] + self.config.period - now
                if sleep_time > 0:
                    logger.warning(f"Rate limit kutish: {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    
            # Qo'ng'iroqni qayd qilish
            self.calls.append(now)
    
    def reset(self) -> None:
        """Rate limiter ni reset qilish"""
        self.calls.clear()
        
    def get_remaining_calls(self) -> int:
        """Qolgan qo'ng'iroqlar soni"""
        now = time.time()
        cutoff = now - self.config.period
        active_calls = sum(1 for call_time in self.calls if call_time > cutoff)
        return max(0, self.config.calls - active_calls)

class PerformanceMonitor:
    """Performance monitoring"""
    def __init__(self, max_samples: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.counters: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}
        
    def start_timer(self, metric: str) -> None:
        """Vaqt o'lchashni boshlash"""
        self._start_times[metric] = time.time()
        
    def stop_timer(self, metric: str) -> float:
        """Vaqt o'lchashni to'xtatish"""
        if metric not in self._start_times:
            return 0.0
            
        elapsed = time.time() - self._start_times[metric]
        self.record(f"{metric}_time", elapsed)
        del self._start_times[metric]
        return elapsed
        
    def record(self, metric: str, value: float) -> None:
        """Metrikani qayd qilish"""
        self.metrics[metric].append({
            'value': value,
            'timestamp': time.time()
        })
        
    def increment(self, counter: str, value: int = 1) -> None:
        """Counterni oshirish"""
        self.counters[counter] += value
        
    def get_average(self, metric: str, window_seconds: Optional[float] = None) -> float:
        """O'rtacha qiymatni olish"""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0.0
            
        values = self.metrics[metric]
        
        if window_seconds:
            cutoff = time.time() - window_seconds
            values = [v for v in values if v['timestamp'] > cutoff]
            
        if not values:
            return 0.0
            
        return np.mean([v['value'] for v in values])
        
    def get_stats(self, metric: str) -> Dict[str, float]:
        """Metrika statistikasi"""
        if metric not in self.metrics or not self.metrics[metric]:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
            
        values = [v['value'] for v in self.metrics[metric]]
        
        return {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values)
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Umumiy performance xulosasi"""
        summary = {
            'metrics': {},
            'counters': dict(self.counters)
        }
        
        for metric, values in self.metrics.items():
            if values:
                summary['metrics'][metric] = self.get_stats(metric)
                
        return summary

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    exceptions: tuple = (Exception,)) -> Callable:
    """Retry decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max attempts reached for {func.__name__}: {e}")
                        raise
                        
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max attempts reached for {func.__name__}: {e}")
                        raise
                        
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class FallbackManager:
    """API fallback boshqaruvi"""
    def __init__(self):
        self.providers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_index: Dict[str, int] = defaultdict(int)
        self.failure_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.max_failures = 3
        
    def register_provider(self, service: str, provider: str, config: Dict[str, Any], priority: int = 0):
        """Provider qo'shish"""
        self.providers[service].append({
            'name': provider,
            'config': config,
            'priority': priority,
            'available': True
        })
        # Sort by priority
        self.providers[service].sort(key=lambda x: x['priority'])
        
    def get_current_provider(self, service: str) -> Optional[Dict[str, Any]]:
        """Joriy providerni olish"""
        if service not in self.providers:
            return None
            
        providers = self.providers[service]
        
        # Find first available provider
        for i, provider in enumerate(providers):
            if provider['available']:
                self.current_index[service] = i
                return provider
                
        # All failed, reset first one
        if providers:
            providers[0]['available'] = True
            self.failure_counts[service][providers[0]['name']] = 0
            self.current_index[service] = 0
            return providers[0]
            
        return None
        
    def report_failure(self, service: str, provider_name: str):
        """Xatolikni qayd qilish"""
        self.failure_counts[service][provider_name] += 1
        
        # Disable provider if too many failures
        if self.failure_counts[service][provider_name] >= self.max_failures:
            for provider in self.providers[service]:
                if provider['name'] == provider_name:
                    provider['available'] = False
                    logger.warning(f"Provider {provider_name} disabled for {service}")
                    break
                    
    def report_success(self, service: str, provider_name: str):
        """Muvaffaqiyatni qayd qilish"""
        self.failure_counts[service][provider_name] = 0
        
    def get_next_provider(self, service: str) -> Optional[Dict[str, Any]]:
        """Keyingi providerni olish"""
        current = self.get_current_provider(service)
        if not current:
            return None
            
        # Mark current as failed
        current['available'] = False
        
        # Get next available
        return self.get_current_provider(service)

def calculate_hash(data: Union[str, Dict, List]) -> str:
    """Ma'lumotlar uchun hash hisoblash"""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()

def format_number(number: float, decimals: int = 2) -> str:
    """Raqamni formatlash"""
    if abs(number) >= 1_000_000_000:
        return f"{number/1_000_000_000:.{decimals}f}B"
    elif abs(number) >= 1_000_000:
        return f"{number/1_000_000:.{decimals}f}M"
    elif abs(number) >= 1_000:
        return f"{number/1_000:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Foiz o'zgarishini hisoblash"""
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    return ((new_value - old_value) / abs(old_value)) * 100

def round_to_precision(value: float, precision: float) -> float:
    """Aniqlikka yaxlitlash"""
    if precision == 0:
        return value
    return round(value / precision) * precision

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Xavfsiz bo'lish"""
    if denominator == 0:
        return default
    return numerator / denominator

class AsyncBatcher:
    """Async so'rovlarni guruhlash"""
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: List[Tuple[Any, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task] = None
        
    async def add(self, item: Any) -> Any:
        """Elementni qo'shish va natijani kutish"""
        future = asyncio.Future()
        
        async with self._lock:
            self.queue.append((item, future))
            
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
            elif not self._timer:
                self._timer = asyncio.create_task(self._timeout_handler())
                
        return await future
        
    async def _timeout_handler(self):
        """Timeout handler"""
        await asyncio.sleep(self.timeout)
        async with self._lock:
            if self.queue:
                await self._process_batch()
            self._timer = None
            
    async def _process_batch(self):
        """Batch ni qayta ishlash"""
        if not self.queue:
            return
            
        batch = self.queue.copy()
        self.queue.clear()
        
        # Process batch (override in subclass)
        results = await self.process_batch([item for item, _ in batch])
        
        # Set results
        for i, (_, future) in enumerate(batch):
            if i < len(results):
                future.set_result(results[i])
            else:
                future.set_exception(Exception("Batch processing failed"))
                
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Batch ni qayta ishlash (override qilish kerak)"""
        raise NotImplementedError

# Global instances
fallback_manager = FallbackManager()
global_rate_limiter = RateLimiter(calls_per_minute=1200)  # 20 per second
