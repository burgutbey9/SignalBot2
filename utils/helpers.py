"""
Helper Utilities for Crypto Trading Bot
Rate limiting, retry logic, fallback manager, time utilities
"""
import asyncio
import time
import functools
from typing import Any, Callable, Optional, Dict, Union, List, TypeVar, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import aiohttp
from dataclasses import dataclass
import hashlib
import hmac
import pytz
from enum import Enum
import random

T = TypeVar('T')

class RateLimitExceeded(Exception):
    """Rate limit oshib ketganda xatolik"""
    pass

class RetryExhausted(Exception):
    """Retry urinishlar tugaganda xatolik"""
    pass

@dataclass
class RateLimitConfig:
    """Rate limiter konfiguratsiyasi"""
    calls: int
    period: float  # sekundlarda
    burst: Optional[int] = None  # burst capacity
    wait_on_limit: bool = True  # kutish yoki exception

class RateLimiter:
    """API rate limiter"""
    def __init__(self, calls: int = 100, period: float = 60.0, burst: Optional[int] = None):
        self.calls = calls
        self.period = period
        self.burst = burst or calls
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = asyncio.Lock()
        
    async def acquire(self, key: str = "default") -> None:
        """Rate limit tekshirish va olish"""
        async with self._lock:
            now = time.time()
            # Eski requestlarni tozalash
            self.requests[key] = deque([t for t in self.requests[key] if now - t < self.period])
            
            if len(self.requests[key]) >= self.calls:
                if len(self.requests[key]) >= self.burst:
                    oldest = self.requests[key][0]
                    wait_time = self.period - (now - oldest)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        return await self.acquire(key)
                        
            self.requests[key].append(now)
            
    def __call__(self, key: Optional[str] = None):
        """Decorator sifatida ishlatish"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.acquire(key or func.__name__)
                return await func(*args, **kwargs)
                
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                asyncio.run(self.acquire(key or func.__name__))
                return func(*args, **kwargs)
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

class RetryManager:
    """Retry logic manager"""
    def __init__(self, 
                 max_retries: int = 3,
                 delay: float = 1.0,
                 backoff: float = 2.0,
                 max_delay: float = 60.0,
                 exceptions: Tuple[type, ...] = (Exception,)):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.max_delay = max_delay
        self.exceptions = exceptions
        
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Retry bilan funksiyani bajarish"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.delay * (self.backoff ** attempt), self.max_delay)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                    
        raise RetryExhausted(f"Max retries ({self.max_retries}) exceeded") from last_exception
        
    def __call__(self, func: Optional[Callable] = None, **retry_kwargs):
        """Decorator sifatida ishlatish"""
        if func is None:
            return functools.partial(self.__call__, **retry_kwargs)
            
        retry_config = {
            'max_retries': retry_kwargs.get('max_retries', self.max_retries),
            'delay': retry_kwargs.get('delay', self.delay),
            'backoff': retry_kwargs.get('backoff', self.backoff),
            'max_delay': retry_kwargs.get('max_delay', self.max_delay),
            'exceptions': retry_kwargs.get('exceptions', self.exceptions)
        }
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_manager = RetryManager(**retry_config)
            return await retry_manager.execute(func, *args, **kwargs)
            
        return wrapper

class TimeUtils:
    """Vaqt bilan ishlash utilitilari"""
    UZB_TZ = pytz.timezone('Asia/Tashkent')
    
    @staticmethod
    def now_uzb() -> datetime:
        """Hozirgi vaqt (UZB timezone)"""
        return datetime.now(TimeUtils.UZB_TZ)
        
    @staticmethod
    def to_uzb(dt: datetime) -> datetime:
        """Vaqtni UZB timezonega o'tkazish"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(TimeUtils.UZB_TZ)
        
    @staticmethod
    def format_uzb(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """UZB vaqtini formatlash"""
        if dt is None: dt = TimeUtils.now_uzb()
        return TimeUtils.to_uzb(dt).strftime(fmt)
        
    @staticmethod
    def is_kill_zone(kill_zones: Dict[str, Dict[str, str]]) -> Tuple[bool, Optional[str]]:
        """Kill zone vaqtimi tekshirish"""
        now = TimeUtils.now_uzb()
        current_time = now.strftime("%H:%M")
        
        for zone_name, zone_info in kill_zones.items():
            start = zone_info.get("start", "00:00")
            end = zone_info.get("end", "00:00")
            
            if start <= current_time <= end:
                return True, zone_name
                
        return False, None
        
    @staticmethod
    def time_until_next_candle(timeframe: str = "5m") -> int:
        """Keyingi shamgacha qolgan vaqt (sekundlarda)"""
        intervals = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
        interval = intervals.get(timeframe, 300)
        now = time.time()
        return interval - int(now % interval)

class CryptoUtils:
    """Crypto-specific utilities"""
    
    @staticmethod
    def format_price(price: float, precision: int = 2) -> str:
        """Narxni formatlash"""
        if price >= 1000: return f"{price:,.0f}"
        elif price >= 1: return f"{price:.{precision}f}"
        else: return f"{price:.8f}".rstrip('0').rstrip('.')
        
    @staticmethod
    def calculate_position_size(balance: float, risk_pct: float, stop_loss_pct: float) -> float:
        """Pozitsiya hajmini hisoblash"""
        risk_amount = balance * (risk_pct / 100)
        position_size = risk_amount / (stop_loss_pct / 100)
        return min(position_size, balance * 0.95)  # Max 95% of balance
        
    @staticmethod
    def generate_order_id() -> str:
        """Unique order ID yaratish"""
        timestamp = int(time.time() * 1000)
        random_suffix = random.randint(1000, 9999)
        return f"ORD{timestamp}{random_suffix}"
        
    @staticmethod
    def calculate_pnl(entry_price: float, exit_price: float, quantity: float, side: str) -> Dict[str, float]:
        """PnL hisoblash"""
        if side.upper() == "BUY":
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl = (entry_price - exit_price) * quantity
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            
        return {"pnl": pnl, "pnl_percent": pnl_percent, "roi": pnl_percent}

class SignatureUtils:
    """API signature yaratish utilitilari"""
    
    @staticmethod
    def generate_signature(secret: str, params: Dict[str, Any], method: str = "HMAC-SHA256") -> str:
        """API signature yaratish"""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        if method == "HMAC-SHA256":
            return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        elif method == "HMAC-SHA512":
            return hmac.new(secret.encode(), query_string.encode(), hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unknown signature method: {method}")
            
    @staticmethod
    def add_timestamp(params: Dict[str, Any]) -> Dict[str, Any]:
        """Timestamp qo'shish"""
        params["timestamp"] = int(time.time() * 1000)
        return params

class AsyncBatcher:
    """Async operatsiyalarni batch qilish"""
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: List[Tuple[Any, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task] = None
        
    async def add(self, item: Any) -> Any:
        """Batchga element qo'shish"""
        future = asyncio.Future()
        
        async with self._lock:
            self.queue.append((item, future))
            
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
            elif not self._timer or self._timer.done():
                self._timer = asyncio.create_task(self._timeout_handler())
                
        return await future
        
    async def _timeout_handler(self) -> None:
        """Timeout handler"""
        await asyncio.sleep(self.timeout)
        async with self._lock:
            if self.queue:
                await self._process_batch()
                
    async def _process_batch(self) -> None:
        """Batchni qayta ishlash"""
        if not self.queue: return
        
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        # Process batch (override in subclass)
        results = await self.process_items([item for item, _ in batch])
        
        # Set results
        for i, (_, future) in enumerate(batch):
            if i < len(results):
                future.set_result(results[i])
            else:
                future.set_exception(Exception("No result"))
                
    async def process_items(self, items: List[Any]) -> List[Any]:
        """Override this method"""
        return items

class ExponentialBackoff:
    """Exponential backoff implementation"""
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, factor: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.factor = factor
        self.attempt = 0
        
    def next_delay(self) -> float:
        """Keyingi kutish vaqtini hisoblash"""
        delay = min(self.base_delay * (self.factor ** self.attempt), self.max_delay)
        self.attempt += 1
        return delay + random.uniform(0, delay * 0.1)  # Jitter qo'shish
        
    def reset(self) -> None:
        """Reset attempts"""
        self.attempt = 0

class CircuitBreaker:
    """Circuit breaker pattern"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Circuit breaker orqali chaqirish"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                
            raise e

class PerformanceMonitor:
    """Performance monitoring"""
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def record(self, metric: str, value: float) -> None:
        """Metrikani qayd qilish"""
        self.metrics[metric].append((time.time(), value))
        
    def get_stats(self, metric: str, window: Optional[float] = None) -> Dict[str, float]:
        """Statistika olish"""
        if metric not in self.metrics or not self.metrics[metric]:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
            
        now = time.time()
        values = [v for t, v in self.metrics[metric] if window is None or now - t <= window]
        
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
            
        values.sort()
        count = len(values)
        
        return {
            "count": count,
            "mean": sum(values) / count,
            "min": values[0],
            "max": values[-1],
            "p50": values[int(count * 0.5)],
            "p95": values[int(count * 0.95)],
            "p99": values[int(count * 0.99)]
        }
        
    @contextlib.contextmanager
    def measure(self, metric: str):
        """Context manager for timing"""
        start = time.time()
        try:
            yield
        finally:
            self.record(metric, time.time() - start)

# Global instances
rate_limiter = RateLimiter(calls=100, period=60)
retry_manager = RetryManager(max_retries=3, delay=1.0, backoff=2.0)
performance_monitor = PerformanceMonitor()

# Utility functions
async def safe_request(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Dict[str, Any]:
    """Xavfsiz HTTP request"""
    timeout = aiohttp.ClientTimeout(total=kwargs.pop('timeout', 30))
    
    async with rate_limiter.acquire(url):
        async with session.request(method, url, timeout=timeout, **kwargs) as response:
            return {
                "status": response.status,
                "headers": dict(response.headers),
                "data": await response.json() if response.content_type == 'application/json' else await response.text()
            }

def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Listni bo'laklarga bo'lish"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Nested dictni flatten qilish"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

import contextlib
