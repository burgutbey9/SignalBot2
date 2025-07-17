"""
API Fallback System Configuration
3-level backup tizimi, automatic failover, health monitoring
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import time
from collections import defaultdict
import json
from pathlib import Path

class HealthStatus(Enum):
    """API sog'liq holati"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    DEAD = auto()

class FallbackStrategy(Enum):
    """Fallback strategiyalari"""
    ROUND_ROBIN = auto()     # Ketma-ket almashtirish
    PRIORITY = auto()        # Prioritet bo'yicha
    LEAST_LOADED = auto()    # Eng kam yuklangan
    FASTEST = auto()         # Eng tez javob bergan
    WEIGHTED = auto()        # Og'irlik bo'yicha

@dataclass
class APIHealth:
    """API sog'liq holati ma'lumotlari"""
    provider: str
    status: HealthStatus = HealthStatus.HEALTHY
    success_rate: float = 100.0
    avg_response_time: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_open: bool = False
    circuit_breaker_until: Optional[datetime] = None

@dataclass
class FallbackChain:
    """Fallback zanjiri"""
    service_type: str  # order_flow, sentiment, news, market
    primary: str
    fallbacks: List[str] = field(default_factory=list)
    strategy: FallbackStrategy = FallbackStrategy.PRIORITY
    weights: Dict[str, float] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 daqiqa

class FallbackManager:
    """API Fallback Manager - 3-level backup system"""
    _instance: Optional['FallbackManager'] = None
    
    def __new__(cls) -> 'FallbackManager':
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if hasattr(self, '_initialized'): return
        self._initialized = True
        self._chains: Dict[str, FallbackChain] = {}
        self._health_status: Dict[str, APIHealth] = {}
        self._request_history: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._config_file = Path(__file__).parent / "fallback_chains.json"
        self._init_default_chains()
        
    def _init_default_chains(self) -> None:
        """Default fallback zanjirlarini sozlash"""
        # Order Flow APIs
        self._chains["order_flow"] = FallbackChain(
            service_type="order_flow",
            primary="1inch",
            fallbacks=["alchemy", "thegraph"],
            strategy=FallbackStrategy.FASTEST,
            weights={"1inch": 0.5, "alchemy": 0.3, "thegraph": 0.2},
            timeout=20.0
        )
        
        # Sentiment Analysis APIs
        self._chains["sentiment"] = FallbackChain(
            service_type="sentiment",
            primary="huggingface",
            fallbacks=["gemini1", "gemini2", "gemini3", "gemini4", "gemini5", "claude"],
            strategy=FallbackStrategy.ROUND_ROBIN,
            weights={"huggingface": 0.4, "gemini1": 0.2, "gemini2": 0.15, "gemini3": 0.1, "gemini4": 0.1, "claude": 0.05},
            timeout=45.0
        )
        
        # News APIs
        self._chains["news"] = FallbackChain(
            service_type="news",
            primary="newsapi",
            fallbacks=["reddit", "cryptopanic"],
            strategy=FallbackStrategy.PRIORITY,
            timeout=15.0
        )
        
        # Market Data APIs
        self._chains["market"] = FallbackChain(
            service_type="market",
            primary="ccxt",
            fallbacks=["coingecko", "binance"],
            strategy=FallbackStrategy.LEAST_LOADED,
            timeout=10.0
        )
        
        # API sog'liq holatini boshlang'ich sozlash
        for chain in self._chains.values():
            self._health_status[chain.primary] = APIHealth(provider=chain.primary)
            for fb in chain.fallbacks:
                self._health_status[fb] = APIHealth(provider=fb)
                
    async def get_provider(self, service_type: str) -> Tuple[str, Dict[str, Any]]:
        """Eng yaxshi API provider tanlash"""
        async with self._lock:
            chain = self._chains.get(service_type)
            if not chain: raise ValueError(f"Unknown service type: {service_type}")
            
            # Circuit breaker tekshirish
            all_providers = [chain.primary] + chain.fallbacks
            available = [p for p in all_providers if not self._is_circuit_open(p)]
            
            if not available: 
                # Barcha providerlar circuit breaker holatida - eng yaxshisini qayta ochish
                self._reset_best_provider(all_providers)
                available = all_providers
                
            # Strategy bo'yicha provider tanlash
            if chain.strategy == FallbackStrategy.ROUND_ROBIN:
                provider = await self._round_robin_select(available, service_type)
            elif chain.strategy == FallbackStrategy.PRIORITY:
                provider = await self._priority_select(available, chain)
            elif chain.strategy == FallbackStrategy.LEAST_LOADED:
                provider = await self._least_loaded_select(available)
            elif chain.strategy == FallbackStrategy.FASTEST:
                provider = await self._fastest_select(available)
            elif chain.strategy == FallbackStrategy.WEIGHTED:
                provider = await self._weighted_select(available, chain.weights)
            else:
                provider = available[0]
                
            config = {
                "timeout": chain.timeout,
                "max_retries": chain.max_retries,
                "retry_delay": chain.retry_delay,
                "is_fallback": provider != chain.primary
            }
            
            return provider, config
            
    def _is_circuit_open(self, provider: str) -> bool:
        """Circuit breaker ochiqmi tekshirish"""
        health = self._health_status.get(provider)
        if not health: return False
        if health.circuit_breaker_open:
            if health.circuit_breaker_until and datetime.now() > health.circuit_breaker_until:
                health.circuit_breaker_open = False
                health.circuit_breaker_until = None
                health.consecutive_failures = 0
                return False
            return True
        return False
        
    def _reset_best_provider(self, providers: List[str]) -> None:
        """Eng yaxshi providerni qayta ochish"""
        best_provider = min(providers, key=lambda p: self._health_status[p].failed_requests)
        health = self._health_status[best_provider]
        health.circuit_breaker_open = False
        health.circuit_breaker_until = None
        health.consecutive_failures = 0
        
    async def _round_robin_select(self, providers: List[str], service_type: str) -> str:
        """Round-robin usulida tanlash"""
        key = f"{service_type}_rr_index"
        if not hasattr(self, '_rr_indices'): self._rr_indices = {}
        idx = self._rr_indices.get(key, 0)
        provider = providers[idx % len(providers)]
        self._rr_indices[key] = idx + 1
        return provider
        
    async def _priority_select(self, providers: List[str], chain: FallbackChain) -> str:
        """Prioritet bo'yicha tanlash"""
        ordered = [chain.primary] + chain.fallbacks
        for provider in ordered:
            if provider in providers: return provider
        return providers[0]
        
    async def _least_loaded_select(self, providers: List[str]) -> str:
        """Eng kam yuklangan provider"""
        loads = {}
        for p in providers:
            recent_requests = len([t for t in self._request_history.get(p, []) if time.time() - t < 60])
            loads[p] = recent_requests
        return min(providers, key=lambda p: loads.get(p, 0))
        
    async def _fastest_select(self, providers: List[str]) -> str:
        """Eng tez javob beruvchi provider"""
        return min(providers, key=lambda p: self._health_status.get(p, APIHealth(p)).avg_response_time or float('inf'))
        
    async def _weighted_select(self, providers: List[str], weights: Dict[str, float]) -> str:
        """Og'irlik bo'yicha tanlash"""
        import random
        available_weights = {p: weights.get(p, 0.1) for p in providers}
        total = sum(available_weights.values())
        if total == 0: return providers[0]
        normalized = {p: w/total for p, w in available_weights.items()}
        rand = random.random()
        cumsum = 0
        for provider, weight in normalized.items():
            cumsum += weight
            if rand <= cumsum: return provider
        return providers[-1]
        
    async def report_success(self, provider: str, response_time: float) -> None:
        """Muvaffaqiyatli so'rovni qayd etish"""
        async with self._lock:
            health = self._health_status.get(provider)
            if not health: return
            
            health.total_requests += 1
            health.consecutive_failures = 0
            health.last_check = datetime.now()
            self._request_history[provider].append(time.time())
            
            # O'rtacha javob vaqtini hisoblash
            alpha = 0.1  # Exponential moving average factor
            health.avg_response_time = alpha * response_time + (1 - alpha) * health.avg_response_time
            
            # Success rate yangilash
            health.success_rate = ((health.total_requests - health.failed_requests) / health.total_requests) * 100
            
            # Status yangilash
            if health.success_rate >= 95: health.status = HealthStatus.HEALTHY
            elif health.success_rate >= 80: health.status = HealthStatus.DEGRADED
            else: health.status = HealthStatus.UNHEALTHY
            
    async def report_failure(self, provider: str, error: str, service_type: str) -> Optional[str]:
        """Xatolikni qayd etish va fallback tavsiya qilish"""
        async with self._lock:
            health = self._health_status.get(provider)
            if not health: return None
            
            health.total_requests += 1
            health.failed_requests += 1
            health.consecutive_failures += 1
            health.last_error = error
            health.last_check = datetime.now()
            
            # Circuit breaker tekshirish
            chain = self._chains.get(service_type)
            if chain and health.consecutive_failures >= chain.circuit_breaker_threshold:
                health.circuit_breaker_open = True
                health.circuit_breaker_until = datetime.now() + timedelta(seconds=chain.circuit_breaker_timeout)
                health.status = HealthStatus.DEAD
                
            # Success rate yangilash
            if health.total_requests > 0:
                health.success_rate = ((health.total_requests - health.failed_requests) / health.total_requests) * 100
                
            # Fallback tavsiya qilish
            if chain:
                all_providers = [chain.primary] + chain.fallbacks
                fallback_provider = None
                for fb in all_providers:
                    if fb != provider and not self._is_circuit_open(fb):
                        fallback_provider = fb
                        break
                return fallback_provider
            return None
            
    async def get_health_report(self) -> Dict[str, Any]:
        """Barcha API providerlar sog'liq hisoboti"""
        async with self._lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }
            
            for service_type, chain in self._chains.items():
                providers_health = {}
                all_providers = [chain.primary] + chain.fallbacks
                
                for provider in all_providers:
                    health = self._health_status.get(provider, APIHealth(provider))
                    providers_health[provider] = {
                        "status": health.status.name,
                        "success_rate": round(health.success_rate, 2),
                        "avg_response_time": round(health.avg_response_time, 3),
                        "consecutive_failures": health.consecutive_failures,
                        "circuit_breaker_open": health.circuit_breaker_open,
                        "last_error": health.last_error,
                        "last_check": health.last_check.isoformat() if health.last_check else None
                    }
                    
                report["services"][service_type] = {
                    "primary": chain.primary,
                    "strategy": chain.strategy.name,
                    "providers": providers_health
                }
                
            return report
            
    async def start_monitoring(self, interval: int = 60) -> None:
        """Health monitoring boshlash"""
        if self._monitoring_task and not self._monitoring_task.done():
            return
            
        async def monitor():
            while True:
                try:
                    # Eski request historyni tozalash
                    async with self._lock:
                        cutoff_time = time.time() - 3600  # 1 soatdan eski
                        for provider in list(self._request_history.keys()):
                            self._request_history[provider] = [t for t in self._request_history[provider] if t > cutoff_time]
                            
                    # Health check qilish (ixtiyoriy)
                    # await self._perform_health_checks()
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    
                await asyncio.sleep(interval)
                
        self._monitoring_task = asyncio.create_task(monitor())
        
    async def stop_monitoring(self) -> None:
        """Monitoring to'xtatish"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try: await self._monitoring_task
            except asyncio.CancelledError: pass
            
    async def save_config(self) -> bool:
        """Konfiguratsiyani saqlash"""
        try:
            config_data = {}
            for service_type, chain in self._chains.items():
                config_data[service_type] = {
                    "primary": chain.primary,
                    "fallbacks": chain.fallbacks,
                    "strategy": chain.strategy.name,
                    "weights": chain.weights,
                    "max_retries": chain.max_retries,
                    "retry_delay": chain.retry_delay,
                    "timeout": chain.timeout,
                    "circuit_breaker_threshold": chain.circuit_breaker_threshold,
                    "circuit_breaker_timeout": chain.circuit_breaker_timeout
                }
                
            with open(self._config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception: return False
        
    async def load_config(self) -> bool:
        """Konfiguratsiyani yuklash"""
        try:
            if not self._config_file.exists(): return False
            
            with open(self._config_file, 'r') as f:
                config_data = json.load(f)
                
            for service_type, cfg in config_data.items():
                self._chains[service_type] = FallbackChain(
                    service_type=service_type,
                    primary=cfg["primary"],
                    fallbacks=cfg["fallbacks"],
                    strategy=FallbackStrategy[cfg["strategy"]],
                    weights=cfg.get("weights", {}),
                    max_retries=cfg.get("max_retries", 3),
                    retry_delay=cfg.get("retry_delay", 1.0),
                    timeout=cfg.get("timeout", 30.0),
                    circuit_breaker_threshold=cfg.get("circuit_breaker_threshold", 5),
                    circuit_breaker_timeout=cfg.get("circuit_breaker_timeout", 300)
                )
            return True
        except Exception: return False

# Global instance
fallback_manager = FallbackManager()
