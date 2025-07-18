"""
Order Flow Analysis for DEX/CEX
1inch API orqali DEX order flow, DeFi activity tahlili
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import numpy as np
import pandas as pd
from decimal import Decimal

from config.config import config_manager
from utils.logger import get_logger
from utils.helpers import TimeUtils, RateLimiter, PerformanceMonitor
from api.trading_apis import unified_client

logger = get_logger(__name__)

class FlowType(Enum):
    """Order flow turlari"""
    BUY_AGGRESSIVE = auto()     # Market buy
    SELL_AGGRESSIVE = auto()    # Market sell
    BUY_PASSIVE = auto()        # Limit buy
    SELL_PASSIVE = auto()       # Limit sell
    NEUTRAL = auto()            # Balanced

class LiquidityEvent(Enum):
    """Likvidlik hodisalari"""
    LARGE_BUY = auto()          # Katta xarid
    LARGE_SELL = auto()         # Katta sotish
    LIQUIDITY_ADD = auto()      # LP qo'shildi
    LIQUIDITY_REMOVE = auto()   # LP olib tashlandi
    ARBITRAGE = auto()          # Arbitraj faoliyati

@dataclass
class OrderFlowData:
    """Order flow ma'lumotlari"""
    symbol: str
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    net_flow: float
    flow_ratio: float
    large_orders: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class DEXFlow:
    """DEX flow ma'lumotlari"""
    protocol: str  # Uniswap, Sushiswap, etc
    pair: str
    volume_24h: float
    liquidity: float
    price_impact_buy: float
    price_impact_sell: float
    swap_count: int
    unique_traders: int

@dataclass
class DeFiMetrics:
    """DeFi metrikalari"""
    total_tvl: float
    protocol_tvl: Dict[str, float]
    lending_rates: Dict[str, float]
    yield_rates: Dict[str, float]
    liquidations_24h: float
    gas_price: float

@dataclass
class OrderFlowSignal:
    """Order flow signali"""
    symbol: str
    flow_type: FlowType
    strength: float
    buy_pressure: float
    sell_pressure: float
    dex_activity: Dict[str, Any]
    defi_metrics: Dict[str, Any]
    events: List[Dict[str, Any]]
    reasoning: List[str]
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

class OrderFlowAnalyzer:
    """Order flow tahlilchisi"""
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_minute=30)
        self.performance_monitor = PerformanceMonitor()
        
        # Cache
        self._flow_cache: Dict[str, OrderFlowData] = {}
        self._dex_cache: Dict[str, List[DEXFlow]] = {}
        self._defi_cache: DeFiMetrics = None
        self._order_book_cache: Dict[str, Dict[str, Any]] = {}
        
        # Parameters
        self.large_order_threshold = 50000  # $50k
        self.flow_window = 60  # 60 minutes
        self.imbalance_threshold = 0.6  # 60% imbalance
        
    async def start(self):
        """Analyzer ishga tushirish"""
        logger.info("📊 Order Flow Analyzer ishga tushmoqda...")
        asyncio.create_task(self._monitor_dex_flows())
        asyncio.create_task(self._update_defi_metrics())
        
    async def analyze(self, symbol: str) -> Optional[OrderFlowSignal]:
        """Order flow tahlil qilish"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Get order flow data
            order_flow = await self._analyze_order_flow(symbol)
            dex_activity = await self._analyze_dex_activity(symbol)
            defi_metrics = await self._get_defi_metrics()
            events = await self._detect_flow_events(symbol, order_flow)
            
            # Generate signal
            signal = await self._generate_signal(
                symbol, order_flow, dex_activity, defi_metrics, events
            )
            
            # Monitor performance
            self.performance_monitor.record("order_flow_analysis", 1)
            
            return signal
            
        except Exception as e:
            logger.error(f"Order flow analyze xatosi {symbol}: {e}")
            return None
            
    async def _analyze_order_flow(self, symbol: str) -> OrderFlowData:
        """Order flow tahlil qilish"""
        try:
            # Get trade data from multiple sources
            trades = await unified_client.get_recent_trades(symbol, limit=1000)
            
            if not trades:
                return OrderFlowData(
                    symbol=symbol,
                    buy_volume=0,
                    sell_volume=0,
                    buy_count=0,
                    sell_count=0,
                    net_flow=0,
                    flow_ratio=0.5,
                    large_orders=[]
                )
                
            # Analyze trades
            buy_volume = 0
            sell_volume = 0
            buy_count = 0
            sell_count = 0
            large_orders = []
            
            for trade in trades:
                volume = trade['price'] * trade['quantity']
                
                if trade['is_buyer_maker']:  # Sell aggressive
                    sell_volume += volume
                    sell_count += 1
                else:  # Buy aggressive
                    buy_volume += volume
                    buy_count += 1
                    
                # Track large orders
                if volume >= self.large_order_threshold:
                    large_orders.append({
                        "side": "BUY" if not trade['is_buyer_maker'] else "SELL",
                        "volume": volume,
                        "price": trade['price'],
                        "timestamp": trade['timestamp']
                    })
                    
            # Calculate metrics
            total_volume = buy_volume + sell_volume
            net_flow = buy_volume - sell_volume
            flow_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
            
            flow_data = OrderFlowData(
                symbol=symbol,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                buy_count=buy_count,
                sell_count=sell_count,
                net_flow=net_flow,
                flow_ratio=flow_ratio,
                large_orders=large_orders
            )
            
            self._flow_cache[symbol] = flow_data
            return flow_data
            
        except Exception as e:
            logger.error(f"Order flow tahlil xatosi: {e}")
            return OrderFlowData(
                symbol=symbol,
                buy_volume=0,
                sell_volume=0,
                buy_count=0,
                sell_count=0,
                net_flow=0,
                flow_ratio=0.5,
                large_orders=[]
            )
            
    async def _analyze_dex_activity(self, symbol: str) -> Dict[str, Any]:
        """DEX faoliyatini tahlil qilish"""
        try:
            # Get DEX data from 1inch
            token = symbol.replace("USDT", "")
            dex_data = await unified_client.get_dex_liquidity(token)
            
            if not dex_data:
                return {"total_liquidity": 0, "protocols": []}
                
            # Analyze DEX flows
            total_liquidity = 0
            total_volume = 0
            protocols = []
            
            for protocol in dex_data.get('protocols', []):
                dex_flow = DEXFlow(
                    protocol=protocol['name'],
                    pair=protocol['pair'],
                    volume_24h=protocol.get('volume24h', 0),
                    liquidity=protocol.get('liquidity', 0),
                    price_impact_buy=protocol.get('priceImpactBuy', 0),
                    price_impact_sell=protocol.get('priceImpactSell', 0),
                    swap_count=protocol.get('txCount', 0),
                    unique_traders=protocol.get('uniqueTraders', 0)
                )
                
                total_liquidity += dex_flow.liquidity
                total_volume += dex_flow.volume_24h
                protocols.append(dex_flow)
                
            # Sort by liquidity
            protocols.sort(key=lambda x: x.liquidity, reverse=True)
            
            # Calculate aggregated metrics
            avg_price_impact = np.mean([p.price_impact_buy + p.price_impact_sell for p in protocols]) / 2
            
            dex_activity = {
                "total_liquidity": total_liquidity,
                "total_volume_24h": total_volume,
                "protocol_count": len(protocols),
                "avg_price_impact": avg_price_impact,
                "top_protocols": [
                    {
                        "name": p.protocol,
                        "liquidity": p.liquidity,
                        "volume": p.volume_24h,
                        "impact": (p.price_impact_buy + p.price_impact_sell) / 2
                    }
                    for p in protocols[:5]
                ]
            }
            
            # Cache DEX flows
            self._dex_cache[symbol] = protocols
            
            return dex_activity
            
        except Exception as e:
            logger.error(f"DEX activity tahlil xatosi: {e}")
            return {"total_liquidity": 0, "protocols": []}
            
    async def _get_defi_metrics(self) -> Dict[str, Any]:
        """DeFi metrikalarini olish"""
        try:
            # Check cache
            if self._defi_cache and (TimeUtils.now_uzb() - self._defi_cache.timestamp).seconds < 300:
                return self._format_defi_metrics(self._defi_cache)
                
            # Get DeFi data
            defi_data = await unified_client.get_defi_metrics()
            
            # Process TVL data
            total_tvl = defi_data.get('totalTvl', 0)
            protocol_tvl = {}
            
            for protocol in defi_data.get('protocols', []):
                protocol_tvl[protocol['name']] = protocol['tvl']
                
            # Get lending rates
            lending_data = await unified_client.get_lending_rates()
            lending_rates = {}
            
            for asset in lending_data.get('assets', []):
                lending_rates[asset['symbol']] = {
                    'supply_apy': asset.get('supplyApy', 0),
                    'borrow_apy': asset.get('borrowApy', 0),
                    'utilization': asset.get('utilization', 0)
                }
                
            # Get yield farming rates
            yield_data = await unified_client.get_yield_rates()
            yield_rates = {}
            
            for pool in yield_data.get('pools', []):
                yield_rates[pool['name']] = pool.get('apy', 0)
                
            # Get liquidation data
            liquidations = await unified_client.get_liquidations_24h()
            liquidations_volume = liquidations.get('totalVolume', 0)
            
            # Get gas price
            gas_data = await unified_client.get_gas_price()
            gas_price = gas_data.get('fast', 0)
            
            # Create metrics object
            self._defi_cache = DeFiMetrics(
                total_tvl=total_tvl,
                protocol_tvl=protocol_tvl,
                lending_rates=lending_rates,
                yield_rates=yield_rates,
                liquidations_24h=liquidations_volume,
                gas_price=gas_price,
                timestamp=TimeUtils.now_uzb()
            )
            
            return self._format_defi_metrics(self._defi_cache)
            
        except Exception as e:
            logger.error(f"DeFi metrics olish xatosi: {e}")
            return {
                "total_tvl": 0,
                "lending_rates": {},
                "yield_rates": {},
                "liquidations_24h": 0,
                "gas_price": 0
            }
            
    def _format_defi_metrics(self, metrics: DeFiMetrics) -> Dict[str, Any]:
        """DeFi metrikalarini formatlash"""
        return {
            "total_tvl": metrics.total_tvl,
            "top_protocols": sorted(
                metrics.protocol_tvl.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "lending_rates": metrics.lending_rates,
            "best_yields": sorted(
                metrics.yield_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "liquidations_24h": metrics.liquidations_24h,
            "gas_price": metrics.gas_price
        }
        
    async def _detect_flow_events(self, symbol: str, flow_data: OrderFlowData) -> List[Dict[str, Any]]:
        """Order flow hodisalarini aniqlash"""
        events = []
        
        try:
            # Large order detection
            for order in flow_data.large_orders[-10:]:  # Last 10 large orders
                event_type = LiquidityEvent.LARGE_BUY if order['side'] == 'BUY' else LiquidityEvent.LARGE_SELL
                events.append({
                    "type": event_type.name,
                    "volume": order['volume'],
                    "price": order['price'],
                    "timestamp": order['timestamp'],
                    "impact": "HIGH" if order['volume'] > self.large_order_threshold * 2 else "MEDIUM"
                })
                
            # Flow imbalance detection
            if flow_data.flow_ratio > self.imbalance_threshold:
                events.append({
                    "type": "BUY_IMBALANCE",
                    "ratio": flow_data.flow_ratio,
                    "net_flow": flow_data.net_flow,
                    "timestamp": flow_data.timestamp,
                    "impact": "HIGH"
                })
            elif flow_data.flow_ratio < (1 - self.imbalance_threshold):
                events.append({
                    "type": "SELL_IMBALANCE",
                    "ratio": flow_data.flow_ratio,
                    "net_flow": flow_data.net_flow,
                    "timestamp": flow_data.timestamp,
                    "impact": "HIGH"
                })
                
            # DEX arbitrage detection
            if symbol in self._dex_cache:
                dex_flows = self._dex_cache[symbol]
                if len(dex_flows) >= 2:
                    # Check price differences between DEXes
                    prices = [self._get_dex_price(f) for f in dex_flows[:5]]
                    if prices:
                        price_diff = (max(prices) - min(prices)) / min(prices)
                        if price_diff > 0.01:  # 1% difference
                            events.append({
                                "type": LiquidityEvent.ARBITRAGE.name,
                                "price_diff_percent": price_diff * 100,
                                "protocols": [f.protocol for f in dex_flows[:2]],
                                "timestamp": TimeUtils.now_uzb(),
                                "impact": "MEDIUM"
                            })
                            
            # Liquidity changes
            await self._detect_liquidity_changes(symbol, events)
            
        except Exception as e:
            logger.error(f"Flow events aniqlash xatosi: {e}")
            
        return events
        
    async def _detect_liquidity_changes(self, symbol: str, events: List[Dict[str, Any]]):
        """Likvidlik o'zgarishlarini aniqlash"""
        try:
            # Get liquidity history
            liquidity_changes = await unified_client.get_liquidity_changes(symbol)
            
            for change in liquidity_changes:
                if abs(change['amount_usd']) > 100000:  # $100k+
                    event_type = LiquidityEvent.LIQUIDITY_ADD if change['type'] == 'add' else LiquidityEvent.LIQUIDITY_REMOVE
                    events.append({
                        "type": event_type.name,
                        "amount_usd": change['amount_usd'],
                        "protocol": change.get('protocol', 'Unknown'),
                        "timestamp": change['timestamp'],
                        "impact": "HIGH" if abs(change['amount_usd']) > 1000000 else "MEDIUM"
                    })
                    
        except Exception as e:
            logger.error(f"Liquidity changes aniqlash xatosi: {e}")
            
    def _get_dex_price(self, dex_flow: DEXFlow) -> float:
        """DEX narxini olish"""
        # Simplified price calculation based on liquidity ratio
        # In real implementation, this would fetch actual price
        return 1.0  # Placeholder
        
    async def _generate_signal(self, symbol: str, order_flow: OrderFlowData,
                             dex_activity: Dict[str, Any], defi_metrics: Dict[str, Any],
                             events: List[Dict[str, Any]]) -> Optional[OrderFlowSignal]:
        """Order flow signalini yaratish"""
        try:
            reasoning = []
            
            # Determine flow type
            if order_flow.flow_ratio > 0.65:
                flow_type = FlowType.BUY_AGGRESSIVE
                reasoning.append(f"💹 Kuchli buy pressure: {order_flow.flow_ratio:.1%}")
            elif order_flow.flow_ratio < 0.35:
                flow_type = FlowType.SELL_AGGRESSIVE
                reasoning.append(f"📉 Kuchli sell pressure: {order_flow.flow_ratio:.1%}")
            else:
                flow_type = FlowType.NEUTRAL
                reasoning.append("⚖️ Balanced order flow")
                
            # Calculate pressures
            total_volume = order_flow.buy_volume + order_flow.sell_volume
            buy_pressure = order_flow.buy_volume / total_volume if total_volume > 0 else 0.5
            sell_pressure = order_flow.sell_volume / total_volume if total_volume > 0 else 0.5
            
            # Analyze large orders
            recent_large_buys = sum(1 for o in order_flow.large_orders if o['side'] == 'BUY')
            recent_large_sells = sum(1 for o in order_flow.large_orders if o['side'] == 'SELL')
            
            if recent_large_buys > recent_large_sells * 1.5:
                reasoning.append(f"🐋 Large buy orders ustunlik qilmoqda: {recent_large_buys}")
            elif recent_large_sells > recent_large_buys * 1.5:
                reasoning.append(f"🐳 Large sell orders ustunlik qilmoqda: {recent_large_sells}")
                
            # DEX activity analysis
            if dex_activity['total_liquidity'] > 0:
                reasoning.append(f"💧 DEX liquidity: ${dex_activity['total_liquidity']:,.0f}")
                
                if dex_activity['avg_price_impact'] > 1:
                    reasoning.append(f"⚠️ Yuqori price impact: {dex_activity['avg_price_impact']:.1f}%")
                    
            # DeFi metrics
            if defi_metrics['liquidations_24h'] > 10000000:  # $10M+
                reasoning.append(f"🔥 High liquidations: ${defi_metrics['liquidations_24h']:,.0f}")
                
            # Event analysis
            for event in events[-5:]:  # Last 5 events
                if event['type'] == 'BUY_IMBALANCE':
                    reasoning.append(f"📊 Buy imbalance: {event['ratio']:.1%}")
                elif event['type'] == 'SELL_IMBALANCE':
                    reasoning.append(f"📊 Sell imbalance: {event['ratio']:.1%}")
                elif event['type'] == LiquidityEvent.ARBITRAGE.name:
                    reasoning.append(f"🔄 Arbitrage opportunity: {event['price_diff_percent']:.2f}%")
                elif event['type'] == LiquidityEvent.LIQUIDITY_ADD.name:
                    reasoning.append(f"➕ Liquidity added: ${event['amount_usd']:,.0f}")
                elif event['type'] == LiquidityEvent.LIQUIDITY_REMOVE.name:
                    reasoning.append(f"➖ Liquidity removed: ${event['amount_usd']:,.0f}")
                    
            # Calculate signal strength
            strength = self._calculate_flow_strength(
                order_flow, dex_activity, events, flow_type
            )
            
            # Create signal
            signal = OrderFlowSignal(
                symbol=symbol,
                flow_type=flow_type,
                strength=strength,
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                dex_activity=dex_activity,
                defi_metrics=defi_metrics,
                events=events[-10:],  # Last 10 events
                reasoning=reasoning
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Order flow signal generation xatosi: {e}")
            return None
            
    def _calculate_flow_strength(self, order_flow: OrderFlowData, dex_activity: Dict[str, Any],
                               events: List[Dict[str, Any]], flow_type: FlowType) -> float:
        """Flow signal kuchini hisoblash"""
        strength = 50.0
        
        # Flow ratio impact
        if flow_type == FlowType.BUY_AGGRESSIVE:
            strength += (order_flow.flow_ratio - 0.5) * 100
        elif flow_type == FlowType.SELL_AGGRESSIVE:
            strength += (0.5 - order_flow.flow_ratio) * 100
            
        # Large order impact
        large_buy_volume = sum(o['volume'] for o in order_flow.large_orders if o['side'] == 'BUY')
        large_sell_volume = sum(o['volume'] for o in order_flow.large_orders if o['side'] == 'SELL')
        
        if large_buy_volume > large_sell_volume * 2:
            strength += 15
        elif large_sell_volume > large_buy_volume * 2:
            strength -= 15
            
        # DEX liquidity impact
        if dex_activity['total_liquidity'] > 10000000:  # $10M+
            strength += 10
        elif dex_activity['total_liquidity'] < 1000000:  # $1M-
            strength -= 10
            
        # Event impacts
        for event in events:
            if event['impact'] == 'HIGH':
                if event['type'] in ['LARGE_BUY', 'BUY_IMBALANCE', 'LIQUIDITY_ADD']:
                    strength += 5
                elif event['type'] in ['LARGE_SELL', 'SELL_IMBALANCE', 'LIQUIDITY_REMOVE']:
                    strength -= 5
                    
        # Normalize
        strength = max(0, min(100, strength))
        
        return strength
        
    async def _monitor_dex_flows(self):
        """DEX oqimlarini monitoring qilish"""
        while True:
            try:
                symbols = config_manager.trading.symbols
                
                for symbol in symbols:
                    # Monitor DEX swaps
                    swaps = await unified_client.get_recent_dex_swaps(symbol)
                    
                    for swap in swaps:
                        if swap['amount_usd'] > 10000:  # $10k+
                            logger.info(f"🔄 DEX Swap {symbol}: ${swap['amount_usd']:,.0f} on {swap['protocol']}")
                            
                    await asyncio.sleep(5)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"DEX monitoring xatosi: {e}")
                await asyncio.sleep(60)
                
    async def _update_defi_metrics(self):
        """DeFi metrikalarini yangilash"""
        while True:
            try:
                await self._get_defi_metrics()
                logger.info("📊 DeFi metrics yangilandi")
                
                # Alert on high liquidations
                if self._defi_cache and self._defi_cache.liquidations_24h > 50000000:  # $50M+
                    logger.warning(f"⚠️ High liquidations detected: ${self._defi_cache.liquidations_24h:,.0f}")
                    
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"DeFi metrics update xatosi: {e}")
                await asyncio.sleep(300)
                
    def get_flow_summary(self, symbol: str) -> Dict[str, Any]:
        """Order flow xulosasi"""
        flow = self._flow_cache.get(symbol)
        dex_flows = self._dex_cache.get(symbol, [])
        
        if not flow:
            return {"error": "No flow data available"}
            
        return {
            "symbol": symbol,
            "flow_ratio": flow.flow_ratio,
            "net_flow": flow.net_flow,
            "buy_volume": flow.buy_volume,
            "sell_volume": flow.sell_volume,
            "large_orders_count": len(flow.large_orders),
            "dex_liquidity": sum(d.liquidity for d in dex_flows),
            "timestamp": flow.timestamp.isoformat()
        }

# Global instance
order_flow_analyzer = OrderFlowAnalyzer()
