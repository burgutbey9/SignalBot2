"""
Trade Execution Engine
Crypto savdolarni bajarish, order management, monitoring
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from decimal import Decimal, ROUND_DOWN
import uuid

from config.config import config_manager, TradingMode
from utils.logger import get_logger
from utils.helpers import TimeUtils, RateLimiter, PerformanceMonitor
from utils.database import DatabaseManager, Trade, Order
from core.risk_manager import risk_manager
from api.trading_apis import unified_client
from trading.trade_analyzer import trade_analyzer

logger = get_logger(__name__)

class OrderType(Enum):
    """Order turlari"""
    MARKET = auto()
    LIMIT = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()

class OrderStatus(Enum):
    """Order holatlari"""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()

class ExecutionMode(Enum):
    """Bajarish rejimlari"""
    AGGRESSIVE = auto()   # Market orders, tez bajarish
    PASSIVE = auto()      # Limit orders, yaxshi narx
    SMART = auto()        # Aralash, adaptiv

@dataclass
class OrderRequest:
    """Order so'rovi"""
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    reduce_only: bool = False
    post_only: bool = False

@dataclass
class OrderResponse:
    """Order javobi"""
    order_id: str
    symbol: str
    status: OrderStatus
    executed_qty: float
    executed_price: float
    commission: float
    timestamp: datetime = field(default_factory=TimeUtils.now_uzb)

@dataclass
class Position:
    """Ochiq pozitsiya"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss_order_id: Optional[str] = None
    take_profit_order_ids: List[str] = field(default_factory=list)
    trailing_stop_enabled: bool = False
    opened_at: datetime = field(default_factory=TimeUtils.now_uzb)

class ExecutionEngine:
    """Savdo bajarish mexanizmi"""
    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.performance_monitor = PerformanceMonitor()
        
        # State management
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderRequest] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        
        # Execution settings
        self.execution_mode = ExecutionMode.SMART
        self.slippage_tolerance = 0.001  # 0.1%
        self.max_retry_attempts = 3
        self.partial_fill_threshold = 0.8  # 80% fill acceptable
        
        # Trading mode
        self.trading_mode = TradingMode.SIGNAL_ONLY
        
    async def start(self):
        """Engine ishga tushirish"""
        logger.info("🚀 Execution Engine ishga tushmoqda...")
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Load trading mode
        self.trading_mode = config_manager.trading.mode
        
        # Start monitoring
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._monitor_orders())
        
    async def execute_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Signalni bajarish"""
        try:
            # Check trading mode
            if self.trading_mode == TradingMode.SIGNAL_ONLY:
                logger.info("Signal-only rejimda, savdo bajarilmaydi")
                return {"success": False, "reason": "signal_only_mode"}
                
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Validate signal
            validation = await self._validate_signal(signal_data)
            if not validation["valid"]:
                logger.warning(f"Signal validation failed: {validation['reason']}")
                return {"success": False, "reason": validation["reason"]}
                
            # Check existing position
            if signal_data["symbol"] in self._positions:
                return await self._handle_existing_position(signal_data)
                
            # Execute new position
            result = await self._execute_new_position(signal_data)
            
            # Monitor performance
            self.performance_monitor.record("signal_executed", 1)
            
            return result
            
        except Exception as e:
            logger.error(f"Signal execution xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _validate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Signalni tekshirish"""
        try:
            # Check required fields
            required_fields = ["symbol", "action", "entry_price", "stop_loss", "position_size"]
            for field in required_fields:
                if field not in signal_data:
                    return {"valid": False, "reason": f"Missing {field}"}
                    
            # Check risk limits
            risk_check = risk_manager.calculate_position_size(
                signal_data["symbol"],
                signal_data["entry_price"],
                signal_data["stop_loss"],
                await self._get_account_balance()
            )
            
            if risk_check["size"] == 0:
                return {"valid": False, "reason": risk_check.get("error", "Risk limit exceeded")}
                
            # Check market hours
            from core.timezone_handler import timezone_handler
            if not timezone_handler.is_trading_hours(signal_data["symbol"]):
                # Allow crypto 24/7
                if "USDT" not in signal_data["symbol"]:
                    return {"valid": False, "reason": "Outside trading hours"}
                    
            # Check for conflicting orders
            if await self._has_conflicting_orders(signal_data["symbol"]):
                return {"valid": False, "reason": "Conflicting orders exist"}
                
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Signal validation xatosi: {e}")
            return {"valid": False, "reason": str(e)}
            
    async def _execute_new_position(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Yangi pozitsiya ochish"""
        try:
            symbol = signal_data["symbol"]
            side = signal_data["action"]  # BUY/SELL
            
            # Prepare order
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=signal_data["position_size"],
                order_type=OrderType.MARKET if self.execution_mode == ExecutionMode.AGGRESSIVE else OrderType.LIMIT,
                price=signal_data["entry_price"] if self.execution_mode != ExecutionMode.AGGRESSIVE else None
            )
            
            # Execute entry order
            entry_result = await self._place_order(order_request)
            
            if not entry_result["success"]:
                return entry_result
                
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_result["executed_price"],
                quantity=entry_result["executed_qty"]
            )
            
            self._positions[symbol] = position
            
            # Place stop loss
            sl_result = await self._place_stop_loss(position, signal_data["stop_loss"])
            if sl_result["success"]:
                position.stop_loss_order_id = sl_result["order_id"]
                
            # Place take profits
            tp_results = await self._place_take_profits(position, signal_data.get("take_profit", []))
            position.take_profit_order_ids = [r["order_id"] for r in tp_results if r["success"]]
            
            # Save to database
            await self._save_trade(position, signal_data)
            
            # Log execution
            logger.info(f"✅ Position opened: {symbol} {side} @ {position.entry_price}")
            
            return {
                "success": True,
                "position": self._format_position(position),
                "entry_order": entry_result,
                "stop_loss": sl_result,
                "take_profits": tp_results
            }
            
        except Exception as e:
            logger.error(f"Position execution xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _handle_existing_position(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mavjud pozitsiyani boshqarish"""
        try:
            position = self._positions[signal_data["symbol"]]
            
            # Check if opposite signal (close position)
            if (position.side == "BUY" and signal_data["action"] == "SELL") or \
               (position.side == "SELL" and signal_data["action"] == "BUY"):
                logger.info(f"Teskari signal keldi, pozitsiya yopilmoqda: {position.symbol}")
                return await self.close_position(position.symbol, reason="reversal_signal")
                
            # Update stop loss if better
            if signal_data.get("stop_loss"):
                await self._update_stop_loss(position, signal_data["stop_loss"])
                
            # Add to position if same direction
            if config_manager.risk_management.allow_position_scaling:
                return await self._scale_position(position, signal_data)
                
            return {"success": False, "reason": "Position already exists"}
            
        except Exception as e:
            logger.error(f"Existing position handling xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _place_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Order joylashtirish"""
        try:
            # Paper trading check
            if self.trading_mode == TradingMode.PAPER:
                return await self._simulate_order(order_request)
                
            # Real order placement
            order_id = str(uuid.uuid4())
            self._orders[order_id] = order_request
            self._order_status[order_id] = OrderStatus.PENDING
            
            # Retry logic
            for attempt in range(self.max_retry_attempts):
                try:
                    # Place order via API
                    result = await unified_client.place_order(
                        symbol=order_request.symbol,
                        side=order_request.side,
                        order_type=order_request.order_type.name,
                        quantity=order_request.quantity,
                        price=order_request.price
                    )
                    
                    if result.get("status") == "FILLED":
                        self._order_status[order_id] = OrderStatus.FILLED
                        
                        return {
                            "success": True,
                            "order_id": order_id,
                            "executed_qty": result.get("executed_qty", order_request.quantity),
                            "executed_price": result.get("price", order_request.price),
                            "commission": result.get("commission", 0)
                        }
                        
                except Exception as e:
                    logger.warning(f"Order placement attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
                    
            # All attempts failed
            self._order_status[order_id] = OrderStatus.REJECTED
            return {"success": False, "reason": "Order placement failed"}
            
        except Exception as e:
            logger.error(f"Order placement xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _simulate_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Paper trading uchun order simulyatsiya"""
        try:
            # Get current price
            ticker = await unified_client.get_ticker(order_request.symbol)
            current_price = ticker.get("last_price", order_request.price)
            
            # Simulate slippage
            import random
            slippage = random.uniform(-self.slippage_tolerance, self.slippage_tolerance)
            
            if order_request.side == "BUY":
                executed_price = current_price * (1 + abs(slippage))
            else:
                executed_price = current_price * (1 - abs(slippage))
                
            # Calculate commission (0.1%)
            commission = order_request.quantity * executed_price * 0.001
            
            return {
                "success": True,
                "order_id": str(uuid.uuid4()),
                "executed_qty": order_request.quantity,
                "executed_price": executed_price,
                "commission": commission
            }
            
        except Exception as e:
            logger.error(f"Order simulation xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _place_stop_loss(self, position: Position, stop_price: float) -> Dict[str, Any]:
        """Stop loss joylashtirish"""
        try:
            # Cancel existing stop loss
            if position.stop_loss_order_id:
                await self._cancel_order(position.stop_loss_order_id)
                
            # Place new stop loss
            sl_order = OrderRequest(
                symbol=position.symbol,
                side="SELL" if position.side == "BUY" else "BUY",
                quantity=position.quantity,
                order_type=OrderType.STOP_LOSS,
                stop_price=stop_price,
                reduce_only=True
            )
            
            result = await self._place_order(sl_order)
            
            if result["success"]:
                logger.info(f"Stop loss o'rnatildi: {position.symbol} @ {stop_price}")
                
            return result
            
        except Exception as e:
            logger.error(f"Stop loss placement xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _place_take_profits(self, position: Position, take_profits: List[float]) -> List[Dict[str, Any]]:
        """Take profit orderlarini joylashtirish"""
        results = []
        
        try:
            # Cancel existing take profits
            for tp_id in position.take_profit_order_ids:
                await self._cancel_order(tp_id)
                
            position.take_profit_order_ids.clear()
            
            # Place new take profits
            for i, tp_price in enumerate(take_profits[:3]):  # Max 3 TPs
                # Calculate quantity for each TP
                if len(take_profits) == 1:
                    tp_quantity = position.quantity
                elif len(take_profits) == 2:
                    tp_quantity = position.quantity * 0.5
                else:  # 3 TPs
                    if i == 0:
                        tp_quantity = position.quantity * 0.5
                    elif i == 1:
                        tp_quantity = position.quantity * 0.3
                    else:
                        tp_quantity = position.quantity * 0.2
                        
                tp_order = OrderRequest(
                    symbol=position.symbol,
                    side="SELL" if position.side == "BUY" else "BUY",
                    quantity=tp_quantity,
                    order_type=OrderType.TAKE_PROFIT,
                    price=tp_price,
                    reduce_only=True
                )
                
                result = await self._place_order(tp_order)
                results.append(result)
                
                if result["success"]:
                    logger.info(f"TP{i+1} o'rnatildi: {position.symbol} @ {tp_price}")
                    
            return results
            
        except Exception as e:
            logger.error(f"Take profit placement xatosi: {e}")
            return [{"success": False, "reason": str(e)}]
            
    async def _update_stop_loss(self, position: Position, new_stop: float):
        """Stop lossni yangilash"""
        try:
            # Check if new stop is better
            current_price = await self._get_current_price(position.symbol)
            
            if position.side == "BUY":
                # For long, new stop should be higher than current
                if new_stop > position.entry_price * 0.99:  # Min 1% below entry
                    logger.info(f"Stop loss yangilanmaydi - juda yaqin")
                    return
            else:
                # For short, new stop should be lower than current
                if new_stop < position.entry_price * 1.01:  # Min 1% above entry
                    logger.info(f"Stop loss yangilanmaydi - juda yaqin")
                    return
                    
            # Update stop loss
            await self._place_stop_loss(position, new_stop)
            
        except Exception as e:
            logger.error(f"Stop loss update xatosi: {e}")
            
    async def _scale_position(self, position: Position, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pozitsiyani kattalashtirish"""
        try:
            # Check if scaling is allowed
            max_position_value = await self._get_max_position_value()
            current_value = position.quantity * await self._get_current_price(position.symbol)
            
            if current_value >= max_position_value * 0.8:
                return {"success": False, "reason": "Position size limit reached"}
                
            # Add to position
            add_result = await self._execute_new_position(signal_data)
            
            if add_result["success"]:
                # Update position
                position.quantity += add_result["position"]["quantity"]
                # Recalculate average entry
                total_cost = position.entry_price * position.quantity + \
                           add_result["position"]["entry_price"] * add_result["position"]["quantity"]
                position.entry_price = total_cost / position.quantity
                
                logger.info(f"Position scaled: {position.symbol} new size: {position.quantity}")
                
            return add_result
            
        except Exception as e:
            logger.error(f"Position scaling xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def close_position(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """Pozitsiyani yopish"""
        try:
            if symbol not in self._positions:
                return {"success": False, "reason": "Position not found"}
                
            position = self._positions[symbol]
            
            # Cancel protective orders
            if position.stop_loss_order_id:
                await self._cancel_order(position.stop_loss_order_id)
                
            for tp_id in position.take_profit_order_ids:
                await self._cancel_order(tp_id)
                
            # Place market order to close
            close_order = OrderRequest(
                symbol=symbol,
                side="SELL" if position.side == "BUY" else "BUY",
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                reduce_only=True
            )
            
            result = await self._place_order(close_order)
            
            if result["success"]:
                # Calculate PnL
                exit_price = result["executed_price"]
                if position.side == "BUY":
                    pnl = (exit_price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - exit_price) * position.quantity
                    
                pnl -= result.get("commission", 0)
                position.realized_pnl = pnl
                
                # Analyze trade
                trade_data = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "position_size": position.quantity,
                    "entry_time": position.opened_at
                }
                
                await trade_analyzer.analyze_trade_exit(trade_data, exit_price, reason)
                
                # Remove position
                del self._positions[symbol]
                
                logger.info(f"Position yopildi: {symbol} PnL: {pnl:+.2f}")
                
                return {
                    "success": True,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": reason
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Position close xatosi: {e}")
            return {"success": False, "reason": str(e)}
            
    async def _cancel_order(self, order_id: str) -> bool:
        """Orderni bekor qilish"""
        try:
            if self.trading_mode == TradingMode.PAPER:
                self._order_status[order_id] = OrderStatus.CANCELLED
                return True
                
            # Real order cancellation
            result = await unified_client.cancel_order(order_id)
            
            if result.get("success"):
                self._order_status[order_id] = OrderStatus.CANCELLED
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Order cancel xatosi: {e}")
            return False
            
    async def _monitor_positions(self):
        """Pozitsiyalarni monitoring qilish"""
        while True:
            try:
                for symbol, position in list(self._positions.items()):
                    # Update unrealized PnL
                    current_price = await self._get_current_price(symbol)
                    
                    if position.side == "BUY":
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                        
                    # Check for trailing stop
                    if position.trailing_stop_enabled:
                        await self._update_trailing_stop(position, current_price)
                        
                    # Risk management checks
                    if position.unrealized_pnl < -1000:  # $1000 loss limit
                        logger.warning(f"Risk limit: {symbol} pozitsiya yopilmoqda")
                        await self.close_position(symbol, reason="risk_management")
                        
                await asyncio.sleep(10)  # Every 10 seconds
                
            except Exception as e:
                logger.error(f"Position monitoring xatosi: {e}")
                await asyncio.sleep(10)
                
    async def _monitor_orders(self):
        """Orderlarni monitoring qilish"""
        while True:
            try:
                for order_id, status in list(self._order_status.items()):
                    if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                        # Check order status
                        if self.trading_mode != TradingMode.PAPER:
                            result = await unified_client.get_order_status(order_id)
                            
                            if result.get("status") == "FILLED":
                                self._order_status[order_id] = OrderStatus.FILLED
                            elif result.get("status") == "CANCELLED":
                                self._order_status[order_id] = OrderStatus.CANCELLED
                                
                await asyncio.sleep(5)  # Every 5 seconds
                
            except Exception as e:
                logger.error(f"Order monitoring xatosi: {e}")
                await asyncio.sleep(5)
                
    async def _update_trailing_stop(self, position: Position, current_price: float):
        """Trailing stopni yangilash"""
        try:
            # Calculate trailing distance (1% default)
            trail_percent = 0.01
            
            if position.side == "BUY":
                # For long, trail below highest price
                new_stop = current_price * (1 - trail_percent)
                # Only update if new stop is higher
                if position.stop_loss_order_id:
                    # Get current stop price and update if better
                    pass
            else:
                # For short, trail above lowest price
                new_stop = current_price * (1 + trail_percent)
                
            # Update stop loss
            await self._place_stop_loss(position, new_stop)
            
        except Exception as e:
            logger.error(f"Trailing stop update xatosi: {e}")
            
    async def _get_current_price(self, symbol: str) -> float:
        """Hozirgi narxni olish"""
        try:
            ticker = await unified_client.get_ticker(symbol)
            return ticker.get("last_price", 0)
        except:
            return 0
            
    async def _get_account_balance(self) -> float:
        """Account balansini olish"""
        try:
            balance = await unified_client.get_balance("USDT")
            return balance.get("free", 10000)  # Default for testing
        except:
            return 10000
            
    async def _get_max_position_value(self) -> float:
        """Maksimal pozitsiya qiymatini olish"""
        balance = await self._get_account_balance()
        return balance * config_manager.risk_management.max_position_size_percent / 100
        
    async def _has_conflicting_orders(self, symbol: str) -> bool:
        """Ziddiyatli orderlar borligini tekshirish"""
        # Check if there are pending orders for this symbol
        for order_id, order in self._orders.items():
            if order.symbol == symbol and self._order_status.get(order_id) in [
                OrderStatus.PENDING, OrderStatus.SUBMITTED
            ]:
                return True
        return False
        
    async def _save_trade(self, position: Position, signal_data: Dict[str, Any]):
        """Savdoni saqlash"""
        try:
            if self.db_manager:
                trade = Trade(
                    id=str(uuid.uuid4()),
                    symbol=position.symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    position_size=position.quantity,
                    stop_loss=signal_data.get("stop_loss", 0),
                    take_profit_1=signal_data.get("take_profit", [0])[0],
                    confidence=signal_data.get("confidence", 0),
                    status="OPEN",
                    opened_at=position.opened_at
                )
                
                await self.db_manager.save_trade(trade)
                
        except Exception as e:
            logger.error(f"Trade saqlash xatosi: {e}")
            
    def _format_position(self, position: Position) -> Dict[str, Any]:
        """Pozitsiyani formatlash"""
        return {
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "opened_at": position.opened_at.isoformat()
        }
        
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Ochiq pozitsiyalarni olish"""
        return [self._format_position(p) for p in self._positions.values()]
        
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Bitta pozitsiyani olish"""
        if symbol in self._positions:
            return self._format_position(self._positions[symbol])
        return None
        
    async def enable_trailing_stop(self, symbol: str) -> bool:
        """Trailing stopni yoqish"""
        if symbol in self._positions:
            self._positions[symbol].trailing_stop_enabled = True
            logger.info(f"Trailing stop yoqildi: {symbol}")
            return True
        return False

# Global instance
execution_engine = ExecutionEngine()
