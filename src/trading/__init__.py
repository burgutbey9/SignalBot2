"""
Trading Module
Signal generation, trade analysis and execution
"""

from .signal_generator import (
    SignalGenerator,
    TradingSignal,
    SignalStrength,
    SignalType,
    MultiStrategySignal
)

from .trade_analyzer import (
    TradeAnalyzer,
    TradeResult,
    StopLossReason,
    TradeMetrics,
    PerformanceReport
)

from .execution_engine import (
    ExecutionEngine,
    OrderType,
    OrderStatus,
    ExecutionMode,
    PositionManager
)

__all__ = [
    # Signal Generator
    "SignalGenerator",
    "TradingSignal",
    "SignalStrength",
    "SignalType",
    "MultiStrategySignal",
    
    # Trade Analyzer
    "TradeAnalyzer",
    "TradeResult",
    "StopLossReason",
    "TradeMetrics",
    "PerformanceReport",
    
    # Execution Engine
    "ExecutionEngine",
    "OrderType",
    "OrderStatus",
    "ExecutionMode",
    "PositionManager"
]
