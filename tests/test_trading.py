"""
Trading Tests
Signal generator, trade analyzer, execution engine tests
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import json

from src.trading.signal_generator import SignalGenerator, TradingSignal, SignalStrength
from src.trading.trade_analyzer import TradeAnalyzer, TradeResult, StopLossReason
from src.trading.execution_engine import ExecutionEngine, OrderStatus, ExecutionMode

class TestSignalGenerator:
    """Signal Generator testlari"""
    
    @pytest.fixture
    def signal_generator(self):
        """SignalGenerator instance"""
        return SignalGenerator()
        
    @pytest.fixture
    def market_data(self):
        """Sample market data"""
        return {
            "symbol": "BTCUSDT",
            "price": 45000,
            "volume": 1000000,
            "change_24h": 2.5,
            "high_24h": 45500,
            "low_24h": 44000,
            "indicators": {
                "rsi": 45,
                "macd": {"value": 100, "signal": 80, "histogram": 20},
                "bb": {"upper": 46000, "middle": 45000, "lower": 44000},
                "ema_short": 44800,
                "ema_long": 44500
            }
        }
        
    @pytest.fixture
    def analysis_results(self):
        """Combined analysis results"""
        return {
            "ict": {
                "signal": "BUY",
                "confidence": 75,
                "reasons": ["SSL swept", "Bullish FVG", "Kill zone active"]
            },
            "smt": {
                "phase": "ACCUMULATION",
                "confidence": 80,
                "whale_activity": "High buying pressure"
            },
            "order_flow": {
                "imbalance": 25,  # 25% more buy orders
                "aggression": "bullish",
                "liquidity": "sufficient"
            },
            "sentiment": {
                "score": 0.65,
                "fear_greed": 70,
                "news_impact": "positive"
            }
        }
        
    def test_signal_generation_logic(self, signal_generator, market_data, analysis_results):
        """Signal generation logic testi"""
        signal = signal_generator.generate_signal(market_data, analysis_results)
        
        assert signal is not None
        assert signal.symbol == "BTCUSDT"
        assert signal.action in ["BUY", "SELL", "NEUTRAL"]
        assert 0 <= signal.confidence <= 100
        assert signal.entry_price > 0
        assert signal.stop_loss > 0
        assert signal.take_profit > 0
        
    def test_risk_reward_calculation(self, signal_generator):
        """Risk/Reward ratio calculation testi"""
        entry = 45000
        stop_loss = 44100  # 2% risk
        take_profit = 46800  # 4% reward
        
        rr_ratio = signal_generator.calculate_risk_reward(entry, stop_loss, take_profit)
        
        assert rr_ratio == 2.0  # 2:1 risk/reward
        
    def test_position_sizing(self, signal_generator):
        """Position sizing calculation testi"""
        account_balance = 10000
        risk_percent = 1.0
        entry_price = 45000
        stop_loss = 44100
        
        position_size = signal_generator.calculate_position_size(
            account_balance, risk_percent, entry_price, stop_loss
        )
        
        # Should risk 1% of account = $100
        # Price difference = $900 (2%)
        # Position size = $100 / 0.02 = $5000 worth
        expected_size = round(5000 / entry_price, 4)
        assert abs(position_size - expected_size) < 0.001
        
    def test_multiple_take_profits(self, signal_generator):
        """Multiple take profit levels testi"""
        entry = 45000
        
        tp_levels = signal_generator.calculate_take_profit_levels(
            entry, "BUY", targets=[1.5, 3.0, 5.0]
        )
        
        assert len(tp_levels) == 3
        assert tp_levels[0] == 45675  # 1.5% profit
        assert tp_levels[1] == 46350  # 3% profit
        assert tp_levels[2] == 47250  # 5% profit
        
    def test_signal_strength_classification(self, signal_generator, analysis_results):
        """Signal strength classification testi"""
        # Strong signal - all indicators agree
        strong_results = analysis_results.copy()
        strength = signal_generator.classify_signal_strength(strong_results)
        assert strength == SignalStrength.STRONG
        
        # Weak signal - mixed indicators
        weak_results = analysis_results.copy()
        weak_results["ict"]["confidence"] = 40
        weak_results["sentiment"]["score"] = -0.2
        strength = signal_generator.classify_signal_strength(weak_results)
        assert strength == SignalStrength.WEAK
        
    def test_signal_filtering(self, signal_generator):
        """Signal filtering testi"""
        signals = [
            TradingSignal(symbol="BTCUSDT", confidence=85, strength=SignalStrength.STRONG),
            TradingSignal(symbol="ETHUSDT", confidence=45, strength=SignalStrength.WEAK),
            TradingSignal(symbol="BNBUSDT", confidence=70, strength=SignalStrength.MEDIUM),
        ]
        
        # Filter by confidence
        filtered = signal_generator.filter_signals(signals, min_confidence=60)
        assert len(filtered) == 2
        
        # Filter by strength
        filtered = signal_generator.filter_signals(signals, min_strength=SignalStrength.MEDIUM)
        assert len(filtered) == 2
        
    @pytest.mark.asyncio
    async def test_signal_validation(self, signal_generator):
        """Signal validation testi"""
        valid_signal = TradingSignal(
            symbol="BTCUSDT",
            action="BUY",
            entry_price=45000,
            stop_loss=44100,
            take_profit=46350,
            confidence=75
        )
        
        # Valid signal
        assert await signal_generator.validate_signal(valid_signal) is True
        
        # Invalid signal - stop loss above entry for buy
        invalid_signal = valid_signal.copy()
        invalid_signal.stop_loss = 45500
        assert await signal_generator.validate_signal(invalid_signal) is False

class TestTradeAnalyzer:
    """Trade Analyzer testlari"""
    
    @pytest.fixture
    def trade_analyzer(self):
        """TradeAnalyzer instance"""
        return TradeAnalyzer()
        
    @pytest.fixture
    def completed_trade(self):
        """Completed trade data"""
        return {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 45000,
            "exit_price": 45900,
            "position_size": 0.1,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now(),
            "stop_loss": 44100,
            "take_profit": 46350,
            "exit_reason": "take_profit_1"
        }
        
    def test_pnl_calculation(self, trade_analyzer, completed_trade):
        """PnL calculation testi"""
        pnl, pnl_percent = trade_analyzer.calculate_pnl(completed_trade)
        
        # (45900 - 45000) * 0.1 = 90 USDT profit
        assert pnl == 90
        assert pnl_percent == 2.0  # 2% profit
        
    def test_trade_duration(self, trade_analyzer, completed_trade):
        """Trade duration calculation testi"""
        duration = trade_analyzer.calculate_duration(completed_trade)
        
        assert duration.total_seconds() == 7200  # 2 hours
        
    def test_win_loss_classification(self, trade_analyzer):
        """Win/Loss classification testi"""
        # Winning trade
        win_trade = {"pnl": 100, "pnl_percent": 2.5}
        assert trade_analyzer.classify_result(win_trade) == TradeResult.WIN
        
        # Losing trade
        loss_trade = {"pnl": -50, "pnl_percent": -1.2}
        assert trade_analyzer.classify_result(loss_trade) == TradeResult.LOSS
        
        # Breakeven trade
        be_trade = {"pnl": 2, "pnl_percent": 0.05}
        assert trade_analyzer.classify_result(be_trade) == TradeResult.BREAKEVEN
        
    def test_stop_loss_reason_analysis(self, trade_analyzer):
        """Stop loss reason analysis testi"""
        trade_data = {
            "exit_reason": "stop_loss",
            "market_conditions": {
                "volatility": "high",
                "trend": "reversal",
                "volume": "spike"
            },
            "price_action": {
                "rejection": True,
                "breakdown": True
            }
        }
        
        reason = trade_analyzer.analyze_stop_loss_reason(trade_data)
        
        assert reason == StopLossReason.MARKET_REVERSAL
        
    def test_trade_statistics(self, trade_analyzer):
        """Trade statistics calculation testi"""
        trades = [
            {"pnl": 100, "pnl_percent": 2.0, "duration_hours": 2},
            {"pnl": -50, "pnl_percent": -1.0, "duration_hours": 1},
            {"pnl": 150, "pnl_percent": 3.0, "duration_hours": 3},
            {"pnl": -30, "pnl_percent": -0.6, "duration_hours": 0.5},
            {"pnl": 80, "pnl_percent": 1.6, "duration_hours": 2.5},
        ]
        
        stats = trade_analyzer.calculate_statistics(trades)
        
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == 60.0
        assert stats["avg_win"] == 110  # (100+150+80)/3
        assert stats["avg_loss"] == 40   # (50+30)/2
        assert stats["profit_factor"] == 2.75  # 330/120
        
    def test_performance_metrics(self, trade_analyzer):
        """Performance metrics calculation testi"""
        trades = [
            {"pnl": 100, "entry_time": datetime.now() - timedelta(days=5)},
            {"pnl": -50, "entry_time": datetime.now() - timedelta(days=4)},
            {"pnl": 150, "entry_time": datetime.now() - timedelta(days=3)},
            {"pnl": -30, "entry_time": datetime.now() - timedelta(days=2)},
            {"pnl": 80, "entry_time": datetime.now() - timedelta(days=1)},
        ]
        
        metrics = trade_analyzer.calculate_performance_metrics(trades, initial_balance=10000)
        
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "recovery_factor" in metrics
        assert "expectancy" in metrics
        
    @pytest.mark.asyncio
    async def test_generate_trade_report(self, trade_analyzer, completed_trade):
        """Trade report generation testi"""
        report = await trade_analyzer.generate_trade_report(completed_trade)
        
        assert report is not None
        assert "summary" in report
        assert "pnl_analysis" in report
        assert "exit_analysis" in report
        assert "recommendations" in report

class TestExecutionEngine:
    """Execution Engine testlari"""
    
    @pytest.fixture
    def execution_engine(self):
        """ExecutionEngine instance"""
        return ExecutionEngine(mode=ExecutionMode.PAPER)
        
    @pytest.fixture
    def trading_signal(self):
        """Sample trading signal"""
        return TradingSignal(
            symbol="BTCUSDT",
            action="BUY",
            entry_price=45000,
            stop_loss=44100,
            take_profit=46350,
            position_size=0.1,
            confidence=80
        )
        
    @pytest.mark.asyncio
    async def test_order_placement(self, execution_engine, trading_signal):
        """Order placement testi"""
        with patch.object(execution_engine, 'place_order') as mock_place:
            mock_place.return_value = {
                "order_id": "12345",
                "status": OrderStatus.FILLED,
                "filled_price": 45010
            }
            
            result = await execution_engine.execute_signal(trading_signal)
            
            assert result["success"] is True
            assert result["order_id"] == "12345"
            assert result["status"] == OrderStatus.FILLED
            
    @pytest.mark.asyncio
    async def test_stop_loss_placement(self, execution_engine, trading_signal):
        """Stop loss order placement testi"""
        with patch.object(execution_engine, 'place_stop_loss') as mock_sl:
            mock_sl.return_value = {
                "order_id": "SL12345",
                "type": "STOP_LOSS",
                "trigger_price": 44100
            }
            
            sl_order = await execution_engine.place_stop_loss_order(
                trading_signal.symbol,
                trading_signal.stop_loss,
                trading_signal.position_size
            )
            
            assert sl_order["order_id"] == "SL12345"
            assert sl_order["trigger_price"] == 44100
            
    @pytest.mark.asyncio
    async def test_multiple_take_profits(self, execution_engine, trading_signal):
        """Multiple take profit orders testi"""
        tp_levels = [45675, 46350, 47250]
        tp_sizes = [0.03, 0.04, 0.03]  # 30%, 40%, 30%
        
        tp_orders = await execution_engine.place_take_profit_orders(
            trading_signal.symbol,
            tp_levels,
            tp_sizes
        )
        
        assert len(tp_orders) == 3
        assert sum(order["size"] for order in tp_orders) == 0.1
        
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_engine):
        """Order cancellation testi"""
        order_id = "12345"
        
        with patch.object(execution_engine, 'cancel_order') as mock_cancel:
            mock_cancel.return_value = {"success": True, "order_id": order_id}
            
            result = await execution_engine.cancel_order(order_id)
            
            assert result["success"] is True
            
    @pytest.mark.asyncio
    async def test_position_management(self, execution_engine):
        """Position management testi"""
        # Open position
        position = await execution_engine.open_position("BTCUSDT", "BUY", 0.1, 45000)
        assert position["status"] == "OPEN"
        
        # Update position
        updated = await execution_engine.update_position(
            position["id"],
            {"stop_loss": 44500}
        )
        assert updated["stop_loss"] == 44500
        
        # Close position
        closed = await execution_engine.close_position(position["id"], 45500)
        assert closed["status"] == "CLOSED"
        assert closed["pnl"] == 50  # 0.1 * (45500 - 45000)
        
    def test_risk_check(self, execution_engine):
        """Risk check testi"""
        # Check position size risk
        account_balance = 10000
        position_value = 5500  # 55% of account
        
        risk_ok = execution_engine.check_risk_limits(position_value, account_balance)
        assert risk_ok is False  # Exceeds max position size
        
        # Check daily loss limit
        daily_loss = -250  # 2.5% loss
        max_daily_loss = 3.0  # 3% limit
        
        within_limit = execution_engine.check_daily_loss_limit(daily_loss, account_balance, max_daily_loss)
        assert within_limit is True
        
    @pytest.mark.asyncio
    async def test_execution_modes(self):
        """Different execution modes testi"""
        # Paper trading mode
        paper_engine = ExecutionEngine(mode=ExecutionMode.PAPER)
        assert paper_engine.mode == ExecutionMode.PAPER
        assert await paper_engine.is_real_money() is False
        
        # Live trading mode
        live_engine = ExecutionEngine(mode=ExecutionMode.LIVE)
        assert live_engine.mode == ExecutionMode.LIVE
        assert await live_engine.is_real_money() is True
        
        # Signal only mode
        signal_engine = ExecutionEngine(mode=ExecutionMode.SIGNAL_ONLY)
        assert signal_engine.mode == ExecutionMode.SIGNAL_ONLY
        assert await signal_engine.can_execute() is False
