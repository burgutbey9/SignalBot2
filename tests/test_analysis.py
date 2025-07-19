"""
Analysis Tests
ICT analyzer, SMT analyzer, order flow, sentiment analysis tests
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.analysis.ict_analyzer import ICTAnalyzer, ICTSignal, MarketStructure
from src.analysis.smt_analyzer import SMTAnalyzer, WhalePhase, OnChainMetrics
from src.analysis.order_flow import OrderFlowAnalyzer, FlowImbalance, LiquidityLevel
from src.analysis.sentiment import SentimentAnalyzer, SentimentScore, NewsImpact

class TestICTAnalyzer:
    """ICT Analyzer testlari"""
    
    @pytest.fixture
    def ict_analyzer(self):
        """ICTAnalyzer instance"""
        return ICTAnalyzer()
        
    @pytest.fixture
    def sample_candles(self):
        """Test uchun sample OHLCV data"""
        return [
            {"open": 45000, "high": 45500, "low": 44800, "close": 45200, "volume": 1000, "timestamp": datetime.now() - timedelta(hours=5)},
            {"open": 45200, "high": 45600, "low": 45000, "close": 45400, "volume": 1200, "timestamp": datetime.now() - timedelta(hours=4)},
            {"open": 45400, "high": 45800, "low": 45200, "close": 45300, "volume": 800, "timestamp": datetime.now() - timedelta(hours=3)},
            {"open": 45300, "high": 45400, "low": 44900, "close": 45000, "volume": 1500, "timestamp": datetime.now() - timedelta(hours=2)},
            {"open": 45000, "high": 45200, "low": 44700, "close": 44800, "volume": 2000, "timestamp": datetime.now() - timedelta(hours=1)},
        ]
        
    def test_pdh_pdl_detection(self, ict_analyzer, sample_candles):
        """Previous Day High/Low detection testi"""
        pdh, pdl = ict_analyzer.calculate_pdh_pdl(sample_candles)
        
        assert pdh == 45800  # Highest high
        assert pdl == 44700  # Lowest low
        
    def test_ssl_bsl_identification(self, ict_analyzer, sample_candles):
        """Sell Side / Buy Side Liquidity testi"""
        ssl_levels, bsl_levels = ict_analyzer.identify_ssl_bsl(sample_candles)
        
        assert len(ssl_levels) > 0
        assert len(bsl_levels) > 0
        assert all(level > 44000 for level in ssl_levels)
        assert all(level < 46000 for level in bsl_levels)
        
    def test_fair_value_gap(self, ict_analyzer, sample_candles):
        """Fair Value Gap detection testi"""
        fvg_zones = ict_analyzer.detect_fair_value_gaps(sample_candles, min_gap_size=50)
        
        # Check if FVG detection works
        for fvg in fvg_zones:
            assert fvg["type"] in ["bullish", "bearish"]
            assert fvg["high"] > fvg["low"]
            assert fvg["size"] >= 50
            
    def test_order_blocks(self, ict_analyzer, sample_candles):
        """Order Block detection testi"""
        order_blocks = ict_analyzer.find_order_blocks(sample_candles, lookback=10)
        
        for block in order_blocks:
            assert block["type"] in ["bullish", "bearish"]
            assert block["strength"] > 0
            assert block["volume"] > 0
            
    def test_market_structure(self, ict_analyzer, sample_candles):
        """Market structure analysis testi"""
        structure = ict_analyzer.analyze_market_structure(sample_candles)
        
        assert structure["trend"] in ["bullish", "bearish", "ranging"]
        assert structure["swing_high"] is not None
        assert structure["swing_low"] is not None
        assert "break_of_structure" in structure
        
    def test_kill_zones(self, ict_analyzer):
        """Kill zones identification testi"""
        current_time = datetime.now()
        
        # Test Asia session
        asia_time = current_time.replace(hour=3, minute=0)
        assert ict_analyzer.is_kill_zone(asia_time, "asia") is True
        
        # Test London session
        london_time = current_time.replace(hour=11, minute=0)
        assert ict_analyzer.is_kill_zone(london_time, "london") is True
        
        # Test New York session
        ny_time = current_time.replace(hour=17, minute=0)
        assert ict_analyzer.is_kill_zone(ny_time, "newyork") is True
        
        # Test non-kill zone time
        off_time = current_time.replace(hour=6, minute=0)
        assert ict_analyzer.is_kill_zone(off_time, "any") is False
        
    @pytest.mark.asyncio
    async def test_generate_ict_signal(self, ict_analyzer, sample_candles):
        """ICT signal generation testi"""
        signal = await ict_analyzer.generate_signal("BTCUSDT", sample_candles)
        
        assert signal is not None
        assert signal.symbol == "BTCUSDT"
        assert signal.direction in ["BUY", "SELL", "NEUTRAL"]
        assert 0 <= signal.confidence <= 100
        assert len(signal.reasons) > 0

class TestSMTAnalyzer:
    """SMT Analyzer testlari"""
    
    @pytest.fixture
    def smt_analyzer(self):
        """SMTAnalyzer instance"""
        return SMTAnalyzer()
        
    @pytest.fixture
    def whale_transactions(self):
        """Whale transaction data"""
        return [
            {"from": "exchange", "to": "cold_wallet", "amount": 1000, "coin": "BTC", "timestamp": datetime.now() - timedelta(hours=2)},
            {"from": "cold_wallet", "to": "exchange", "amount": 500, "coin": "BTC", "timestamp": datetime.now() - timedelta(hours=1)},
            {"from": "exchange", "to": "cold_wallet", "amount": 2000, "coin": "BTC", "timestamp": datetime.now()},
        ]
        
    def test_whale_phase_detection(self, smt_analyzer, whale_transactions):
        """Whale phase detection testi"""
        phase = smt_analyzer.detect_whale_phase(whale_transactions)
        
        assert phase in [WhalePhase.ACCUMULATION, WhalePhase.DISTRIBUTION, WhalePhase.MANIPULATION]
        
        # Net accumulation case
        net_flow = sum(tx["amount"] if tx["to"] == "cold_wallet" else -tx["amount"] for tx in whale_transactions)
        if net_flow > 1000:
            assert phase == WhalePhase.ACCUMULATION
            
    def test_on_chain_analysis(self, smt_analyzer):
        """On-chain metrics analysis testi"""
        metrics = OnChainMetrics(
            exchange_inflow=5000,
            exchange_outflow=8000,
            large_transactions=150,
            active_addresses=50000,
            network_fees=100,
            hash_rate=300
        )
        
        analysis = smt_analyzer.analyze_on_chain_metrics(metrics)
        
        assert "sentiment" in analysis
        assert analysis["sentiment"] in ["bullish", "bearish", "neutral"]
        assert "strength" in analysis
        assert 0 <= analysis["strength"] <= 100
        
    def test_manipulation_detection(self, smt_analyzer):
        """Market manipulation detection testi"""
        price_data = [
            {"price": 45000, "volume": 1000, "timestamp": datetime.now() - timedelta(minutes=10)},
            {"price": 45500, "volume": 5000, "timestamp": datetime.now() - timedelta(minutes=5)},  # Spike
            {"price": 45100, "volume": 1200, "timestamp": datetime.now()},
        ]
        
        is_manipulation = smt_analyzer.detect_manipulation(price_data)
        
        # High volume spike suggests manipulation
        assert is_manipulation is True
        
    def test_whale_accumulation_score(self, smt_analyzer, whale_transactions):
        """Whale accumulation score testi"""
        score = smt_analyzer.calculate_accumulation_score(whale_transactions)
        
        assert 0 <= score <= 100
        
        # More inflows should increase score
        inflow_heavy = [
            {"from": "exchange", "to": "cold_wallet", "amount": 5000, "coin": "BTC", "timestamp": datetime.now()}
        ]
        high_score = smt_analyzer.calculate_accumulation_score(inflow_heavy)
        assert high_score > 70
        
    @pytest.mark.asyncio
    async def test_generate_smt_signal(self, smt_analyzer, whale_transactions):
        """SMT signal generation testi"""
        with patch.object(smt_analyzer, 'get_on_chain_data', return_value=whale_transactions):
            signal = await smt_analyzer.generate_signal("BTCUSDT")
            
            assert signal is not None
            assert signal.phase in [WhalePhase.ACCUMULATION, WhalePhase.DISTRIBUTION, WhalePhase.MANIPULATION]
            assert 0 <= signal.confidence <= 100
            assert len(signal.whale_activities) > 0

class TestOrderFlowAnalyzer:
    """Order Flow Analyzer testlari"""
    
    @pytest.fixture
    def order_flow_analyzer(self):
        """OrderFlowAnalyzer instance"""
        return OrderFlowAnalyzer()
        
    @pytest.fixture
    def order_book_data(self):
        """Sample order book data"""
        return {
            "bids": [
                [45000, 10],
                [44990, 15],
                [44980, 20],
                [44970, 25],
                [44960, 30],
            ],
            "asks": [
                [45010, 12],
                [45020, 18],
                [45030, 22],
                [45040, 28],
                [45050, 35],
            ]
        }
        
    def test_order_flow_imbalance(self, order_flow_analyzer, order_book_data):
        """Order flow imbalance calculation testi"""
        imbalance = order_flow_analyzer.calculate_flow_imbalance(order_book_data)
        
        assert -100 <= imbalance.ratio <= 100
        assert imbalance.bid_volume > 0
        assert imbalance.ask_volume > 0
        
        # Check direction
        if imbalance.bid_volume > imbalance.ask_volume:
            assert imbalance.direction == "bullish"
        else:
            assert imbalance.direction == "bearish"
            
    def test_liquidity_levels(self, order_flow_analyzer, order_book_data):
        """Liquidity level detection testi"""
        levels = order_flow_analyzer.find_liquidity_levels(order_book_data, threshold=20)
        
        assert len(levels) > 0
        
        for level in levels:
            assert level.price > 0
            assert level.volume >= 20
            assert level.side in ["bid", "ask"]
            
    def test_volume_profile(self, order_flow_analyzer):
        """Volume profile analysis testi"""
        trades = [
            {"price": 45000, "volume": 100, "timestamp": datetime.now() - timedelta(minutes=5)},
            {"price": 45010, "volume": 150, "timestamp": datetime.now() - timedelta(minutes=4)},
            {"price": 45005, "volume": 200, "timestamp": datetime.now() - timedelta(minutes=3)},
            {"price": 44995, "volume": 180, "timestamp": datetime.now() - timedelta(minutes=2)},
            {"price": 45000, "volume": 220, "timestamp": datetime.now() - timedelta(minutes=1)},
        ]
        
        profile = order_flow_analyzer.calculate_volume_profile(trades, bins=5)
        
        assert len(profile) > 0
        assert "poc" in profile  # Point of Control
        assert "value_area_high" in profile
        assert "value_area_low" in profile
        
    def test_aggressive_orders(self, order_flow_analyzer):
        """Aggressive order detection testi"""
        orders = [
            {"type": "market", "side": "buy", "size": 1000, "timestamp": datetime.now()},
            {"type": "limit", "side": "sell", "size": 500, "timestamp": datetime.now()},
            {"type": "market", "side": "buy", "size": 2000, "timestamp": datetime.now()},
        ]
        
        aggressive = order_flow_analyzer.detect_aggressive_orders(orders, size_threshold=1000)
        
        assert len(aggressive) == 2
        assert all(order["size"] >= 1000 for order in aggressive)
        
    @pytest.mark.asyncio
    async def test_dex_flow_analysis(self, order_flow_analyzer):
        """DEX flow analysis testi"""
        mock_dex_data = {
            "swaps": [
                {"from_token": "USDT", "to_token": "ETH", "amount_usd": 50000},
                {"from_token": "ETH", "to_token": "USDT", "amount_usd": 30000},
            ],
            "liquidity_changes": [
                {"type": "add", "amount_usd": 100000},
                {"type": "remove", "amount_usd": 20000},
            ]
        }
        
        with patch.object(order_flow_analyzer, 'get_dex_data', return_value=mock_dex_data):
            analysis = await order_flow_analyzer.analyze_dex_flow("ETH")
            
            assert "net_flow" in analysis
            assert "liquidity_trend" in analysis
            assert analysis["net_flow"] == 20000  # 50k in - 30k out

class TestSentimentAnalyzer:
    """Sentiment Analyzer testlari"""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """SentimentAnalyzer instance"""
        return SentimentAnalyzer()
        
    def test_text_sentiment_analysis(self, sentiment_analyzer):
        """Text sentiment analysis testi"""
        positive_text = "Bitcoin is going to the moon! Bullish on BTC!"
        negative_text = "Crypto crash incoming. Sell everything!"
        neutral_text = "Bitcoin trading at $45,000 today."
        
        pos_sentiment = sentiment_analyzer.analyze_text_sentiment(positive_text)
        assert pos_sentiment.score > 0.5
        assert pos_sentiment.label == "positive"
        
        neg_sentiment = sentiment_analyzer.analyze_text_sentiment(negative_text)
        assert neg_sentiment.score < -0.5
        assert neg_sentiment.label == "negative"
        
        neu_sentiment = sentiment_analyzer.analyze_text_sentiment(neutral_text)
        assert -0.3 <= neu_sentiment.score <= 0.3
        assert neu_sentiment.label == "neutral"
        
    def test_news_impact_calculation(self, sentiment_analyzer):
        """News impact calculation testi"""
        high_impact_news = {
            "title": "US Approves Bitcoin ETF
