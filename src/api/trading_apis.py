"""
Trading APIs Client
1inch (DEX order flow), Alchemy (on-chain), CCXT (exchanges), CoinGecko (market data)
"""
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import ccxt.async_support as ccxt
import json
from dataclasses import dataclass
from enum import Enum

from config.config import config_manager, APIProvider
from config.fallback_config import fallback_manager
from utils.logger import get_logger, LogCategory
from utils.helpers import rate_limiter, retry_manager, safe_request, CryptoUtils

logger = get_logger(__name__, LogCategory.API)

class OrderFlowType(Enum):
    """Order flow turlari"""
    DEX_SWAP = "dex_swap"
    WHALE_TRANSFER = "whale_transfer"
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"

@dataclass
class OrderFlowData:
    """Order flow ma'lumotlari"""
    flow_type: OrderFlowType
    token_in: str
    token_out: str
    amount_in: Decimal
    amount_out: Decimal
    usd_value: Decimal
    from_address: str
    to_address: str
    tx_hash: str
    timestamp: datetime
    metadata: Dict[str, Any]

class OneInchClient:
    """1inch DEX aggregator API client"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.1inch.dev/fusion"
        self.swap_url = "https://api.1inch.dev/swap/v5.2"
        self.chains = {
            "ethereum": 1,
            "bsc": 56,
            "polygon": 137,
            "arbitrum": 42161,
            "optimism": 10,
            "avalanche": 43114
        }
        self._api_key = None
        
    async def _get_headers(self) -> Dict[str, str]:
        """API headers olish"""
        if not self._api_key:
            api_config = config_manager.get_api_config(APIProvider.ONEINCG)
            self._api_key = api_config.api_key if api_config else ""
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        
    @retry_manager(max_retries=3, delay=1.0)
    @rate_limiter(calls=100, period=60)
    async def get_liquidity_sources(self, chain: str = "ethereum") -> List[Dict[str, Any]]:
        """DEX liquidity manbalarini olish"""
        try:
            chain_id = self.chains.get(chain, 1)
            url = f"{self.swap_url}/{chain_id}/liquidity-sources"
            
            response = await safe_request(self.session, "GET", url, headers=await self._get_headers())
            if response["status"] == 200:
                logger.info(f"✅ 1inch liquidity sources: {chain}")
                return response["data"].get("protocols", [])
            else:
                logger.error(f"❌ 1inch liquidity error: {response['status']}")
                return []
        except Exception as e:
            logger.error(f"❌ 1inch client xato: {e}")
            return []
            
    @retry_manager(max_retries=3, delay=1.0)
    @rate_limiter(calls=100, period=60)
    async def get_swap_quote(self, chain: str, from_token: str, to_token: str, amount: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Swap quote olish"""
        try:
            chain_id = self.chains.get(chain, 1)
            url = f"{self.swap_url}/{chain_id}/quote"
            
            params = {
                "src": from_token,
                "dst": to_token,
                "amount": str(amount),
                "includeTokensInfo": "true",
                "includeProtocols": "true"
            }
            
            response = await safe_request(self.session, "GET", url, headers=await self._get_headers(), params=params)
            if response["status"] == 200:
                data = response["data"]
                logger.info(f"✅ 1inch quote: {data['srcToken']['symbol']} -> {data['dstToken']['symbol']}")
                return data
            return None
        except Exception as e:
            logger.error(f"❌ 1inch quote xato: {e}")
            return None
            
    async def analyze_order_flow(self, chain: str = "ethereum", tokens: List[str] = None) -> List[OrderFlowData]:
        """Order flow tahlili"""
        flows = []
        
        # Get active liquidity sources
        sources = await self.get_liquidity_sources(chain)
        
        # Analyze major token pairs
        if not tokens:
            tokens = ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                     "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                     "0xdAC17F958D2ee523a2206206994597C13D831ec7"]  # USDT
                     
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                quote = await self.get_swap_quote(chain, tokens[i], tokens[j], 10**18)
                if quote:
                    flow = OrderFlowData(
                        flow_type=OrderFlowType.DEX_SWAP,
                        token_in=quote["srcToken"]["symbol"],
                        token_out=quote["dstToken"]["symbol"],
                        amount_in=Decimal(quote["srcAmount"]) / 10**quote["srcToken"]["decimals"],
                        amount_out=Decimal(quote["dstAmount"]) / 10**quote["dstToken"]["decimals"],
                        usd_value=Decimal(quote.get("srcUSD", "0")),
                        from_address="",
                        to_address="",
                        tx_hash="",
                        timestamp=datetime.now(),
                        metadata={"protocols": quote.get("protocols", [])}
                    )
                    flows.append(flow)
                    
        return flows

class AlchemyClient:
    """Alchemy blockchain data API client"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.networks = {
            "ethereum": "eth-mainnet",
            "polygon": "polygon-mainnet",
            "arbitrum": "arb-mainnet",
            "optimism": "opt-mainnet"
        }
        self._api_key = None
        
    def _get_base_url(self, network: str = "ethereum") -> str:
        """Network URL olish"""
        if not self._api_key:
            api_config = config_manager.get_api_config(APIProvider.ALCHEMY)
            self._api_key = api_config.api_key if api_config else ""
        network_name = self.networks.get(network, "eth-mainnet")
        return f"https://{network_name}.g.alchemy.com/v2/{self._api_key}"
        
    @retry_manager(max_retries=3, delay=1.0)
    @rate_limiter(calls=300, period=60)
    async def get_token_transfers(self, network: str, address: str, block_range: int = 100) -> List[Dict[str, Any]]:
        """Token transferlarini olish"""
        try:
            url = self._get_base_url(network)
            
            # Get latest block
            latest_block_response = await safe_request(
                self.session, "POST", url,
                json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
            )
            
            if latest_block_response["status"] != 200:
                return []
                
            latest_block = int(latest_block_response["data"]["result"], 16)
            from_block = hex(latest_block - block_range)
            to_block = hex(latest_block)
            
            # Get transfers
            params = {
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "fromAddress": address,
                    "category": ["erc20", "external"],
                    "withMetadata": True,
                    "excludeZeroValue": True,
                    "maxCount": "0x3e8"  # 1000
                }],
                "id": 1
            }
            
            response = await safe_request(self.session, "POST", url, json=params)
            if response["status"] == 200 and "result" in response["data"]:
                transfers = response["data"]["result"]["transfers"]
                logger.info(f"✅ Alchemy transfers: {len(transfers)} ta")
                return transfers
            return []
            
        except Exception as e:
            logger.error(f"❌ Alchemy transfers xato: {e}")
            return []
            
    @retry_manager(max_retries=3, delay=1.0)
    @rate_limiter(calls=300, period=60)
    async def get_whale_transactions(self, network: str = "ethereum", min_value_usd: float = 1000000) -> List[OrderFlowData]:
        """Whale transactionlarni olish"""
        try:
            url = self._get_base_url(network)
            
            # Get pending transactions
            params = {
                "jsonrpc": "2.0",
                "method": "alchemy_pendingTransactions",
                "params": [{
                    "toAddress": ["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                                 "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"],  # WETH
                    "hashesOnly": False
                }],
                "id": 1
            }
            
            response = await safe_request(self.session, "POST", url, json=params)
            flows = []
            
            if response["status"] == 200 and "result" in response["data"]:
                for tx in response["data"]["result"]:
                    value = int(tx.get("value", "0x0"), 16) / 10**18
                    if value * 3000 >= min_value_usd:  # Assuming ETH price ~$3000
                        flow = OrderFlowData(
                            flow_type=OrderFlowType.WHALE_TRANSFER,
                            token_in="ETH",
                            token_out="ETH",
                            amount_in=Decimal(value),
                            amount_out=Decimal(value),
                            usd_value=Decimal(value * 3000),
                            from_address=tx.get("from", ""),
                            to_address=tx.get("to", ""),
                            tx_hash=tx.get("hash", ""),
                            timestamp=datetime.now(),
                            metadata={"gas": tx.get("gas", ""), "gasPrice": tx.get("gasPrice", "")}
                        )
                        flows.append(flow)
                        
            logger.info(f"✅ Alchemy whale txs: {len(flows)} ta")
            return flows
            
        except Exception as e:
            logger.error(f"❌ Alchemy whale xato: {e}")
            return []

class CCXTClient:
    """CCXT unified exchange API client"""
    
    def __init__(self):
        self.exchanges = {}
        self.default_exchange = "binance"
        self._initialized = False
        
    async def initialize(self, exchange_name: str = "binance") -> None:
        """Exchange clientni sozlash"""
        if exchange_name in self.exchanges:
            return
            
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'rateLimit': 50,  # milliseconds
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            # Load markets
            await exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            self._initialized = True
            logger.info(f"✅ CCXT {exchange_name} initialized")
            
        except Exception as e:
            logger.error(f"❌ CCXT init xato: {e}")
            
    async def get_exchange(self, name: str = None) -> ccxt.Exchange:
        """Exchange instance olish"""
        name = name or self.default_exchange
        if name not in self.exchanges:
            await self.initialize(name)
        return self.exchanges.get(name)
        
    @retry_manager(max_retries=3, delay=1.0)
    async def fetch_ticker(self, symbol: str, exchange: str = None) -> Optional[Dict[str, Any]]:
        """Ticker ma'lumotlarini olish"""
        try:
            ex = await self.get_exchange(exchange)
            ticker = await ex.fetch_ticker(symbol)
            logger.info(f"✅ CCXT ticker: {symbol} = ${ticker['last']}")
            return ticker
        except Exception as e:
            logger.error(f"❌ CCXT ticker xato: {e}")
            return None
            
    @retry_manager(max_retries=3, delay=1.0)
    async def fetch_order_book(self, symbol: str, limit: int = 20, exchange: str = None) -> Optional[Dict[str, Any]]:
        """Order book olish"""
        try:
            ex = await self.get_exchange(exchange)
            orderbook = await ex.fetch_order_book(symbol, limit)
            
            # Calculate imbalance
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:5]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:5]])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            orderbook['imbalance'] = imbalance
            orderbook['bid_volume'] = bid_volume
            orderbook['ask_volume'] = ask_volume
            
            logger.info(f"✅ CCXT orderbook: {symbol} imbalance={imbalance:.2%}")
            return orderbook
            
        except Exception as e:
            logger.error(f"❌ CCXT orderbook xato: {e}")
            return None
            
    @retry_manager(max_retries=3, delay=1.0)
    async def fetch_trades(self, symbol: str, limit: int = 100, exchange: str = None) -> List[Dict[str, Any]]:
        """So'nggi savdolarni olish"""
        try:
            ex = await self.get_exchange(exchange)
            trades = await ex.fetch_trades(symbol, limit=limit)
            
            # Analyze trade flow
            buy_volume = sum([t['amount'] for t in trades if t['side'] == 'buy'])
            sell_volume = sum([t['amount'] for t in trades if t['side'] == 'sell'])
            
            logger.info(f"✅ CCXT trades: {symbol} buy={buy_volume:.2f} sell={sell_volume:.2f}")
            return trades
            
        except Exception as e:
            logger.error(f"❌ CCXT trades xato: {e}")
            return []
            
    @retry_manager(max_retries=3, delay=1.0)
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100, exchange: str = None) -> List[List[Union[int, float]]]:
        """OHLCV ma'lumotlarini olish"""
        try:
            ex = await self.get_exchange(exchange)
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.info(f"✅ CCXT OHLCV: {symbol} {timeframe} {len(ohlcv)} candles")
            return ohlcv
        except Exception as e:
            logger.error(f"❌ CCXT OHLCV xato: {e}")
            return []
            
    async def close_all(self) -> None:
        """Barcha exchangelarni yopish"""
        for exchange in self.exchanges.values():
            await exchange.close()
        self.exchanges.clear()

class CoinGeckoClient:
    """CoinGecko market data API client"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://pro-api.coingecko.com/api/v3"
        self._api_key = None
        
    def _get_headers(self) -> Dict[str, str]:
        """Headers olish"""
        if not self._api_key:
            api_config = config_manager.get_api_config(APIProvider.COINGECKO)
            self._api_key = api_config.api_key if api_config else ""
        return {"x-cg-pro-api-key": self._api_key} if self._api_key else {}
        
    def _get_url(self, endpoint: str) -> str:
        """URL tanlash (free yoki pro)"""
        base = self.pro_url if self._api_key else self.base_url
        return f"{base}/{endpoint}"
        
    @retry_manager(max_retries=3, delay=2.0)
    @rate_limiter(calls=50, period=60)
    async def get_price(self, coin_ids: List[str], vs_currencies: str = "usd") -> Dict[str, Dict[str, float]]:
        """Coin narxlarini olish"""
        try:
            url = self._get_url("simple/price")
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": vs_currencies,
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            response = await safe_request(self.session, "GET", url, headers=self._get_headers(), params=params)
            if response["status"] == 200:
                logger.info(f"✅ CoinGecko prices: {len(response['data'])} coins")
                return response["data"]
            return {}
            
        except Exception as e:
            logger.error(f"❌ CoinGecko price xato: {e}")
            return {}
            
    @retry_manager(max_retries=3, delay=2.0)
    @rate_limiter(calls=50, period=60)
    async def get_market_chart(self, coin_id: str, days: int = 1, interval: str = "hourly") -> Dict[str, List[List[float]]]:
        """Market chart ma'lumotlarini olish"""
        try:
            url = self._get_url(f"coins/{coin_id}/market_chart")
            params = {"vs_currency": "usd", "days": days, "interval": interval}
            
            response = await safe_request(self.session, "GET", url, headers=self._get_headers(), params=params)
            if response["status"] == 200:
                data = response["data"]
                logger.info(f"✅ CoinGecko chart: {coin_id} {len(data.get('prices', []))} points")
                return data
            return {"prices": [], "market_caps": [], "total_volumes": []}
            
        except Exception as e:
            logger.error(f"❌ CoinGecko chart xato: {e}")
            return {"prices": [], "market_caps": [], "total_volumes": []}
            
    @retry_manager(max_retries=3, delay=2.0)
    @rate_limiter(calls=50, period=60)
    async def get_trending(self) -> Dict[str, Any]:
        """Trending coinlarni olish"""
        try:
            url = self._get_url("search/trending")
            response = await safe_request(self.session, "GET", url, headers=self._get_headers())
            
            if response["status"] == 200:
                data = response["data"]
                logger.info(f"✅ CoinGecko trending: {len(data.get('coins', []))} coins")
                return data
            return {"coins": []}
            
        except Exception as e:
            logger.error(f"❌ CoinGecko trending xato: {e}")
            return {"coins": []}
            
    @retry_manager(max_retries=3, delay=2.0)
    @rate_limiter(calls=50, period=60)
    async def get_global_data(self) -> Dict[str, Any]:
        """Global crypto market ma'lumotlari"""
        try:
            url = self._get_url("global")
            response = await safe_request(self.session, "GET", url, headers=self._get_headers())
            
            if response["status"] == 200:
                data = response["data"]["data"]
                logger.info(f"✅ CoinGecko global: MCap ${data['total_market_cap']['usd']:,.0f}")
                return data
            return {}
            
        except Exception as e:
            logger.error(f"❌ CoinGecko global xato: {e}")
            return {}

class TradingAPIsManager:
    """Barcha trading API'larni boshqaruvchi"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.oneinch: Optional[OneInchClient] = None
        self.alchemy: Optional[AlchemyClient] = None
        self.ccxt: Optional[CCXTClient] = None
        self.coingecko: Optional[CoinGeckoClient] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """API clientlarni ishga tushirish"""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession()
        self.oneinch = OneInchClient(self.session)
        self.alchemy = AlchemyClient(self.session)
        self.ccxt = CCXTClient()
        self.coingecko = CoinGeckoClient(self.session)
        
        # Start fallback monitoring
        await fallback_manager.start_monitoring()
        
        self._initialized = True
        logger.info("✅ Trading APIs initialized")
        
    async def close(self) -> None:
        """API clientlarni yopish"""
        if self.ccxt:
            await self.ccxt.close_all()
        if self.session:
            await self.session.close()
        await fallback_manager.stop_monitoring()
        self._initialized = False
        
    async def get_comprehensive_market_data(self, symbol: str) -> Dict[str, Any]:
        """To'liq bozor ma'lumotlarini olish"""
        if not self._initialized:
            await self.initialize()
            
        data = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "ticker": None,
            "orderbook": None,
            "trades": [],
            "chart": None,
            "order_flow": []
        }
        
        # Get exchange data
        data["ticker"] = await self.ccxt.fetch_ticker(symbol)
        data["orderbook"] = await self.ccxt.fetch_order_book(symbol)
        data["trades"] = await self.ccxt.fetch_trades(symbol, limit=50)
        
        # Get market data
        coin_id = symbol.split("/")[0].lower()
        prices = await self.coingecko.get_price([coin_id])
        if prices:
            data["market_data"] = prices.get(coin_id, {})
            
        # Get order flow
        flows = await self.oneinch.analyze_order_flow()
        data["order_flow"] = [
            {
                "type": f.flow_type.value,
                "tokens": f"{f.token_in}->{f.token_out}",
                "usd_value": float(f.usd_value),
                "timestamp": f.timestamp.isoformat()
            }
            for f in flows[:5]
        ]
        
        return data

# Global instance
trading_apis = TradingAPIsManager()
