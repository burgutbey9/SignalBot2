# 🤖 SignalBot - Professional Crypto Trading Bot

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/telegram-bot-blue.svg)](https://telegram.org/)

SignalBot - bu professional darajadagi crypto trading bot bo'lib, DEX Order Flow, AI Sentiment Analysis, ICT (Inner Circle Trader) va SMT (Smart Money Theory) metodlaridan foydalanib avtomatik trading signallari beradi. Bot O'zbekcha tilida ishlaydi va Telegram orqali signallarni yuboradi.

## 🌟 Asosiy Xususiyatlari

- **📊 Multi-Strategy Analysis**
  - ICT (Inner Circle Trader) metodologiyasi
  - SMT (Smart Money Theory) tahlili
  - DEX Order Flow monitoring
  - AI-powered sentiment analysis

- **🔄 3-Level Fallback System**
  - Automatic API failover
  - Circuit breaker pattern
  - Health monitoring

- **💬 O'zbekcha Telegram Interface**
  - Professional signal formatting
  - Interactive control panel
  - Real-time notifications

- **🛡️ Risk Management**
  - Dynamic position sizing
  - Stop loss tracking
  - Daily loss limits
  - Multi-timeframe analysis

- **🚀 High Performance**
  - Async/await architecture
  - Rate limiting
  - Caching system
  - Performance monitoring

## 📋 Talablar

- Python 3.9+
- SQLite/PostgreSQL
- Telegram Bot Token
- API keys (1inch, Alchemy, HuggingFace, etc.)

## 🔧 O'rnatish

### 1. Repository'ni clone qiling

```bash
git clone https://github.com/yourusername/SignalBot2.git
cd SignalBot2
```

### 2. Virtual environment yarating

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# yoki
venv\Scripts\activate  # Windows
```

### 3. Dependencies o'rnating

```bash
pip install -r requirements.txt
```

### 4. Environment variables sozlang

```bash
cp .env.example .env
# .env faylini tahrirlang va API key'larni kiriting
```

### 5. Database migratsiyasi

```bash
python -m alembic upgrade head
```

## 🚀 Ishga tushirish

### Signal-only rejimda

```bash
python src/main.py --mode signal
```

### Paper trading rejimda

```bash
python src/main.py --mode paper
```

### Live trading rejimda (ehtiyot bo'ling!)

```bash
python src/main.py --mode live
```

## 📁 Loyiha Strukturasi

```
SignalBot2/
├── config/                 # Konfiguratsiya fayllari
│   ├── config.py          # Asosiy config manager
│   ├── settings.json      # JSON sozlamalar
│   └── fallback_config.py # Fallback tizimi
├── src/
│   ├── api/               # API clientlar
│   │   ├── trading_apis.py
│   │   ├── ai_clients.py
│   │   ├── news_social.py
│   │   └── telegram_client.py
│   ├── core/              # Asosiy tizim
│   │   ├── bot_manager.py
│   │   ├── risk_manager.py
│   │   └── timezone_handler.py
│   ├── analysis/          # Tahlil modullari
│   │   ├── ict_analyzer.py
│   │   ├── smt_analyzer.py
│   │   ├── order_flow.py
│   │   └── sentiment.py
│   ├── trading/           # Trading engine
│   │   ├── signal_generator.py
│   │   ├── trade_analyzer.py
│   │   └── execution_engine.py
│   ├── telegram/          # Telegram interface
│   │   ├── bot_interface.py
│   │   └── message_handler.py
│   ├── utils/             # Yordamchi funksiyalar
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   └── database.py
│   └── main.py           # Asosiy fayl
├── tests/                # Test fayllari
├── data/                 # Ma'lumotlar
├── logs/                 # Log fayllari
├── requirements.txt
├── .env.example
└── README.md
```

## 🔐 Xavfsizlik

- API key'larni **hech qachon** public repository'ga yuklamang
- `.env` faylini `.gitignore`ga qo'shing
- Production'da API key encryption'ni yoqing
- Regular security audit o'tkazing

## 📊 Signal Formati

```
📊 SIGNAL KELDI
════════════════
📈 Savdo: BUY BTCUSDT
💰 Narx: $45,230
📊 Lot: 0.1 BTC
🛡️ Stop Loss: $44,500 (1.6%)
🎯 Take Profit: $46,500 (2.8%)
⚡ Ishonch: 85%
🔥 Risk: 0.5%
════════════════
📝 Sabab: Whale accumulation + SSL break + FVG mitigation + bullish sentiment
🐋 On-chain: 15,000 BTC to cold storage
⏰ Vaqt: 14:30 (UZB)
[🟢 AVTO SAVDO] [🔴 BEKOR QILISH]
```

## 🧪 Testing

### Unit testlarni ishga tushirish

```bash
pytest tests/
```

### Coverage report

```bash
pytest --cov=src tests/
```

### Specific test

```bash
pytest tests/test_analysis.py::TestICTAnalyzer
```

## 📈 Monitoring

Bot quyidagi metrikalarni kuzatadi:

- Signal generation rate
- API health status
- Trade performance
- System resources
- Error rates

Prometheus metrics endpoint: `http://localhost:9090/metrics`

## 🤝 Hissa qo'shish

1. Fork qiling
2. Feature branch yarating (`git checkout -b feature/amazing-feature`)
3. O'zgarishlarni commit qiling (`git commit -m 'Add amazing feature'`)
4. Branch'ga push qiling (`git push origin feature/amazing-feature`)
5. Pull Request oching

## 📝 Litsenziya

Bu loyiha MIT litsenziyasi ostida tarqatiladi. Batafsil ma'lumot uchun [LICENSE](LICENSE) faylini ko'ring.

## ⚠️ Ogohlantirish

**MUHIM**: Bu bot faqat ta'lim va tadqiqot maqsadlarida yaratilgan. Real pul bilan trading qilishdan oldin:

- Kichik summalar bilan boshlang
- Paper trading rejimda sinab ko'ring
- Risk management qoidalariga qat'iy amal qiling
- Faqat yo'qotishga tayyor bo'lgan pulni ishlating

Trading yuqori risk darajasiga ega va siz barcha pulingizni yo'qotishingiz mumkin.

## 🆘 Yordam

Savollar yoki muammolar bo'lsa:

1. [Issues](https://github.com/yourusername/SignalBot2/issues) bo'limini tekshiring
2. Yangi issue oching
3. Telegram: @yourusername

## 🙏 Minnatdorchilik

- ICT metodologiyasi uchun Michael J. Huddleston
- O'zbek crypto community
- Barcha contributor'lar

---

**Eslatma**: Har doim o'z tadqiqotingizni qiling (DYOR) va professional moliyaviy maslahat oling.
