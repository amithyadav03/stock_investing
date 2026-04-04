# 🚀 AI Autonomous Swing Trading Agent (NSE)

An advanced, production-ready autonomous trading system designed for the Indian equity market. It combines multi-modal AI reasoning (Vision + Mathematical Technicals) with a disciplined risk management engine and human-in-the-loop trade execution via Telegram.

---

## 🏗️ Core Architecture

The agent is built on a **cyclic graph architecture** using `LangGraph`, allowing it to perform deep, multi-step research before proposing a trade.

- **Technical Analyst Node**: Combines `pandas-ta` mathematical indicators with **GPT-4o Vision** analysis of candlestick charts.
- **Fundamental & Sentiment Node**: Scrapes `Screener.in` for Indian equity health and aggregates live **RSS feeds** (MoneyControl, ET, Mint) for macro and micro sentiment.
- **Risk Manager Node**: Enforces "Hard Python Guardrails" (no hallucinations allowed) and calculates ATR-governed Stop Losses and Targets.
- **Orchestrator**: Manages the state, parallelizes data fetching, and handles the hand-off to human approval.

---

## 🛡️ Risk Management & Safety

This system is built for **real money safety**:
- **ATR-Based SL**: Stop losses are dynamically placed at `2.0 * ATR` below entry to survive market noise.
- **Macro Filtering**: If the broad market sentiment is `BEARISH`, the system automatically halves trade size or aborts BUY signals.
- **Price Slippage Validation**: Before final execution, the system re-validates the live price against the proposed entry to prevent buying during a "gap up."
- **Human-in-the-Loop**: No trade is ever executed without a 1-tap `Approve` button click on your Telegram bot.

---

## ⚙️ Configuration & Customization

The project is fully decoupled, allowing you to tune the strategy without touching the code.

### 1. Strategy Settings (`strategy_config.yaml`)
Control your risk limits, symbols to scan, and Zerodha order types in one place.
- `symbols_to_scan`: List of NSE tickers.
- `max_risk_per_trade`: Capital exposure limit.
- `order_variety`: Toggle between `AMO` (After Market) and `Regular` (Live Market).

### 2. AI Prompt Management (`prompts/`)
Modify the "personality" and reasoning of the agents in individual `.txt` files:
- `technical_analyst.txt`: Tuning chart interpretation.
- `risk_manager.txt`: Defining the Chain-of-Thought for risk logic.

---

## 🚀 Getting Started

### 1. Environment Setup
Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY`: For the reasoning engine.
- `KITE_API_KEY` & `KITE_ACCESS_TOKEN`: From Zerodha Kite Connect.
- `TELEGRAM_BOT_TOKEN` & `CHAT_ID`: From BotFather.
- `NGROK_AUTH_TOKEN`: For local webhook tunneling.

### 2. Launch the System
```bash
# Start the FastAPI server + Ngrok Tunnel + Webhook Registration
python launch.py
```

### 3. Run a Market Scan
In a second terminal, trigger the AI to scan your configured stock list:
```bash
python auto_scanner.py
```

---

## 📊 Backtesting
Simulate the AI's performance over a historical window (e.g., 6 months) to see how it would have performed during different market cycles.
```bash
python backtest.py
```
*Results are saved to `db/backtest_results.csv` and charts to `db/backtest_charts/`.*

---

## 📂 Project Structure

```text
├── agents/             # LangGraph Nodes & State Definition
├── core/               # Auth, Config, and Telegram Bot Logic
├── tools/              # Market Data, Fundamentals & News Scrapers
├── prompts/            # Externalized LLM Instruction Sets
├── tests/              # Connection and Unit Testing
├── db/                 # SQLite Logs, Vector Memory & Local Assets
├── strategy_config.yaml # Strategy & Risk Parameters
├── launch.py           # Main Entry Point (Server + Webhook)
├── auto_scanner.py     # Batch Symbol Analysis Trigger
└── backtest.py         # Historical Performance Engine
```

---

## ⚖️ Disclaimer
*This project is for educational and research purposes. Trading in the stock market involves significant risk. Never trade with money you cannot afford to lose.*