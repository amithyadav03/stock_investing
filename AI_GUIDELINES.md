# 🤖 AI Coding Assistant Guidelines & Guardrails

Welcome to the AI Swing Trading Agent codebase. As an AI Coding Assistant working on this repository, you **MUST** strictly adhere to the following implementation guidelines and guardrails. This project deals with real money execution; therefore, safety, transparency, and clean architecture are paramount.

Please read and acknowledge these rules before proposing or writing any code.

---

## 1. 📂 Repository Structure & Organization
Maintain a strictly decoupled and organized repository structure to ensure maintainability.

*   **Logic & Core**: Application core logic, agents, and tool abstractions go in `core/`, `agents/`, and `tools/`.
*   **Scripts**: Top-level execution files (e.g., `backtest.py`, `auto_scanner.py`, `launch.py`, utility scripts) must be placed cleanly into a `scripts/` directory or appropriately organized so the root directory remains uncluttered.
*   **Data & Assets**: All database files, vector memories, and generated PNG charts must reside in `db/` and must never be committed to version control.

## 2. 🧠 Prompt Management
**Rule: Absolutely no hardcoded AI instruction strings inside Python code.**

*   All system and user prompts must be externalized into the `prompts/` directory.
*   Each prompt must have its own dedicated `.txt` file (e.g., `technical_analyst.txt`).
*   Scripts must load these files dynamically using a file-reading helper function (with explicit `encoding="utf-8"` to prevent Windows charmap errors).

## 3. 🚫 Zero Fallback & Anti-Hallucination Guardrails
**Rule: Never implement "mock" or "fallback" trades if an operation fails.**

*   If a requirement is missing, an API rate limits you, an LLM parses incorrectly, or a generic exception occurs, **fail explicitly**.
*   The system must enter an `ERROR` state and log exactly what happened.
*   Errors must be passed up the chain and explicitly **notified to the user** (e.g., via the Telegram Webhook).
*   Never use `except: pass` silently around critical network, scraping, or generation code.

## 4. 🧱 Structured Input/Output validations
**Rule: LLM Outputs must be deterministic.**

*   Always use `Pydantic` models alongside LangChain's `.with_structured_output()` when receiving execution commands, intents, or analyses from an LLM.
*   Enforce hard Python rule checks (e.g., "Stop Loss must be lower than Entry") on the returned Pydantic object *before* approving the state. Never trust the LLM with raw execution execution safely.

## 5. 🔐 Configuration & Secrets Management
**Rule: No magic numbers, no hardcoded API endpoints, no embedded keys.**

*   **Secrets**: API keys, Database URLs, and secure tokens exclusively belong in local `.env` files. Maintain a `.env.example` that mirrors the structure with dummy values.
*   **Strategy**: All trading parameters (symbols to scan, risk percentages, maximum trade quantities, Kite order types, RSS feed URLs) must be stored in the `strategy_config.yaml`.
*   Scripts must dynamically parse these configurations.

## 6. 🛡️ Best Practices & Additional Guardrails
When building or scaling this system, apply the following engineering principles:

*   **Type Hinting**: All Python functions must include strict type hints for parameters and return types (e.g., `def fetch_data(symbol: str) -> dict:`).
*   **Exponential Backoffs**: For network requests (Kite Connect, web scrapers, LLM API calls), avoid arbitrary `time.sleep()`. Implement resilient retries using libraries like `tenacity` to handle transient network limits gracefully.
*   **Validator-First Execution**: Whenever touching `kite.place_order()`, ensure `pre_execution_validation` (slippage/drift checks) runs milliseconds before the network request.
*   **Asynchronous UX**: Long-running LLM generation loops should not block the main FastAPI thread. Utilize `BackgroundTasks` so the Orchestrator remains responsive for incoming webhook callbacks.

---
*By reading this document, the Assistant confirms its understanding of the critical safety requirements necessary to evolve this real-money algorithmic trading system.*
