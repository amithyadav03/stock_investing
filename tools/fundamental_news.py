import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Dict, Any, List
import yfinance as yf

try:
    import feedparser as _feedparser
    _HAS_FEEDPARSER = True
except Exception:
    _feedparser = None
    _HAS_FEEDPARSER = False

from core.config import settings


class MacroContext(BaseModel):
    sentiment_enum: str   # "BULLISH" | "NEUTRAL" | "BEARISH"
    risk_multiplier: float
    summary: str


class FundamentalNewsTool:
    def __init__(self):
        self.rss_feeds = settings.strategy.get("news", {}).get("rss_feeds", [])

    def get_comparative_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Scrapes Screener.in for key Indian metrics. Falls back to yfinance."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                ratios: dict = {}
                ratios_ul = soup.find('ul', id='top-ratios')
                if ratios_ul:
                    for li in ratios_ul.find_all('li'):
                        name_el = li.find('span', class_='name')
                        val_el = li.find('span', class_='number')
                        if name_el and val_el:
                            ratios[name_el.text.strip()] = val_el.text.strip()

                # Revenue / profit growth (quarterly)
                growth_pct = "N/A"
                try:
                    profit_table = soup.find('table', id='quarters')
                    if profit_table:
                        rows = profit_table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if cells and 'Net Profit' in cells[0].text:
                                vals = [c.text.strip().replace(',', '') for c in cells[1:] if c.text.strip()]
                                if len(vals) >= 2:
                                    try:
                                        prev, curr = float(vals[-2]), float(vals[-1])
                                        if prev != 0:
                                            growth_pct = f"{round((curr - prev) / abs(prev) * 100, 1)}%"
                                    except ValueError:
                                        pass
                                break
                except Exception:
                    pass

                return {
                    "source": "screener.in",
                    "symbol": symbol,
                    "pe_ratio": ratios.get("Stock P/E", "N/A"),
                    "roe": ratios.get("ROE", "N/A"),
                    "roce": ratios.get("ROCE", "N/A"),
                    "debt_to_equity": ratios.get("Debt to equity", "N/A"),
                    "promoter_holding": ratios.get("Promoter holding", "N/A"),
                    "market_cap": ratios.get("Market Cap", "N/A"),
                    "dividend_yield": ratios.get("Dividend Yield", "N/A"),
                    "profit_growth_qoq": growth_pct,
                }
        except Exception as e:
            print(f"[Fundamentals] Screener failed for {symbol}: {e}. Using yfinance.")

        try:
            info = yf.Ticker(f"{symbol}.NS").info
            return {
                "source": "yfinance",
                "symbol": symbol,
                "pe_ratio": info.get("trailingPE", "N/A"),
                "roe": info.get("returnOnEquity", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "earnings_growth": info.get("earningsGrowth", "N/A"),
            }
        except Exception:
            return {"error": "Could not fetch fundamentals.", "symbol": symbol}

    def fetch_live_news_snippets(self, target_keyword: str = None) -> List[str]:
        """Aggregates RSS headlines. Filters by keyword when provided."""
        if not _HAS_FEEDPARSER:
            return []
        headlines: list[str] = []
        for url in self.rss_feeds:
            try:
                feed = _feedparser.parse(url)
                for entry in feed.entries[:10]:
                    title = getattr(entry, 'title', '')
                    desc = getattr(entry, 'description', getattr(entry, 'summary', ''))
                    if target_keyword:
                        if target_keyword.lower() in title.lower() or target_keyword.lower() in desc.lower():
                            headlines.append(f"{title}: {desc[:120]}...")
                    else:
                        headlines.append(f"{title}: {desc[:120]}...")
            except Exception as e:
                print(f"[RSS] Failed {url}: {e}")
        return list(dict.fromkeys(headlines))[:15]  # dedupe, top 15

    def get_micro_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """
        Scores news sentiment for a specific stock using Claude.
        Returns score (-1.0 to +1.0), label, and key headlines.
        """
        news = self.fetch_live_news_snippets(target_keyword=symbol)
        if not news:
            return {"score": 0.0, "label": "NEUTRAL", "summary": f"No recent news found for {symbol}."}

        # Use Claude for sentiment scoring
        try:
            from core.claude_client import get_client, call_structured
            client = get_client()
            if client:
                result = call_structured(
                    client=client,
                    system_prompt=(
                        "You are a financial news sentiment analyst specializing in Indian equities. "
                        "Given news headlines for a stock, output a precise sentiment score and label."
                    ),
                    user_text=(
                        f"Analyze the sentiment of these news headlines for {symbol}:\n\n"
                        + "\n".join(f"- {h}" for h in news)
                    ),
                    tool_name="submit_sentiment",
                    tool_description="Submit the sentiment analysis result",
                    tool_schema={
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "description": "Sentiment score from -1.0 (very negative) to +1.0 (very positive)"},
                            "label": {"type": "string", "enum": ["VERY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "VERY_BEARISH"]},
                            "key_themes": {"type": "string", "description": "2-sentence summary of the key news themes"},
                        },
                        "required": ["score", "label", "key_themes"],
                    },
                    cache_system=True,
                )
                if result:
                    return {
                        "score": result.get("score", 0.0),
                        "label": result.get("label", "NEUTRAL"),
                        "summary": result.get("key_themes", ""),
                        "headlines_count": len(news),
                    }
        except Exception as e:
            print(f"[Sentiment] Claude scoring failed for {symbol}: {e}")

        # Fallback: rule-based heuristic
        pos = sum(1 for h in news if any(w in h.lower() for w in ["surge", "rally", "growth", "profit", "record", "beat", "upgrade"]))
        neg = sum(1 for h in news if any(w in h.lower() for w in ["fall", "fraud", "loss", "default", "penalty", "decline", "downgrade"]))
        score = round((pos - neg) / len(news), 2)
        label = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
        return {"score": score, "label": label, "summary": f"{len(news)} articles. Pos:{pos} Neg:{neg}"}

    def get_macro_context(self) -> Dict[str, Any]:
        """Global index performance + macro headlines for regime classification."""
        macro_news = self.fetch_live_news_snippets(target_keyword=None)

        index_data: dict = {}
        for name, ticker in [
            ("Nifty 50", "^NSEI"), ("Nifty Bank", "^NSEBANK"),
            ("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"), ("Hang Seng", "^HSI"),
        ]:
            try:
                df = yf.Ticker(ticker).history(period="30d")
                if not df.empty and len(df) >= 2:
                    perf = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                    index_data[name] = f"{'+' if perf > 0 else ''}{round(perf * 100, 2)}%"
            except Exception:
                index_data[name] = "N/A"

        return {"headlines": macro_news, "index_performance": index_data}

    def get_sector_performance(self) -> Dict[str, Any]:
        """Returns 5-day performance of key NSE sector indices."""
        sectors = {
            "Bank": "^NSEBANK", "IT": "^CNXIT", "Pharma": "^CNXPHARMA",
            "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG", "Energy": "^CNXENERGY",
            "Metal": "^CNXMETAL", "Realty": "^CNXREALTY",
        }
        perf: dict = {}
        for name, ticker in sectors.items():
            try:
                df = yf.Ticker(ticker).history(period="5d")
                if not df.empty and len(df) >= 2:
                    p = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                    perf[name] = f"{'+' if p > 0 else ''}{round(p * 100, 2)}%"
            except Exception:
                perf[name] = "N/A"
        return perf


fundamental_news_tool = FundamentalNewsTool()
