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
        """
        Fetches fundamentals with Screener.in as primary, yfinance as fallback.
        Results cached 24h in TTL cache. Returns standardized numeric fields where possible.
        """
        from core.cache import cache, TTL_FUNDAMENTALS

        cache_key = f"fundamentals_{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        result = self._fetch_screener(symbol)
        if result.get("error"):
            result = self._fetch_yfinance(symbol)

        # Compute quality score
        result["quality_score"] = self._compute_quality_score(result)

        cache.set(cache_key, result, TTL_FUNDAMENTALS)
        return result

    def _safe_float(self, val, default=None):
        """Convert string/number to float, return default on failure."""
        if val is None or val == "N/A" or val == "":
            return default
        try:
            return float(str(val).replace(',', '').replace('%', '').strip())
        except (ValueError, TypeError):
            return default

    def _fetch_screener(self, symbol: str) -> Dict[str, Any]:
        """Scrapes Screener.in with retries and structured extraction."""
        import time
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.screener.in',
        }

        urls_to_try = [
            f"https://www.screener.in/company/{symbol}/consolidated/",
            f"https://www.screener.in/company/{symbol}/",
        ]

        for url in urls_to_try:
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        return self._parse_screener_html(resp.text, symbol)
                    elif resp.status_code == 404:
                        break  # Symbol not found, try other URL
                    elif resp.status_code == 429:
                        time.sleep(2 ** attempt)  # Rate limited
                        continue
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        time.sleep(1)
                    continue
                except Exception as e:
                    print(f"[Fundamentals] Screener fetch failed for {symbol}: {e}")
                    break

        return {"error": f"Screener unavailable for {symbol}", "symbol": symbol, "source": "none"}

    def _parse_screener_html(self, html: str, symbol: str) -> Dict[str, Any]:
        """Extract all relevant ratios from Screener.in HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        ratios: dict = {}

        # Extract top ratios section
        ratios_ul = soup.find('ul', id='top-ratios')
        if ratios_ul:
            for li in ratios_ul.find_all('li'):
                name_el = li.find('span', class_='name')
                val_el = li.find('span', class_='number')
                if name_el and val_el:
                    ratios[name_el.text.strip()] = val_el.text.strip()

        # Extract profit growth from quarterly table
        profit_growth_qoq = "N/A"
        revenue_growth_qoq = "N/A"
        profit_growth_annual = "N/A"
        try:
            for table_id in ['quarters', 'profit-loss']:
                profit_table = soup.find('table', id=table_id)
                if not profit_table:
                    continue
                rows = profit_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    row_text = cells[0].text.strip().lower()
                    vals = []
                    for c in cells[1:]:
                        t = c.text.strip().replace(',', '')
                        if t and t != '':
                            try:
                                vals.append(float(t))
                            except ValueError:
                                pass
                    if len(vals) >= 2:
                        prev, curr = vals[-2], vals[-1]
                        if prev != 0:
                            growth = round((curr - prev) / abs(prev) * 100, 1)
                            if 'net profit' in row_text or 'profit' in row_text:
                                if table_id == 'quarters':
                                    profit_growth_qoq = f"{growth}%"
                                else:
                                    profit_growth_annual = f"{growth}%"
                            elif 'revenue' in row_text or 'sales' in row_text:
                                revenue_growth_qoq = f"{growth}%"
        except Exception:
            pass

        # Extract EPS if available
        eps_growth = "N/A"
        try:
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if cells and 'eps' in cells[0].text.lower():
                    vals = [c.text.strip().replace(',', '') for c in cells[1:] if c.text.strip()]
                    if len(vals) >= 2:
                        try:
                            prev, curr = float(vals[-2]), float(vals[-1])
                            if prev != 0:
                                eps_growth = f"{round((curr - prev) / abs(prev) * 100, 1)}%"
                        except ValueError:
                            pass
                    break
        except Exception:
            pass

        # Promoter pledge (look in shareholding section)
        promoter_pledge = "N/A"
        try:
            pledge_text = soup.find(string=lambda t: t and 'pledge' in t.lower())
            if pledge_text:
                parent = pledge_text.find_parent('td')
                if parent and parent.find_next_sibling('td'):
                    promoter_pledge = parent.find_next_sibling('td').text.strip()
        except Exception:
            pass

        return {
            "source": "screener.in",
            "symbol": symbol,
            "pe_ratio": self._safe_float(ratios.get("Stock P/E") or ratios.get("P/E")),
            "roe": self._safe_float(ratios.get("ROE")),
            "roce": self._safe_float(ratios.get("ROCE")),
            "debt_to_equity": self._safe_float(ratios.get("Debt to equity") or ratios.get("Debt / Equity")),
            "promoter_holding": self._safe_float(ratios.get("Promoter holding")),
            "promoter_pledge": promoter_pledge,
            "market_cap": ratios.get("Market Cap", "N/A"),
            "dividend_yield": self._safe_float(ratios.get("Dividend Yield")),
            "book_value": self._safe_float(ratios.get("Book Value")),
            "current_ratio": self._safe_float(ratios.get("Current ratio") or ratios.get("Current Ratio")),
            "profit_growth_qoq": profit_growth_qoq,
            "profit_growth_annual": profit_growth_annual,
            "revenue_growth": revenue_growth_qoq,
            "eps_growth": eps_growth,
        }

    def _fetch_yfinance(self, symbol: str) -> Dict[str, Any]:
        """yfinance fallback — returns standardized dict with numeric fields."""
        try:
            info = yf.Ticker(f"{symbol}.NS").info
            roe_raw = info.get("returnOnEquity")
            roce_raw = info.get("returnOnAssets")  # yfinance doesn't have ROCE; ROA as proxy

            return {
                "source": "yfinance",
                "symbol": symbol,
                "pe_ratio": self._safe_float(info.get("trailingPE") or info.get("forwardPE")),
                "roe": round(float(roe_raw) * 100, 2) if roe_raw else None,
                "roce": round(float(roce_raw) * 100, 2) if roce_raw else None,
                "debt_to_equity": self._safe_float(info.get("debtToEquity")),
                "promoter_holding": None,
                "promoter_pledge": "N/A",
                "market_cap": info.get("marketCap"),
                "dividend_yield": self._safe_float(info.get("dividendYield")),
                "book_value": self._safe_float(info.get("bookValue")),
                "current_ratio": self._safe_float(info.get("currentRatio")),
                "profit_growth_qoq": "N/A",
                "profit_growth_annual": "N/A",
                "revenue_growth": self._safe_float(info.get("revenueGrowth")),
                "eps_growth": self._safe_float(info.get("earningsGrowth")),
            }
        except Exception as e:
            return {"error": f"yfinance also failed: {e}", "symbol": symbol, "source": "none"}

    def _compute_quality_score(self, data: dict) -> int:
        """
        Computes a 0-100 quality score based on fundamentals.
        Empirically validated factors for Indian equities:
        - ROCE > 15% (strong moat)
        - ROE > 15% (efficient equity use)
        - Debt/Equity < 0.5 (low leverage)
        - Promoter holding > 50% (skin in the game)
        - Low promoter pledge (governance risk)
        - Positive revenue and profit growth
        """
        score = 0

        roe = self._safe_float(data.get("roe"), 0)
        roce = self._safe_float(data.get("roce"), 0)
        de = self._safe_float(data.get("debt_to_equity"), 99)
        promoter = self._safe_float(data.get("promoter_holding"), 0)
        pledge = data.get("promoter_pledge", "N/A")
        rev_growth = data.get("revenue_growth", "N/A")
        profit_growth = data.get("profit_growth_qoq", "N/A")
        cr = self._safe_float(data.get("current_ratio"), 1)

        # ROCE scoring (0-25 pts) - most important for Indian equities
        if roce and roce >= 20: score += 25
        elif roce and roce >= 15: score += 20
        elif roce and roce >= 10: score += 12
        elif roce and roce >= 5: score += 5

        # ROE scoring (0-20 pts)
        if roe and roe >= 20: score += 20
        elif roe and roe >= 15: score += 15
        elif roe and roe >= 10: score += 8
        elif roe and roe >= 5: score += 3

        # Debt/Equity (0-20 pts)
        if de is not None:
            if de <= 0.1: score += 20
            elif de <= 0.3: score += 16
            elif de <= 0.5: score += 12
            elif de <= 1.0: score += 6
            elif de > 2.0: score -= 5  # Heavy debt penalty

        # Promoter holding (0-15 pts)
        if promoter and promoter >= 60: score += 15
        elif promoter and promoter >= 50: score += 12
        elif promoter and promoter >= 40: score += 7
        elif promoter and promoter < 25: score -= 5  # Very low promoter = concern

        # Promoter pledge penalty
        if pledge != "N/A" and pledge:
            pledge_val = self._safe_float(pledge, 0)
            if pledge_val and pledge_val > 50: score -= 15
            elif pledge_val and pledge_val > 30: score -= 8
            elif pledge_val and pledge_val > 10: score -= 3

        # Revenue growth (0-10 pts)
        rev_f = self._safe_float(str(rev_growth).replace('%', ''), 0)
        if rev_f and rev_f >= 20: score += 10
        elif rev_f and rev_f >= 10: score += 6
        elif rev_f and rev_f < 0: score -= 5

        # Current ratio liquidity (0-5 pts)
        if cr and cr >= 2.0: score += 5
        elif cr and cr >= 1.5: score += 3
        elif cr and cr < 1.0: score -= 3

        return max(0, min(100, score))

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
