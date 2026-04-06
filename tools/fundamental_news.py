import requests
import feedparser
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Dict, Any, List
import yfinance as yf

from core.config import settings

class MacroContext(BaseModel):
    sentiment_enum: str
    risk_multiplier: float
    summary: str

class FundamentalNewsTool:
    def __init__(self):
        # Top tier Indian financial RSS feeds from strategy config
        self.rss_feeds = settings.strategy.get("news", {}).get("rss_feeds", [])

    def get_comparative_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Scrapes Screener.in for precise Indian metrics. Fallback to yfinance if blocked.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Screener puts key metrics in a 'company-ratios' list
                ratios = {}
                ratios_ul = soup.find('ul', id='top-ratios')
                if ratios_ul:
                    for li in ratios_ul.find_all('li'):
                        name = li.find('span', class_='name').text.strip()
                        value_span = li.find('span', class_='number')
                        if value_span:
                            ratios[name] = value_span.text.strip()
                
                return {
                    "source": "screener.in",
                    "symbol": symbol,
                    "pe_ratio": ratios.get("Stock P/E", "N/A"),
                    "roe": ratios.get("ROE", "N/A"),
                    "roce": ratios.get("ROCE", "N/A"),
                    "debt_to_equity": ratios.get("Debt to equity", "N/A"),
                    "promoter_holding": ratios.get("Promoter holding", "N/A")
                }
        except Exception as e:
            print(f"[Fundamentals] Screener failed for {symbol}: {e}. Falling back to yfinance.")

        # Fallback to yfinance
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            return {
                "source": "yfinance",
                "symbol": symbol,
                "pe_ratio": info.get("trailingPE", "N/A"),
                "roe": info.get("returnOnEquity", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                # Yfinance lacks promoter pledge natively for India
            }
        except:
            return {"error": "Could not fetch fundamentals from any source."}

    def fetch_live_news_snippets(self, target_keyword: str = None) -> List[str]:
        """
        Aggregates live RSS feeds. If target_keyword is provided, filters for that stock.
        Otherwise returns top macro headlines.
        """
        headlines = []
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]: # Top 10 per feed
                    # Basic filtering for micro
                    if target_keyword:
                        if target_keyword.lower() in entry.title.lower() or target_keyword.lower() in entry.description.lower():
                            headlines.append(f"{entry.title}: {entry.description[:100]}...")
                    else:
                        headlines.append(f"{entry.title}: {entry.description[:100]}...")
            except Exception as e:
                print(f"[RSS Error] Failed fetching {url}: {e}")
                
        # Deduplicate and return top 15
        return list(set(headlines))[:15]

    def get_micro_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves live news for a specific stock snippet.
        """
        news = self.fetch_live_news_snippets(target_keyword=symbol)
        if not news:
            return {"score": 0, "summary": f"No immediate breaking news found for {symbol} on Indian RSS feeds."}
        
        # Here we would use the map-reduce GPT-4o-mini pass. For architecture proofing:
        joined_news = "\n".join(news)
        return {
            "score": 0, # To be dynamically populated by LLM
            "summary": f"Found {len(news)} live articles.\nSample: {news[0]}"
        }
        
    def get_macro_context(self) -> Dict[str, Any]:
        """
        Fetches the broader market headlines and global index performance (Nifty 50, S&P 500)
        to establish the Macro Environment evidence.
        """
        macro_news = self.fetch_live_news_snippets(target_keyword=None)
        
        # Global Index Trends (30 Days)
        index_data = {}
        for name, ticker in [("Nifty 50", "^NSEI"), ("S&P 500", "^GSPC")]:
            try:
                df = yf.Ticker(ticker).history(period="30d")
                perf = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                index_data[name] = f"{'+' if perf > 0 else ''}{round(perf*100, 2)}%"
            except:
                index_data[name] = "Data Unavailable"

        return {
            "headlines": macro_news,
            "index_performance": index_data
        }

fundamental_news_tool = FundamentalNewsTool()
