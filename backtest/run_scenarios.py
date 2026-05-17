"""
Comprehensive backtest scenarios for swing trading validation.

Run from project root:
    python -m backtest.run_scenarios

Covers 5 distinct market regimes on NSE:
  1. Pre-COVID bull + correction  (2018–2019)
  2. COVID crash + V-recovery     (2020)
  3. Post-COVID bull run          (2021)
  4. FII selloff + rate hike bear (2022)
  5. Recovery + recent bull       (2023–2024)
  6. Full period (2019–2024)      — master validation

Each scenario runs both swing and positional strategies.
Results saved to backtest/results/.
"""

import json
import sys
import os

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import run_backtest, run_monte_carlo

# ── Symbol universe — Nifty 50 liquid names ────────────────────────────────────
# These are the most liquid NSE stocks with 7+ years of clean data.
# Good liquidity means fills are realistic and results are trustworthy.
SYMBOLS = [
    # Banking (most traded, highest liquidity)
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
    # IT (high momentum historically)
    "INFY", "TCS", "WIPRO", "HCLTECH",
    # Pharma (defensive, works well in bear markets)
    "SUNPHARMA", "DRREDDY", "CIPLA",
    # Auto (cyclical, good swing trading)
    "TATAMOTORS", "MARUTI",
    # Energy / Conglomerate
    "RELIANCE",
    # FMCG (defensive)
    "HINDUNILVR", "ITC",
    # Metals (high beta, good for momentum)
    "TATASTEEL", "HINDALCO",
    # Infra
    "LT",
]

# ── Market regime scenarios ─────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "1_pre_covid_bull_correction",
        "label": "Pre-COVID Bull + Correction (2018–2019)",
        "start": "2018-01-01",
        "end":   "2019-12-31",
        "note":  "Nifty fell ~15% in late 2018, recovered in 2019. Tests: drawdown control + recovery capture.",
    },
    {
        "name": "2_covid_crash_recovery",
        "label": "COVID Crash + V-Recovery (2020)",
        "start": "2020-01-01",
        "end":   "2020-12-31",
        "note":  "Nifty -38% in 8 weeks (Feb–Mar 2020), then +80% recovery by year end. "
                 "Ultimate stress test: does the circuit breaker protect capital in the crash? "
                 "Does the system get back in during the recovery?",
    },
    {
        "name": "3_post_covid_bull",
        "label": "Post-COVID Bull Run (2021)",
        "start": "2021-01-01",
        "end":   "2021-12-31",
        "note":  "Nifty +24% in 2021, strong trending. Tests: momentum capture in ideal conditions.",
    },
    {
        "name": "4_fii_selloff_rate_hike_bear",
        "label": "FII Selloff + Rate Hike Bear (2022)",
        "start": "2022-01-01",
        "end":   "2022-12-31",
        "note":  "Nifty volatile, mid-year correction -17% from peak. FII sold ₹1.7L cr. "
                 "Tests: how system behaves in choppy, news-driven market.",
    },
    {
        "name": "5_recovery_and_recent_bull",
        "label": "Recovery + Recent Bull (2023–2024)",
        "start": "2023-01-01",
        "end":   "2024-12-31",
        "note":  "Nifty hit all-time highs. Tests: recent performance, most relevant to forward expectations.",
    },
    {
        "name": "6_full_period_master",
        "label": "Full Period Master Validation (2019–2024)",
        "start": "2019-01-01",
        "end":   "2024-12-31",
        "note":  "6-year walk-forward across all regimes. This is the primary validation.",
    },
]


def run_all_scenarios(strategy_type: str = "swing", capital: float = 500_000):
    summary_rows = []

    for s in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {s['label']}")
        print(f"Note: {s['note']}")
        print(f"{'='*70}")

        try:
            results = run_backtest(
                symbols=SYMBOLS,
                start_date=s["start"],
                end_date=s["end"],
                strategy_type=strategy_type,
                initial_capital=capital,
                in_sample_months=6,
                out_of_sample_months=3,
                output_dir=f"backtest/results/{s['name']}",
            )

            # Monte Carlo on the trade list from this scenario
            mc = {}
            if results.trades and len(results.trades) >= 10:
                mc = run_monte_carlo(results.trades, capital)

            row = {
                "scenario": s["label"],
                "period": f"{s['start']} → {s['end']}",
                "total_return_pct": results.total_return_pct,
                "benchmark_return_pct": results.benchmark_return_pct,
                "alpha_pct": results.alpha_pct,
                "sharpe": results.sharpe_ratio,
                "max_drawdown_pct": results.max_drawdown_pct,
                "total_trades": results.total_trades,
                "win_rate_pct": results.win_rate_pct,
                "avg_win_pct": results.avg_win_pct,
                "avg_loss_pct": results.avg_loss_pct,
                "profit_factor": results.profit_factor,
                "avg_holding_days": results.avg_holding_days,
                "mc_p5_sharpe": mc.get("sharpe_p5", "N/A"),
                "mc_robust": mc.get("is_robust", "N/A"),
                "mc_worst5_dd": mc.get("max_dd_worst5_pct", "N/A"),
            }
            summary_rows.append(row)

            # Print scenario summary
            print(f"\n  ── RESULTS ──")
            print(f"  Strategy return:   {results.total_return_pct:+.1f}%")
            print(f"  Nifty benchmark:   {results.benchmark_return_pct:+.1f}%")
            print(f"  Alpha:             {results.alpha_pct:+.1f}%")
            print(f"  Sharpe ratio:      {results.sharpe_ratio:.2f}")
            print(f"  Max drawdown:      {results.max_drawdown_pct:.1f}%")
            print(f"  Total trades:      {results.total_trades}")
            print(f"  Win rate:          {results.win_rate_pct:.1f}%")
            print(f"  Avg win / loss:    +{results.avg_win_pct:.1f}% / {results.avg_loss_pct:.1f}%")
            print(f"  Profit factor:     {results.profit_factor:.2f}")
            if mc:
                robust = "✅ PASS" if mc.get("is_robust") else "❌ FAIL"
                print(f"  Monte Carlo:       {robust} | p5 Sharpe: {mc.get('sharpe_p5', 0):.2f} | Worst 5% DD: {mc.get('max_dd_worst5_pct', 0):.1f}%")

        except Exception as e:
            print(f"  ERROR in scenario {s['name']}: {e}")
            import traceback
            traceback.print_exc()

    # ── Print master comparison table ─────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"MASTER COMPARISON TABLE — {strategy_type.upper()} STRATEGY")
    print(f"{'='*90}")
    print(f"{'Scenario':<42} {'Return':>8} {'Nifty':>7} {'Alpha':>7} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} {'PF':>5} {'MC':>6}")
    print(f"{'-'*90}")
    for r in summary_rows:
        mc_tag = "✅" if r['mc_robust'] is True else ("❌" if r['mc_robust'] is False else "—")
        print(
            f"{r['scenario'][:42]:<42} "
            f"{r['total_return_pct']:>+7.1f}% "
            f"{r['benchmark_return_pct']:>+6.1f}% "
            f"{r['alpha_pct']:>+6.1f}% "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_drawdown_pct']:>6.1f}% "
            f"{r['win_rate_pct']:>5.1f}% "
            f"{r['profit_factor']:>5.2f} "
            f"{mc_tag:>5}"
        )

    # Save summary JSON
    os.makedirs("backtest/results", exist_ok=True)
    out_path = f"backtest/results/master_summary_{strategy_type}.json"
    with open(out_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved → {out_path}")

    # ── Deployment guidance ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DEPLOYMENT GUIDANCE")
    print(f"{'='*70}")

    full = next((r for r in summary_rows if "Full Period" in r["scenario"]), None)
    if full:
        issues = []
        if full["sharpe"] < 0.8:
            issues.append(f"❌ Sharpe {full['sharpe']:.2f} below 0.8 — strategy needs tuning")
        if full["max_drawdown_pct"] < -25:
            issues.append(f"❌ Max drawdown {full['max_drawdown_pct']:.1f}% too deep for swing trading")
        if full["win_rate_pct"] < 45:
            issues.append(f"❌ Win rate {full['win_rate_pct']:.1f}% below 45% — too many losses")
        if full["profit_factor"] < 1.3:
            issues.append(f"❌ Profit factor {full['profit_factor']:.2f} below 1.3 — not enough edge")
        if full["mc_robust"] is False:
            issues.append(f"❌ Monte Carlo FAIL — strategy performance is fragile (p5 Sharpe ≤ 0)")

        covid_s = next((r for r in summary_rows if "COVID Crash" in r["scenario"]), None)
        if covid_s and covid_s["max_drawdown_pct"] < -30:
            issues.append(f"❌ COVID crash drawdown {covid_s['max_drawdown_pct']:.1f}% — circuit breaker not protecting capital")

        if not issues:
            print("✅ All checks passed. System is ready for paper trading.")
            print("   Next step: 60 days paper trading → compare vs these backtest results.")
            print("   If paper results within ±20% of backtest OOS → consider small live allocation (25% of capital).")
        else:
            print("System needs adjustment before paper trading:")
            for issue in issues:
                print(f"  {issue}")

    return summary_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest scenarios")
    parser.add_argument("--strategy", default="swing", choices=["swing", "positional"],
                        help="Strategy type to test (default: swing)")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Initial capital in INR (default: 500000)")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Run single scenario by number 1-6 (default: 0 = all)")
    args = parser.parse_args()

    if args.scenario > 0:
        SCENARIOS = [SCENARIOS[args.scenario - 1]]

    run_all_scenarios(strategy_type=args.strategy, capital=args.capital)
