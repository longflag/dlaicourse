#!/usr/bin/env python3
"""
백테스트 — 규칙을 믿기 전에 검증한다.

전략을 벤치마크와 나란히 비교한다:
  - BTC 단순보유
  - 4종목 동일비중 보유
  - 적립식 DCA

사용법:
    python run_backtest.py
    python run_backtest.py --start 2020-01-01
    python run_backtest.py --vol-target 0.30 --ma 150
"""

import argparse

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from systematic import backtest, config, data

console = Console()

PERCENT_KEYS = {"총수익률", "CAGR", "연변동성", "최대낙폭(MDD)", "평균 투자비중"}


def fmt(key, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if key in PERCENT_KEYS:
        return f"{value:.1%}"
    if key == "누적 거래비용":
        return f"${value:,.0f}"
    return f"{value:,.2f}"


def main():
    parser = argparse.ArgumentParser(description="규칙 기반 전략 백테스트")
    parser.add_argument("--start", help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--initial", type=float, default=10_000.0, help="초기 자본 (USD)")
    parser.add_argument("--vol-target", type=float, help="포트폴리오 목표 변동성 (예: 0.30)")
    parser.add_argument("--ma", type=int, help="추세 필터 이동평균 일수")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 재다운로드")
    args = parser.parse_args()

    # 파라미터 오버라이드 (민감도 확인용)
    if args.vol_target:
        config.PORTFOLIO_VOL_TARGET = args.vol_target
    if args.ma:
        config.TREND_MA_DAYS = args.ma

    console.print("[dim]가격 데이터 로드 중...[/dim]")
    prices = data.load_prices(use_cache=not args.no_cache)

    console.print()
    console.rule("[bold cyan]백테스트[/bold cyan]")
    console.print(
        f"[dim]기간: {prices.index[0].date()} ~ {prices.index[-1].date()}  |  "
        f"자산: {', '.join(prices.columns)}  |  "
        f"리밸런싱: {'매월' if config.REBALANCE_FREQ == 'M' else '매주'}  |  "
        f"MA{config.TREND_MA_DAYS}  |  목표변동성 {config.PORTFOLIO_VOL_TARGET:.0%}[/dim]"
    )
    console.print()

    result = backtest.run(prices, config, initial=args.initial, start=args.start)
    strat_equity = result["equity"]

    # 벤치마크는 전략과 동일 구간으로 맞춘다
    window = prices.loc[strat_equity.index[0]:strat_equity.index[-1]]

    results = {
        "규칙 기반 전략": backtest.metrics(strat_equity, result["exposure"], result["trades"]),
    }
    for ticker in ["BTC", "ETH"]:
        if ticker in window.columns:
            bh = backtest.buy_and_hold(window, ticker, args.initial)
            results[f"{ticker} 단순보유"] = backtest.metrics(bh)

    ew = backtest.equal_weight_hold(window, config, args.initial)
    results["동일비중 보유"] = backtest.metrics(ew)

    dca_value, dca_principal = backtest.monthly_dca(window, config, monthly=500.0)

    # ── 비교 테이블 ──
    keys = [
        "기간(년)", "총수익률", "CAGR", "연변동성",
        "샤프", "소르티노", "최대낙폭(MDD)", "칼마", "평균 투자비중",
    ]
    table = Table(title="전략 vs 벤치마크", box=box.ROUNDED)
    table.add_column("지표", style="cyan")
    for name in results:
        table.add_column(name, justify="right")

    for key in keys:
        row = [key]
        for name in results:
            row.append(fmt(key, results[name].get(key)))
        style = "bold" if key in ("CAGR", "샤프", "최대낙폭(MDD)") else None
        table.add_row(*row, style=style)
    console.print(table)

    # ── 거래 통계 ──
    stats = results["규칙 기반 전략"]
    trade_bits = []
    for key in ("연간 거래횟수", "연간 회전율", "누적 거래비용"):
        if key in stats:
            trade_bits.append(f"{key}: [bold]{fmt(key, stats[key])}[/bold]")
    if trade_bits:
        console.print()
        console.print(
            Panel(
                "  |  ".join(trade_bits)
                + f"\n[dim]리밸런싱 시도 {len(backtest.rebalance_dates(strat_equity.index, config))}회 중 "
                f"실제 거래 {len(result['trades'])}회 — 나머지는 밴드 이내라 거래하지 않음[/dim]",
                title="거래 활동",
                border_style="dim",
            )
        )

    # ── 적립식 비교 ──
    final_dca = dca_value.iloc[-1]
    invested = dca_principal.iloc[-1]
    if invested > 0:
        console.print()
        console.print(
            Panel(
                f"매월 $500 적립 시: 투입 ${invested:,.0f} → 평가액 ${final_dca:,.0f} "
                f"([bold]{final_dca / invested - 1:+.1%}[/bold])\n"
                "[dim]적립식은 타이밍 판단을 아예 제거하는 가장 단순한 대안입니다.[/dim]",
                title="적립식 DCA 참고",
                border_style="dim",
            )
        )

    # ── 연도별 수익률 ──
    yearly = strat_equity.resample("YE").last().pct_change()
    first_year_value = strat_equity.resample("YE").last().iloc[0]
    yearly.iloc[0] = first_year_value / strat_equity.iloc[0] - 1

    ytable = Table(title="연도별 수익률", box=box.SIMPLE)
    ytable.add_column("연도", style="cyan")
    ytable.add_column("전략", justify="right")
    btc_yearly = None
    if "BTC" in window.columns:
        btc = backtest.buy_and_hold(window, "BTC", args.initial)
        btc_yearly = btc.resample("YE").last().pct_change()
        btc_yearly.iloc[0] = btc.resample("YE").last().iloc[0] / btc.iloc[0] - 1
        ytable.add_column("BTC 보유", justify="right")

    for ts, val in yearly.items():
        if pd.isna(val):
            continue
        row = [str(ts.year), f"[{'green' if val >= 0 else 'red'}]{val:+.1%}[/]"]
        if btc_yearly is not None:
            bval = btc_yearly.get(ts)
            row.append(
                f"[{'green' if bval >= 0 else 'red'}]{bval:+.1%}[/]"
                if bval is not None and not pd.isna(bval) else "-"
            )
        ytable.add_row(*row)
    console.print()
    console.print(ytable)

    console.print()
    console.print(
        Panel(
            "[bold]백테스트 해석 시 주의[/bold]\n"
            "· 암호화폐 역사는 짧습니다. 실질적으로 사이클 2~3개뿐이라 통계적 신뢰도가 낮습니다.\n"
            "· 여기 쓰인 4종목은 '살아남은' 종목입니다(생존 편향). 당시엔 사라진 코인도 후보였습니다.\n"
            "· 파라미터를 바꿔가며 최고 성과를 찾는 순간 과최적화가 시작됩니다.\n"
            "  --ma, --vol-target 을 몇 개 바꿔봐서 결과가 급변하면 그 규칙은 신뢰할 수 없습니다.\n"
            "· 과거 성과는 미래를 보장하지 않습니다. 이 백테스트는 규칙의 '견고함'을 보는 도구이지,\n"
            "  수익률을 예측하는 도구가 아닙니다.",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    main()
