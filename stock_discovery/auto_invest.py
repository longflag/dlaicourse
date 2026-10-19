#!/usr/bin/env python3
"""
규칙 기반 자동투자 시스템 (Rule-Based Automated Investing)

투자 원칙
---------
1. 규칙을 먼저 쓴다        : 적어두지 않은 것은 규칙이 아니라 느낌입니다.
2. 규칙을 검증한다         : 백테스트를 통과하지 못한 규칙은 쓰지 않습니다.
3. 관찰 빈도를 낮춘다      : 월 1회만 확인합니다. 매일 보면 매일 틀립니다.
4. 리스크를 먼저 정한다    : 종목 선정보다 비중과 손절 규칙이 중요합니다.
5. 기계적으로 집행한다     : 신호가 나오면 이유를 묻지 않고 그대로 실행합니다.

사용법
------
  # 1단계: 규칙 검증 (전략별 과거 성과 비교)
  python auto_invest.py compare --market us --period 5y

  # 2단계: 단일 전략 상세 백테스트
  python auto_invest.py backtest --strategy dual_momentum --market us

  # 3단계: 이번 달 실행 지시 (무엇을 얼마나 살지)
  python auto_invest.py signal --strategy dual_momentum --market us

  # 4단계: 보유 종목과 비교해 매매 지시서 생성
  python auto_invest.py signal --strategy dual_momentum --holdings holdings.json
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import KOREAN_STOCKS, US_STOCKS
from strategy import STRATEGIES
from portfolio import size_positions, RiskManager
from backtest import (
    build_price_panel, Backtester, calc_metrics,
    benchmark_equity, format_metrics_row, get_rebalance_dates,
)

warnings.filterwarnings("ignore")

SEPARATOR = "=" * 72


def select_stocks(market: str) -> dict:
    if market == "kr":
        return KOREAN_STOCKS
    if market == "us":
        return US_STOCKS
    return {**KOREAN_STOCKS, **US_STOCKS}


def print_header(title: str):
    print("\n" + SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)


def check_data_source(panel: pd.DataFrame):
    """시뮬레이션 데이터 여부를 경고합니다."""
    from data_fetcher import _fetch_from_yahoo
    test = _fetch_from_yahoo("SPY", "1mo")
    if test is None:
        print("\n  [경고] 외부 데이터 접속이 불가하여 시뮬레이션 데이터로 실행됩니다.")
        print("         성과 수치는 시스템 동작 확인용이며 실제 검증 결과가 아닙니다.")
        print("         인터넷이 되는 환경에서 다시 실행하세요.")
        return False
    return True


# ============================================================
# 명령 1: 전략 비교
# ============================================================

def cmd_compare(args):
    stocks = select_stocks(args.market)
    print_header(f"전략 비교 백테스트 ({args.market.upper()}, {args.period})")

    panel = build_price_panel(stocks, args.period)
    if panel.empty:
        print("  데이터를 가져올 수 없습니다.")
        return

    is_real = check_data_source(panel)

    print(f"\n  기간: {panel.index[0].date()} ~ {panel.index[-1].date()}")
    print(f"  종목 수: {len(panel.columns)}개")
    print(f"  리밸런싱: {args.rebalance}  |  거래비용: {args.cost}bps  |  "
          f"종목당 최대비중: {args.max_weight:.0%}")

    rm = None
    if args.max_dd:
        rm = RiskManager(max_drawdown_limit=args.max_dd,
                         recovery_threshold=args.max_dd / 2)
        print(f"  서킷브레이커: 낙폭 -{args.max_dd:.0%} 도달 시 현금화")

    bt = Backtester(
        panel,
        rebalance_freq=args.rebalance,
        cost_bps=args.cost,
        weight_scheme=args.weighting,
        max_weight=args.max_weight,
        risk_manager=rm,
    )

    print("\n  백테스트 실행 중...")
    rows = []
    results = {}

    for key, cls in STRATEGIES.items():
        strat = cls(top_n=args.top_n) if key != "buyhold" else cls()
        res = bt.run(strat)
        if not res:
            continue
        m = calc_metrics(res["equity"], warmup=res["warmup"])
        if not m:
            continue
        results[key] = (res, m)
        row = format_metrics_row(strat.name, m)
        row["회전율"] = f"{res['avg_turnover']:.0%}"
        rows.append(row)

    # 벤치마크 추가
    bench_ticker = "SPY" if args.market == "us" else None
    if bench_ticker and bench_ticker in panel.columns:
        warmup = max(r[0]["warmup"] for r in results.values()) if results else 0
        be = benchmark_equity(panel, bench_ticker)
        bm = calc_metrics(be, warmup=warmup)
        if bm:
            row = format_metrics_row(f"{bench_ticker} 매수보유", bm)
            row["회전율"] = "0%"
            rows.append(row)

    if not rows:
        print("  백테스트 가능한 결과가 없습니다. 기간을 늘려보세요 (--period 5y).")
        return

    print()
    print(tabulate(rows, headers="keys", tablefmt="grid", stralign="center"))

    # 최고 전략 안내
    best = max(results.items(), key=lambda x: x[1][1]["칼마지수"], default=None)
    if best:
        key, (res, m) = best
        print(f"\n  위험조정 수익 기준 최우수: {STRATEGIES[key]().name} "
              f"(칼마 {m['칼마지수']:.2f})")
        print(f"  → 상세 분석: python auto_invest.py backtest --strategy {key} "
              f"--market {args.market}")

    print("\n  [지표 읽는 법]")
    print("   CAGR    : 연평균 복리 수익률")
    print("   샤프     : 변동성 1단위당 초과수익. 1.0 이상이면 양호")
    print("   최대낙폭 : 고점 대비 최대 손실. 실제로 견딜 수 있는 수준인지가 핵심")
    print("   칼마     : CAGR / 최대낙폭. 높을수록 '덜 아프게 번다'")
    print("   회전율   : 리밸런싱당 교체 비율. 높으면 거래비용이 수익을 잠식")

    if not is_real:
        print("\n  ※ 위 수치는 시뮬레이션 데이터 기준이므로 참고하지 마세요.")


# ============================================================
# 명령 2: 단일 전략 백테스트
# ============================================================

def cmd_backtest(args):
    if args.strategy not in STRATEGIES:
        print(f"  알 수 없는 전략: {args.strategy}")
        print(f"  사용 가능: {', '.join(STRATEGIES.keys())}")
        return

    stocks = select_stocks(args.market)
    cls = STRATEGIES[args.strategy]
    strat = cls(top_n=args.top_n) if args.strategy != "buyhold" else cls()

    print_header(f"백테스트: {strat.name} ({args.market.upper()}, {args.period})")
    print(f"\n  전략 설명: {strat.description}")

    panel = build_price_panel(stocks, args.period)
    if panel.empty:
        print("  데이터를 가져올 수 없습니다.")
        return

    is_real = check_data_source(panel)

    rm = None
    if args.max_dd:
        rm = RiskManager(max_drawdown_limit=args.max_dd,
                         recovery_threshold=args.max_dd / 2)

    bt = Backtester(
        panel,
        rebalance_freq=args.rebalance,
        cost_bps=args.cost,
        weight_scheme=args.weighting,
        max_weight=args.max_weight,
        risk_manager=rm,
    )

    res = bt.run(strat)
    if not res:
        print("  백테스트 실패. 기간이 너무 짧습니다 (--period 5y 권장).")
        return

    m = calc_metrics(res["equity"], warmup=res["warmup"])
    if not m:
        print("  성과 지표를 계산할 수 없습니다.")
        return

    e = res["equity"].iloc[res["warmup"]:]

    print(f"\n  --- 운용 설정 ---")
    print(f"  기간         : {e.index[0].date()} ~ {e.index[-1].date()} "
          f"({m['기간(년)']:.1f}년)")
    print(f"  보유 종목 수 : {args.top_n}개")
    print(f"  리밸런싱     : {args.rebalance} ({res['rebalance_count']}회)")
    print(f"  비중 방식    : {args.weighting}")
    print(f"  거래비용     : {args.cost}bps")

    print(f"\n  --- 성과 ---")
    print(f"  초기 자본    : {e.iloc[0]:,.0f}원")
    print(f"  최종 자산    : {e.iloc[-1]:,.0f}원")
    print(f"  총수익률     : {m['총수익률']:+.1%}")
    print(f"  CAGR         : {m['CAGR']:+.2%}")

    print(f"\n  --- 위험 ---")
    print(f"  변동성       : {m['변동성']:.1%}")
    print(f"  최대낙폭     : {m['최대낙폭']:.1%}")
    print(f"  샤프지수     : {m['샤프지수']:.2f}")
    print(f"  소르티노     : {m['소르티노']:.2f}")
    print(f"  칼마지수     : {m['칼마지수']:.2f}")
    print(f"  월간승률     : {m['월간승률']:.0%}")

    print(f"\n  --- 비용/운용 ---")
    print(f"  평균 회전율  : {res['avg_turnover']:.0%} (리밸런싱 1회당)")
    print(f"  누적 거래비용: {res['total_cost']:,.0f}원 "
          f"(최종자산 대비 {res['total_cost']/e.iloc[-1]:.1%})")
    if rm:
        print(f"  서킷브레이커 발동: {res['halt_count']}회")

    # 연도별 수익률
    yearly = e.resample("YE").last().pct_change().dropna()
    if len(yearly) > 0:
        print(f"\n  --- 연도별 수익률 ---")
        for date, r in yearly.items():
            bar = "+" * int(abs(r) * 40) if r > 0 else "-" * int(abs(r) * 40)
            print(f"  {date.year}: {r:+7.1%}  {bar}")

    # 최근 목표 포트폴리오
    if res["weights"]:
        last_date, last_w = res["weights"][-1]
        if last_w:
            print(f"\n  --- 마지막 리밸런싱 포트폴리오 ({last_date.date()}) ---")
            for t, w in sorted(last_w.items(), key=lambda x: -x[1]):
                name = stocks.get(t, t)
                print(f"    {name:<24s} {w:6.1%}")
            cash_w = 1 - sum(last_w.values())
            if cash_w > 0.005:
                print(f"    {'현금':<24s} {cash_w:6.1%}")
        else:
            print(f"\n  --- 마지막 리밸런싱 ({last_date.date()}): 전액 현금 ---")

    if not is_real:
        print("\n  ※ 시뮬레이션 데이터 기준입니다. 실제 검증 결과가 아닙니다.")


# ============================================================
# 명령 3: 실행 지시서 생성
# ============================================================

def cmd_signal(args):
    if args.strategy not in STRATEGIES:
        print(f"  알 수 없는 전략: {args.strategy}")
        return

    stocks = select_stocks(args.market)
    cls = STRATEGIES[args.strategy]
    strat = cls(top_n=args.top_n) if args.strategy != "buyhold" else cls()

    print_header(f"실행 지시서: {strat.name} ({datetime.now().date()})")

    panel = build_price_panel(stocks, "2y")
    if panel.empty:
        print("  데이터를 가져올 수 없습니다.")
        return

    is_real = check_data_source(panel)

    # 오늘 기준 신호 생성
    raw = strat.generate_weights(panel)
    selected = list(raw.keys())

    if not selected:
        print("\n  === 목표 포트폴리오: 전액 현금 ===")
        print("\n  전략 규칙상 매수 조건을 만족하는 종목이 없습니다.")
        print("  이것도 하나의 신호입니다. 억지로 종목을 찾지 마세요.")
        return

    target = size_positions(
        selected, panel,
        scheme=args.weighting,
        max_weight=args.max_weight,
    )

    latest_prices = panel.iloc[-1]
    capital = args.capital

    print(f"\n  기준일   : {panel.index[-1].date()}")
    print(f"  운용자금 : {capital:,.0f}원")
    print(f"  보유종목 : {len(target)}개")

    # 목표 포트폴리오
    rows = []
    for t, w in sorted(target.items(), key=lambda x: -x[1]):
        price = latest_prices.get(t, np.nan)
        if np.isnan(price) or price <= 0:
            continue
        amount = capital * w
        qty = int(amount / price)
        rows.append({
            "종목명": stocks.get(t, t),
            "티커": t,
            "목표비중": f"{w:.1%}",
            "현재가": f"{price:,.2f}",
            "목표금액": f"{amount:,.0f}",
            "수량": qty,
        })

    print()
    print(tabulate(rows, headers="keys", tablefmt="grid", stralign="center"))

    cash_w = 1 - sum(target.values())
    if cash_w > 0.005:
        print(f"\n  현금 보유: {cash_w:.1%} ({capital * cash_w:,.0f}원)")

    # 보유 종목과 비교하여 매매 지시 생성
    if args.holdings and os.path.exists(args.holdings):
        with open(args.holdings, "r", encoding="utf-8") as f:
            holdings = json.load(f)
        print_trade_orders(holdings, target, latest_prices, stocks, capital)
    else:
        print("\n  [매매 지시서를 원하면]")
        print("  현재 보유 종목을 holdings.json 으로 저장한 뒤 --holdings 옵션을 쓰세요.")
        print('  예시: {"AAPL": 10, "MSFT": 5}   (티커: 보유수량)')

    # 다음 리밸런싱 안내
    print_next_rebalance(panel.index[-1], args.rebalance)

    if not is_real:
        print("\n  ※ 시뮬레이션 데이터 기준입니다. 실제 주문에 사용하지 마세요.")


def print_trade_orders(holdings: dict, target: dict, prices: pd.Series,
                       stocks: dict, capital: float):
    """현재 보유와 목표 비중을 비교해 매매 지시서를 만듭니다.

    목표 수량은 위에 출력한 목표 포트폴리오와 동일하게 운용자금(capital)
    기준으로 계산합니다. 보유 평가액과 운용자금의 차이는 현금입니다.
    """
    current_value = sum(
        qty * prices.get(t, 0) for t, qty in holdings.items()
        if not np.isnan(prices.get(t, np.nan))
    )

    orders = []
    all_tickers = set(holdings) | set(target)

    buy_amount = 0.0
    sell_amount = 0.0

    for t in sorted(all_tickers):
        price = prices.get(t, np.nan)
        if np.isnan(price) or price <= 0:
            continue
        cur_qty = holdings.get(t, 0)
        tgt_qty = int((capital * target.get(t, 0)) / price)
        diff = tgt_qty - cur_qty
        if diff == 0:
            continue

        amount = abs(diff) * price
        if diff > 0:
            buy_amount += amount
        else:
            sell_amount += amount

        orders.append({
            "종목명": stocks.get(t, t),
            "티커": t,
            "구분": "매수" if diff > 0 else "매도",
            "현재수량": cur_qty,
            "목표수량": tgt_qty,
            "주문수량": abs(diff),
            "금액": f"{amount:,.0f}",
        })

    print(f"\n  === 매매 지시서 ===")
    print(f"  운용자금       : {capital:,.0f}원")
    print(f"  현재 보유 평가액: {current_value:,.0f}원")
    print(f"  현금           : {capital - current_value:,.0f}원")

    if not orders:
        print("\n  변경 사항 없음. 그대로 유지하세요.")
        return

    print()
    print(tabulate(orders, headers="keys", tablefmt="grid",
                   stralign="center", numalign="center"))
    print(f"\n  매수 합계: {buy_amount:,.0f}원  |  매도 합계: {sell_amount:,.0f}원")
    print("  이 지시서대로만 실행하세요. 종목을 추가하거나 빼지 마세요.")


def print_next_rebalance(last_date: pd.Timestamp, freq: str):
    """다음 리밸런싱 시점을 안내합니다."""
    freq_kr = {"W": "매주", "M": "매월", "Q": "매분기", "Y": "매년"}
    if freq == "M":
        nxt = (last_date + pd.offsets.MonthEnd(1)).date()
    elif freq == "Q":
        nxt = (last_date + pd.offsets.QuarterEnd(1)).date()
    elif freq == "W":
        nxt = (last_date + pd.offsets.Week(weekday=4)).date()
    else:
        nxt = (last_date + pd.offsets.YearEnd(1)).date()

    print(f"\n  {'─' * 60}")
    print(f"  다음 리밸런싱: {nxt} ({freq_kr.get(freq, freq)} 말)")
    print(f"  그때까지 시세와 뉴스를 확인할 필요가 없습니다.")
    print(f"  확인하지 않는 것이 이 시스템의 기능입니다.")
    print(f"  {'─' * 60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="규칙 기반 자동투자 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, default_period="5y"):
        p.add_argument("--market", choices=["kr", "us", "all"], default="us",
                       help="대상 시장")
        p.add_argument("--period", default=default_period,
                       help="데이터 기간 (2y/3y/5y/10y)")
        p.add_argument("--top-n", type=int, default=10,
                       help="보유 종목 수")
        p.add_argument("--rebalance", choices=["W", "M", "Q", "Y"], default="M",
                       help="리밸런싱 주기 (기본: 월간)")
        p.add_argument("--cost", type=float, default=25.0,
                       help="편도 거래비용 bps (기본 25 = 0.25%%)")
        p.add_argument("--weighting", choices=["equal", "inverse_vol"],
                       default="inverse_vol", help="비중 산출 방식")
        p.add_argument("--max-weight", type=float, default=0.20,
                       help="종목당 최대 비중")
        p.add_argument("--max-dd", type=float, default=None,
                       help="서킷브레이커 낙폭 한도 (예: 0.20)")

    p_cmp = sub.add_parser("compare", help="전략별 성과 비교")
    add_common(p_cmp)

    p_bt = sub.add_parser("backtest", help="단일 전략 상세 백테스트")
    p_bt.add_argument("--strategy", default="dual_momentum",
                      choices=list(STRATEGIES.keys()))
    add_common(p_bt)

    p_sig = sub.add_parser("signal", help="이번 회차 실행 지시서 생성")
    p_sig.add_argument("--strategy", default="dual_momentum",
                       choices=list(STRATEGIES.keys()))
    p_sig.add_argument("--capital", type=float, default=10_000_000,
                       help="운용 자금")
    p_sig.add_argument("--holdings", default=None,
                       help="현재 보유 종목 JSON 파일 경로")
    add_common(p_sig, default_period="2y")

    args = parser.parse_args()

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "signal":
        cmd_signal(args)

    print("\n  ※ 투자 판단과 결과의 책임은 본인에게 있습니다.")
    print("  ※ 실제 자금 투입 전 반드시 모의투자로 최소 6개월 검증하세요.\n")


if __name__ == "__main__":
    main()
