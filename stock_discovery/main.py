#!/usr/bin/env python3
"""
주식종목 발굴 프로그램 (Stock Discovery Program)

기술적 분석 지표를 활용하여 매수/매도 유망 종목을 자동으로 발굴합니다.

사용법:
    python main.py                      # 전체 종목 분석
    python main.py --market kr          # 한국 종목만 분석
    python main.py --market us          # 미국 종목만 분석
    python main.py --ticker 005930.KS   # 특정 종목 상세 분석
    python main.py --strategy buy       # 매수 신호 종목만
    python main.py --strategy sell      # 매도 신호 종목만
    python main.py --top 10             # 상위 10개만 표시
    python main.py --period 1y          # 1년치 데이터 분석
"""

import argparse
import sys
import warnings

import pandas as pd
from tabulate import tabulate

from config import KOREAN_STOCKS, US_STOCKS, DEFAULT_PERIOD
from data_fetcher import fetch_stock_data
from indicators import calc_all_indicators
from screener import run_screening, BUY_STRATEGIES, SELL_STRATEGIES
from analyzer import calc_score, generate_summary

warnings.filterwarnings("ignore")


def analyze_single_stock(ticker: str, name: str, period: str) -> dict | None:
    """단일 종목을 분석합니다."""
    df = fetch_stock_data(ticker, period)
    if df is None:
        return None

    df = calc_all_indicators(df)
    signals = run_screening(df)
    score = calc_score(df, signals)
    return generate_summary(ticker, name, df, signals, score)


def print_detail(ticker: str, name: str, period: str):
    """특정 종목의 상세 분석 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"  상세 분석: {name} ({ticker})")
    print(f"{'='*60}")

    df = fetch_stock_data(ticker, period)
    if df is None:
        print(f"  ⚠ 데이터를 가져올 수 없습니다: {ticker}")
        return

    df = calc_all_indicators(df)
    signals = run_screening(df)
    score = calc_score(df, signals)
    last = df.iloc[-1]

    print(f"\n  현재가: {last['Close']:.2f}")
    print(f"  거래량: {int(last['Volume']):,}")

    print(f"\n  --- 기술적 지표 ---")
    if pd.notna(last.get("RSI")):
        print(f"  RSI(14): {last['RSI']:.1f}")
    if pd.notna(last.get("MA5")):
        print(f"  MA5:  {last['MA5']:.2f}")
    if pd.notna(last.get("MA20")):
        print(f"  MA20: {last['MA20']:.2f}")
    if pd.notna(last.get("MA60")):
        print(f"  MA60: {last['MA60']:.2f}")
    if pd.notna(last.get("MACD")):
        print(f"  MACD: {last['MACD']:.4f}  Signal: {last['MACD_signal']:.4f}")
    if pd.notna(last.get("BB_upper")):
        print(f"  볼린저밴드: [{last['BB_lower']:.2f} / {last['BB_mid']:.2f} / {last['BB_upper']:.2f}]")
    if pd.notna(last.get("Stoch_K")):
        print(f"  스토캐스틱: K={last['Stoch_K']:.1f}  D={last['Stoch_D']:.1f}")
    if pd.notna(last.get("Vol_ratio")):
        print(f"  거래량비율: {last['Vol_ratio']:.2f}x (20일 평균 대비)")
    if pd.notna(last.get("Volatility")):
        print(f"  연환산 변동성: {last['Volatility']:.1f}%")

    # 수익률
    if len(df) >= 6:
        ret_5d = ((last["Close"] / df.iloc[-6]["Close"]) - 1) * 100
        print(f"\n  5일 수익률: {ret_5d:+.2f}%")
    if len(df) >= 21:
        ret_20d = ((last["Close"] / df.iloc[-21]["Close"]) - 1) * 100
        print(f"  20일 수익률: {ret_20d:+.2f}%")
    if len(df) >= 61:
        ret_60d = ((last["Close"] / df.iloc[-61]["Close"]) - 1) * 100
        print(f"  60일 수익률: {ret_60d:+.2f}%")

    print(f"\n  --- 감지된 신호 ---")
    if signals["buy"]:
        for s in signals["buy"]:
            print(f"  [매수] {s}")
    if signals["sell"]:
        for s in signals["sell"]:
            print(f"  [매도] {s}")
    if not signals["buy"] and not signals["sell"]:
        print("  특별한 신호 없음")

    print(f"\n  종합점수: {score}/100")
    print(f"{'='*60}\n")


def run_full_scan(stocks: dict, period: str, strategy: str, top_n: int):
    """전체 종목 스캔 실행"""
    results = []
    total = len(stocks)

    print(f"\n  총 {total}개 종목 분석 중...\n")

    for i, (ticker, name) in enumerate(stocks.items(), 1):
        sys.stdout.write(f"\r  [{i}/{total}] {name} ({ticker}) 분석 중...")
        sys.stdout.flush()
        result = analyze_single_stock(ticker, name, period)
        if result:
            results.append(result)

    print(f"\r  {total}개 종목 분석 완료!{' ' * 30}\n")

    if not results:
        print("  분석 가능한 종목이 없습니다.")
        return

    # 전략 필터링
    if strategy == "buy":
        results = [r for r in results if r["매수신호"] != "없음"]
        print(f"  [매수 신호 감지 종목: {len(results)}개]\n")
    elif strategy == "sell":
        results = [r for r in results if r["매도신호"] != "없음"]
        print(f"  [매도 신호 감지 종목: {len(results)}개]\n")

    # 점수 기준 정렬
    results.sort(key=lambda x: x["종합점수"], reverse=True)

    if top_n:
        results = results[:top_n]

    # 테이블 출력
    display_cols = ["종목명", "현재가", "RSI", "거래량비율",
                    "5일수익률(%)", "20일수익률(%)", "매수신호", "종합점수"]
    table_data = [{k: r[k] for k in display_cols} for r in results]

    print(tabulate(table_data, headers="keys", tablefmt="grid",
                   stralign="center", numalign="center"))

    # 매수/매도 신호 요약
    buy_stocks = [r for r in results if r["매수신호"] != "없음"]
    sell_stocks = [r for r in results if r["매도신호"] != "없음"]

    if buy_stocks:
        print(f"\n  === 매수 신호 종목 ({len(buy_stocks)}개) ===")
        for r in buy_stocks:
            print(f"  {r['종목명']:>12s}  점수:{r['종합점수']:3d}  신호: {r['매수신호']}")

    if sell_stocks:
        print(f"\n  === 매도/주의 신호 종목 ({len(sell_stocks)}개) ===")
        for r in sell_stocks:
            print(f"  {r['종목명']:>12s}  점수:{r['종합점수']:3d}  신호: {r['매도신호']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="주식종목 발굴 프로그램 - 기술적 분석 기반 종목 스크리닝"
    )
    parser.add_argument(
        "--market", choices=["kr", "us", "all"], default="all",
        help="분석 대상 시장 (kr: 한국, us: 미국, all: 전체)"
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="특정 종목 티커 (예: 005930.KS, AAPL)"
    )
    parser.add_argument(
        "--strategy", choices=["buy", "sell", "all"], default="all",
        help="필터링 전략 (buy: 매수신호, sell: 매도신호, all: 전체)"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="상위 N개 종목만 표시"
    )
    parser.add_argument(
        "--period", type=str, default=DEFAULT_PERIOD,
        help="데이터 조회 기간 (예: 3mo, 6mo, 1y, 2y)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("       주식종목 발굴 프로그램 (Stock Discovery)")
    print("       기술적 분석 기반 자동 종목 스크리닝")
    print("=" * 60)

    # 특정 종목 상세 분석
    if args.ticker:
        all_stocks = {**KOREAN_STOCKS, **US_STOCKS}
        name = all_stocks.get(args.ticker, args.ticker)
        print_detail(args.ticker, name, args.period)
        return

    # 시장 선택
    if args.market == "kr":
        stocks = KOREAN_STOCKS
        print("\n  [한국 시장 분석]")
    elif args.market == "us":
        stocks = US_STOCKS
        print("\n  [미국 시장 분석]")
    else:
        stocks = {**KOREAN_STOCKS, **US_STOCKS}
        print("\n  [전체 시장 분석]")

    run_full_scan(stocks, args.period, args.strategy, args.top)

    print("  ※ 본 프로그램은 투자 참고용이며, 투자 판단의 책임은 본인에게 있습니다.")
    print("  ※ 기술적 분석만으로는 완전한 투자 판단이 어렵습니다.\n")


if __name__ == "__main__":
    main()
