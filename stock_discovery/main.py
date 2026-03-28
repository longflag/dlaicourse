#!/usr/bin/env python3
"""
주식종목 발굴 프로그램 (Stock Discovery Program)

기술적 분석 + 기업실적 + 뉴스감성 + 시장상황을 종합하여
매수/매도 유망 종목을 자동으로 발굴합니다.

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
from screener import run_screening
from analyzer import calc_score, generate_summary
from fundamental_fetcher import fetch_fundamentals, evaluate_fundamentals
from news_sentiment import fetch_news_sentiment, evaluate_sentiment
from market_conditions import (
    analyze_market_conditions, calc_market_alignment,
    get_market_summary, clear_cache,
)

warnings.filterwarnings("ignore")


def _determine_market(ticker: str) -> str:
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "kr"
    return "us"


def analyze_single_stock(ticker: str, name: str, period: str,
                         market_conds: dict = None) -> dict | None:
    """단일 종목을 분석합니다."""
    df = fetch_stock_data(ticker, period)
    if df is None:
        return None

    df = calc_all_indicators(df)
    signals = run_screening(df)

    # 기업 실적
    fundamentals = fetch_fundamentals(ticker)
    fund_eval = evaluate_fundamentals(fundamentals)

    # 뉴스 감성
    sentiment = fetch_news_sentiment(ticker, name)
    sent_eval = evaluate_sentiment(sentiment)

    # 시장 상황
    market = _determine_market(ticker)
    if market_conds is None:
        market_conds = analyze_market_conditions(market, period)
    alignment = calc_market_alignment(df, market, period)

    scores = calc_score(
        df, signals,
        fundamentals=fundamentals, fund_eval=fund_eval,
        sentiment=sentiment, sent_eval=sent_eval,
        market_conditions=market_conds,
        market_alignment=alignment,
    )

    return generate_summary(
        ticker, name, df, signals, scores,
        fundamentals=fundamentals, fund_eval=fund_eval,
        sentiment=sentiment, sent_eval=sent_eval,
        market_conditions=market_conds,
    )


def print_detail(ticker: str, name: str, period: str):
    """특정 종목의 상세 분석 결과를 출력합니다."""
    print(f"\n{'='*65}")
    print(f"  상세 분석: {name} ({ticker})")
    print(f"{'='*65}")

    df = fetch_stock_data(ticker, period)
    if df is None:
        print(f"  데이터를 가져올 수 없습니다: {ticker}")
        return

    df = calc_all_indicators(df)
    signals = run_screening(df)
    last = df.iloc[-1]

    # 기업 실적
    fundamentals = fetch_fundamentals(ticker)
    fund_eval = evaluate_fundamentals(fundamentals)

    # 뉴스 감성
    sentiment = fetch_news_sentiment(ticker, name)
    sent_eval = evaluate_sentiment(sentiment)

    # 시장 상황
    market = _determine_market(ticker)
    market_conds = analyze_market_conditions(market, period)
    alignment = calc_market_alignment(df, market, period)

    scores = calc_score(
        df, signals,
        fundamentals=fundamentals, fund_eval=fund_eval,
        sentiment=sentiment, sent_eval=sent_eval,
        market_conditions=market_conds,
        market_alignment=alignment,
    )

    # === 기본 정보 ===
    print(f"\n  현재가: {last['Close']:.2f}")
    print(f"  거래량: {int(last['Volume']):,}")

    # === 기술적 지표 ===
    print(f"\n  --- 기술적 분석 (점수: {scores['technical']}/100) ---")
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

    # === 기업 실적 ===
    print(f"\n  --- 기업 실적 (점수: {fund_eval['score']}/100, {fund_eval['label']}) ---")
    if fundamentals:
        per = fundamentals.get("per")
        print(f"  PER: {per if per else '-'}")
        print(f"  PBR: {fundamentals.get('pbr', '-')}")
        print(f"  EPS 성장률: {fundamentals.get('eps_growth', '-')}%")
        print(f"  매출 성장률: {fundamentals.get('revenue_growth', '-')}%")
        print(f"  이익률: {fundamentals.get('profit_margin', '-')}%")
        print(f"  ROE: {fundamentals.get('roe', '-')}%")
        print(f"  부채비율: {fundamentals.get('debt_to_equity', '-')}%")
        print(f"  배당수익률: {fundamentals.get('dividend_yield', '-')}%")
        src = "Yahoo Finance" if fundamentals.get("source") == "yahoo" else "시뮬레이션"
        print(f"  (데이터 출처: {src})")

    # === 뉴스 감성 ===
    print(f"\n  --- 뉴스 감성 (점수: {sent_eval['score']}/100, {sent_eval['label']}) ---")
    print(f"  감성 점수: {sentiment['sentiment_score']:+.3f}")
    print(f"  뉴스 건수: {sentiment['headline_count']}건")
    if sentiment.get("top_headlines"):
        print(f"  주요 헤드라인:")
        for h in sentiment["top_headlines"][:3]:
            print(f"    - {h}")
    src = "Yahoo Finance" if sentiment.get("source") == "yahoo" else "시뮬레이션"
    print(f"  (데이터 출처: {src})")

    # === 시장 상황 ===
    trend_kr = {"bullish": "상승세", "bearish": "하락세", "neutral": "혼조세"}
    vol_kr = {"low": "낮음", "medium": "보통", "high": "높음"}
    print(f"\n  --- 시장 상황 (점수: {market_conds['score']}/100) ---")
    print(f"  {market_conds['index_name']}: {trend_kr[market_conds['trend']]} ({market_conds['ma_status']})")
    print(f"  시장 모멘텀: 5일 {market_conds['momentum_5d']:+.1f}% / 20일 {market_conds['momentum_20d']:+.1f}%")
    print(f"  시장 변동성: {vol_kr[market_conds['volatility_regime']]} ({market_conds['volatility']:.1f}%)")
    print(f"  종목-시장 상관계수: {alignment:+.3f}")

    # === 감지된 신호 ===
    print(f"\n  --- 감지된 신호 ---")
    if signals["buy"]:
        for s in signals["buy"]:
            print(f"  [매수] {s}")
    if signals["sell"]:
        for s in signals["sell"]:
            print(f"  [매도] {s}")
    if not signals["buy"] and not signals["sell"]:
        print("  특별한 신호 없음")

    # === 종합 점수 ===
    print(f"\n  --- 종합 점수 ---")
    print(f"  기술적 분석: {scores['technical']:3d}/100 (가중치 40%)")
    print(f"  기업 실적:   {scores['fundamental']:>3s}/100 (가중치 30%)" if isinstance(scores['fundamental'], str) else f"  기업 실적:   {scores['fundamental']:3d}/100 (가중치 30%)")
    print(f"  뉴스 감성:   {scores['sentiment']:>3s}/100 (가중치 15%)" if isinstance(scores['sentiment'], str) else f"  뉴스 감성:   {scores['sentiment']:3d}/100 (가중치 15%)")
    print(f"  시장 상황:   {scores['market']:>3s}/100 (가중치 15%)" if isinstance(scores['market'], str) else f"  시장 상황:   {scores['market']:3d}/100 (가중치 15%)")
    print(f"  {'─'*30}")
    print(f"  종합 점수:   {scores['total']:3d}/100")
    print(f"{'='*65}\n")


def run_full_scan(stocks: dict, period: str, strategy: str, top_n: int):
    """전체 종목 스캔 실행"""
    results = []
    total = len(stocks)

    # 시장 상황 먼저 분석 (캐싱)
    print("\n  시장 상황 분석 중...")
    kr_market = analyze_market_conditions("kr", period)
    us_market = analyze_market_conditions("us", period)

    print(f"  KR: {get_market_summary('kr', period)}")
    print(f"  US: {get_market_summary('us', period)}")

    print(f"\n  총 {total}개 종목 분석 중 (기술+실적+뉴스+시장)...\n")

    for i, (ticker, name) in enumerate(stocks.items(), 1):
        sys.stdout.write(f"\r  [{i}/{total}] {name} ({ticker}) 분석 중...")
        sys.stdout.flush()
        market = _determine_market(ticker)
        mc = kr_market if market == "kr" else us_market
        result = analyze_single_stock(ticker, name, period, market_conds=mc)
        if result:
            results.append(result)

    print(f"\r  {total}개 종목 분석 완료!{' ' * 40}\n")

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

    # 테이블 출력 (확장)
    display_cols = [
        "종목명", "현재가", "PER", "뉴스감성", "실적판정",
        "기술점수", "실적점수", "감성점수", "시장점수", "종합점수",
    ]
    table_data = [{k: r[k] for k in display_cols} for r in results]

    print(tabulate(table_data, headers="keys", tablefmt="grid",
                   stralign="center", numalign="center"))

    # 매수/매도 신호 요약
    buy_stocks = [r for r in results if r["매수신호"] != "없음"]
    sell_stocks = [r for r in results if r["매도신호"] != "없음"]

    if buy_stocks:
        print(f"\n  === 매수 신호 종목 ({len(buy_stocks)}개) ===")
        for r in buy_stocks:
            sentiment_tag = f"[{r['뉴스감성']}]" if r["뉴스감성"] != "-" else ""
            fund_tag = f"[실적:{r['실적판정']}]" if r["실적판정"] != "-" else ""
            print(f"  {r['종목명']:>16s}  종합:{r['종합점수']:3d}  "
                  f"{fund_tag} {sentiment_tag}  신호: {r['매수신호']}")

    if sell_stocks:
        print(f"\n  === 매도/주의 신호 종목 ({len(sell_stocks)}개) ===")
        for r in sell_stocks:
            print(f"  {r['종목명']:>16s}  종합:{r['종합점수']:3d}  신호: {r['매도신호']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="주식종목 발굴 프로그램 - 기술+실적+뉴스+시장 종합 분석"
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

    print("\n" + "=" * 65)
    print("         주식종목 발굴 프로그램 (Stock Discovery)")
    print("     기술분석 + 기업실적 + 뉴스감성 + 시장상황 종합 분석")
    print("=" * 65)

    clear_cache()

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
    print("  ※ 점수 가중치: 기술(40%) + 실적(30%) + 뉴스(15%) + 시장(15%)\n")


if __name__ == "__main__":
    main()
