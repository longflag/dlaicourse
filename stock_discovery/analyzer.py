"""종목 분석 및 점수 산출 모듈"""

import pandas as pd
import numpy as np


def calc_score(df: pd.DataFrame, signals: dict) -> int:
    """종합 점수 계산 (0~100)

    매수 신호 수, 추세, 모멘텀 등을 종합하여 점수화합니다.
    점수가 높을수록 매수 관점에서 유리한 종목입니다.
    """
    score = 50  # 기본 점수

    # 매수 신호 가점 (+8점씩)
    score += len(signals.get("buy", [])) * 8

    # 매도 신호 감점 (-10점씩)
    score -= len(signals.get("sell", [])) * 10

    last = df.iloc[-1]

    # RSI 기반 가감점
    if pd.notna(last.get("RSI")):
        rsi = last["RSI"]
        if 40 <= rsi <= 60:
            score += 5  # 안정적 구간
        elif rsi < 25:
            score += 3  # 극단적 과매도 - 반등 기대
        elif rsi > 80:
            score -= 5  # 극단적 과매수

    # 이평선 정배열 보너스
    if all(col in df.columns for col in ["MA5", "MA20", "MA60"]):
        if (pd.notna(last.get("MA5")) and pd.notna(last.get("MA20"))
                and pd.notna(last.get("MA60"))):
            if last["Close"] > last["MA5"] > last["MA20"] > last["MA60"]:
                score += 10  # 완벽 정배열

    # MACD 히스토그램 방향
    if "MACD_hist" in df.columns and len(df) >= 2:
        curr_hist = last.get("MACD_hist")
        prev_hist = df.iloc[-2].get("MACD_hist")
        if pd.notna(curr_hist) and pd.notna(prev_hist):
            if curr_hist > prev_hist and curr_hist > 0:
                score += 5  # MACD 히스토그램 상승 & 양수

    # 거래량 증가 보너스
    if pd.notna(last.get("Vol_ratio")):
        if 1.5 <= last["Vol_ratio"] <= 3.0:
            score += 5  # 적당한 거래량 증가

    return max(0, min(100, score))


def generate_summary(ticker: str, name: str, df: pd.DataFrame,
                     signals: dict, score: int) -> dict:
    """종목 분석 요약 생성"""
    last = df.iloc[-1]

    # 최근 5일/20일 수익률
    ret_5d = ((last["Close"] / df.iloc[-6]["Close"]) - 1) * 100 if len(df) >= 6 else None
    ret_20d = ((last["Close"] / df.iloc[-21]["Close"]) - 1) * 100 if len(df) >= 21 else None

    summary = {
        "티커": ticker,
        "종목명": name,
        "현재가": round(last["Close"], 2),
        "RSI": round(last["RSI"], 1) if pd.notna(last.get("RSI")) else "-",
        "거래량비율": round(last["Vol_ratio"], 2) if pd.notna(last.get("Vol_ratio")) else "-",
        "5일수익률(%)": round(ret_5d, 2) if ret_5d is not None else "-",
        "20일수익률(%)": round(ret_20d, 2) if ret_20d is not None else "-",
        "변동성(%)": round(last["Volatility"], 1) if pd.notna(last.get("Volatility")) else "-",
        "매수신호": ", ".join(signals["buy"]) if signals["buy"] else "없음",
        "매도신호": ", ".join(signals["sell"]) if signals["sell"] else "없음",
        "종합점수": score,
    }
    return summary
