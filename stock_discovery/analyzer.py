"""종목 분석 및 점수 산출 모듈

기술적 분석 + 기업실적 + 뉴스감성 + 시장상황을 종합하여 점수를 산출합니다.

점수 가중치:
  - 기술적 분석: 40%
  - 기업 실적:   30%
  - 뉴스 감성:   15%
  - 시장 상황:   15%
"""

import pandas as pd
import numpy as np

# 가중치 설정
WEIGHT_TECHNICAL = 40
WEIGHT_FUNDAMENTAL = 30
WEIGHT_SENTIMENT = 15
WEIGHT_MARKET = 15


def _calc_technical_score(df: pd.DataFrame, signals: dict) -> int:
    """기술적 분석 점수 (0~100)"""
    score = 50

    score += len(signals.get("buy", [])) * 8
    score -= len(signals.get("sell", [])) * 10

    last = df.iloc[-1]

    if pd.notna(last.get("RSI")):
        rsi = last["RSI"]
        if 40 <= rsi <= 60:
            score += 5
        elif rsi < 25:
            score += 3
        elif rsi > 80:
            score -= 5

    if all(col in df.columns for col in ["MA5", "MA20", "MA60"]):
        if (pd.notna(last.get("MA5")) and pd.notna(last.get("MA20"))
                and pd.notna(last.get("MA60"))):
            if last["Close"] > last["MA5"] > last["MA20"] > last["MA60"]:
                score += 10

    if "MACD_hist" in df.columns and len(df) >= 2:
        curr_hist = last.get("MACD_hist")
        prev_hist = df.iloc[-2].get("MACD_hist")
        if pd.notna(curr_hist) and pd.notna(prev_hist):
            if curr_hist > prev_hist and curr_hist > 0:
                score += 5

    if pd.notna(last.get("Vol_ratio")):
        if 1.5 <= last["Vol_ratio"] <= 3.0:
            score += 5

    return max(0, min(100, score))


def calc_score(df: pd.DataFrame, signals: dict,
               fundamentals: dict = None, fund_eval: dict = None,
               sentiment: dict = None, sent_eval: dict = None,
               market_conditions: dict = None,
               market_alignment: float = None) -> dict:
    """종합 점수 계산.

    Returns:
        dict with keys: total, technical, fundamental, sentiment, market
    """
    tech_score = _calc_technical_score(df, signals)

    fund_score = fund_eval["score"] if fund_eval else None
    sent_score = sent_eval["score"] if sent_eval else None

    # 시장 점수: 시장 상황 + 개별 종목 정렬도 반영
    mkt_score = None
    if market_conditions:
        mkt_score = market_conditions["score"]
        if market_alignment is not None:
            # 시장이 좋고 종목도 시장과 정렬 -> 보너스
            alignment_bonus = int(market_alignment * 10)
            if market_conditions["trend"] == "bullish":
                mkt_score += alignment_bonus
            elif market_conditions["trend"] == "bearish":
                mkt_score -= alignment_bonus
            mkt_score = max(0, min(100, mkt_score))

    # 가중 평균 계산 (None인 항목은 가중치 재분배)
    components = {
        "technical": (tech_score, WEIGHT_TECHNICAL),
        "fundamental": (fund_score, WEIGHT_FUNDAMENTAL),
        "sentiment": (sent_score, WEIGHT_SENTIMENT),
        "market": (mkt_score, WEIGHT_MARKET),
    }

    total_weight = 0
    weighted_sum = 0
    for key, (s, w) in components.items():
        if s is not None:
            total_weight += w
            weighted_sum += s * w

    total = int(weighted_sum / total_weight) if total_weight > 0 else tech_score

    return {
        "total": max(0, min(100, total)),
        "technical": tech_score,
        "fundamental": fund_score if fund_score is not None else "-",
        "sentiment": sent_score if sent_score is not None else "-",
        "market": mkt_score if mkt_score is not None else "-",
    }


def generate_summary(ticker: str, name: str, df: pd.DataFrame,
                     signals: dict, scores: dict,
                     fundamentals: dict = None, fund_eval: dict = None,
                     sentiment: dict = None, sent_eval: dict = None,
                     market_conditions: dict = None) -> dict:
    """종목 분석 요약 생성"""
    last = df.iloc[-1]

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
        # 점수
        "종합점수": scores["total"],
        "기술점수": scores["technical"],
        "실적점수": scores["fundamental"],
        "감성점수": scores["sentiment"],
        "시장점수": scores["market"],
    }

    # 기업 실적 데이터
    if fundamentals:
        summary["PER"] = fundamentals.get("per", "-")
        summary["매출성장(%)"] = fundamentals.get("revenue_growth", "-")
        summary["이익률(%)"] = fundamentals.get("profit_margin", "-")
        summary["실적판정"] = fund_eval["label"] if fund_eval else "-"
    else:
        summary["PER"] = "-"
        summary["매출성장(%)"] = "-"
        summary["이익률(%)"] = "-"
        summary["실적판정"] = "-"

    # 뉴스 감성
    if sent_eval:
        summary["뉴스감성"] = sent_eval["label"]
    else:
        summary["뉴스감성"] = "-"

    return summary
