"""시장 상황 분석 모듈 (Market Conditions)

시장 지수(SPY, KOSPI 등)의 추세, 모멘텀, 변동성을 분석하여
전반적인 시장 상황을 판단합니다.
"""

import numpy as np
import pandas as pd

from data_fetcher import fetch_stock_data
from indicators import calc_all_indicators

# 시장 지수 티커
MARKET_INDEX = {
    "us": {"ticker": "SPY", "name": "S&P 500 ETF"},
    "kr": {"ticker": "^KS11", "name": "KOSPI"},
}

# 캐시: 동일 실행에서 시장 데이터를 한 번만 조회
_market_cache = {}


def _determine_market(ticker: str) -> str:
    """티커로 시장(kr/us)을 판별합니다."""
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "kr"
    return "us"


def analyze_market_conditions(market: str = "us", period: str = "6mo") -> dict:
    """시장 전체 상황을 분석합니다.

    Returns:
        dict with keys:
            trend: 'bullish' / 'bearish' / 'neutral'
            momentum_5d: 5일 수익률 (%)
            momentum_20d: 20일 수익률 (%)
            volatility: 연환산 변동성 (%)
            volatility_regime: 'low' / 'medium' / 'high'
            ma_status: 이평선 상태 설명
            score: 시장 점수 (0~100)
    """
    cache_key = f"{market}_{period}"
    if cache_key in _market_cache:
        return _market_cache[cache_key]

    index_info = MARKET_INDEX.get(market, MARKET_INDEX["us"])
    df = fetch_stock_data(index_info["ticker"], period)

    if df is None or len(df) < 60:
        result = {
            "trend": "neutral",
            "momentum_5d": 0,
            "momentum_20d": 0,
            "volatility": 20,
            "volatility_regime": "medium",
            "ma_status": "데이터 부족",
            "score": 50,
            "index_name": index_info["name"],
        }
        _market_cache[cache_key] = result
        return result

    df = calc_all_indicators(df)
    last = df.iloc[-1]

    # 추세 판단
    trend = "neutral"
    ma_status = ""

    if pd.notna(last.get("MA20")) and pd.notna(last.get("MA60")):
        if last["Close"] > last["MA20"] > last["MA60"]:
            trend = "bullish"
            ma_status = "정배열 (상승추세)"
        elif last["Close"] < last["MA20"] < last["MA60"]:
            trend = "bearish"
            ma_status = "역배열 (하락추세)"
        elif last["Close"] > last["MA20"]:
            trend = "neutral"
            ma_status = "단기 회복 중"
        else:
            trend = "neutral"
            ma_status = "혼조세"

    # 모멘텀
    mom_5d = ((last["Close"] / df.iloc[-6]["Close"]) - 1) * 100 if len(df) >= 6 else 0
    mom_20d = ((last["Close"] / df.iloc[-21]["Close"]) - 1) * 100 if len(df) >= 21 else 0

    # 변동성 레짐
    vol = last.get("Volatility", 20)
    if not pd.notna(vol):
        vol = 20
    vol = float(vol)

    if vol < 15:
        vol_regime = "low"
    elif vol < 25:
        vol_regime = "medium"
    else:
        vol_regime = "high"

    # 시장 점수 계산
    score = 50

    if trend == "bullish":
        score += 15
    elif trend == "bearish":
        score -= 15

    if mom_20d > 5:
        score += 10
    elif mom_20d > 0:
        score += 5
    elif mom_20d < -5:
        score -= 10
    elif mom_20d < 0:
        score -= 5

    if vol_regime == "low":
        score += 5  # 안정적 시장
    elif vol_regime == "high":
        score -= 10  # 불안정 시장

    # RSI 기반
    rsi = last.get("RSI")
    if pd.notna(rsi):
        if rsi > 70:
            score -= 5  # 과열
        elif rsi < 30:
            score += 5  # 반등 기대

    score = max(0, min(100, score))

    result = {
        "trend": trend,
        "momentum_5d": round(float(mom_5d), 2),
        "momentum_20d": round(float(mom_20d), 2),
        "volatility": round(vol, 1),
        "volatility_regime": vol_regime,
        "ma_status": ma_status,
        "score": score,
        "index_name": index_info["name"],
    }

    _market_cache[cache_key] = result
    return result


def calc_market_alignment(stock_df: pd.DataFrame, market: str,
                          period: str = "6mo") -> float:
    """개별 종목과 시장 지수의 추세 정렬도를 계산합니다 (-1.0 ~ +1.0).

    양수: 시장과 같은 방향으로 움직임
    음수: 시장과 반대 방향으로 움직임
    """
    index_info = MARKET_INDEX.get(market, MARKET_INDEX["us"])
    market_df = fetch_stock_data(index_info["ticker"], period)

    if market_df is None or len(market_df) < 20 or len(stock_df) < 20:
        return 0.0

    # 최근 20일 수익률 상관관계
    stock_returns = stock_df["Close"].pct_change().tail(20).dropna()
    market_returns = market_df["Close"].pct_change().tail(20).dropna()

    min_len = min(len(stock_returns), len(market_returns))
    if min_len < 10:
        return 0.0

    stock_returns = stock_returns.iloc[-min_len:].values
    market_returns = market_returns.iloc[-min_len:].values

    correlation = np.corrcoef(stock_returns, market_returns)[0, 1]

    if np.isnan(correlation):
        return 0.0

    return round(float(correlation), 3)


def get_market_summary(market: str, period: str = "6mo") -> str:
    """시장 상황 요약 문자열을 반환합니다."""
    mc = analyze_market_conditions(market, period)

    trend_kr = {"bullish": "상승세", "bearish": "하락세", "neutral": "혼조세"}
    vol_kr = {"low": "낮음", "medium": "보통", "high": "높음"}

    return (
        f"{mc['index_name']}: {trend_kr[mc['trend']]} | "
        f"5일 {mc['momentum_5d']:+.1f}% | "
        f"20일 {mc['momentum_20d']:+.1f}% | "
        f"변동성 {vol_kr[mc['volatility_regime']]}({mc['volatility']:.0f}%)"
    )


def clear_cache():
    """캐시를 초기화합니다."""
    _market_cache.clear()
