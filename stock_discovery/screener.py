"""종목 스크리닝 전략 모듈"""

import pandas as pd
from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SURGE_RATIO


def screen_golden_cross(df: pd.DataFrame) -> bool:
    """골든크로스 감지: 단기(20일) 이평선이 장기(60일) 이평선을 상향 돌파"""
    if len(df) < 3:
        return False
    recent = df.iloc[-3:]
    if "MA20" not in df.columns or "MA60" not in df.columns:
        return False
    prev = recent.iloc[-2]
    curr = recent.iloc[-1]
    return (
        pd.notna(prev["MA20"]) and pd.notna(prev["MA60"])
        and pd.notna(curr["MA20"]) and pd.notna(curr["MA60"])
        and prev["MA20"] <= prev["MA60"]
        and curr["MA20"] > curr["MA60"]
    )


def screen_dead_cross(df: pd.DataFrame) -> bool:
    """데드크로스 감지: 단기(20일) 이평선이 장기(60일) 이평선을 하향 돌파"""
    if len(df) < 3:
        return False
    recent = df.iloc[-3:]
    if "MA20" not in df.columns or "MA60" not in df.columns:
        return False
    prev = recent.iloc[-2]
    curr = recent.iloc[-1]
    return (
        pd.notna(prev["MA20"]) and pd.notna(prev["MA60"])
        and pd.notna(curr["MA20"]) and pd.notna(curr["MA60"])
        and prev["MA20"] >= prev["MA60"]
        and curr["MA20"] < curr["MA60"]
    )


def screen_rsi_oversold(df: pd.DataFrame) -> bool:
    """RSI 과매도 상태 (반등 가능성)"""
    if "RSI" not in df.columns:
        return False
    rsi = df["RSI"].iloc[-1]
    return pd.notna(rsi) and rsi < RSI_OVERSOLD


def screen_rsi_overbought(df: pd.DataFrame) -> bool:
    """RSI 과매수 상태 (조정 가능성)"""
    if "RSI" not in df.columns:
        return False
    rsi = df["RSI"].iloc[-1]
    return pd.notna(rsi) and rsi > RSI_OVERBOUGHT


def screen_volume_surge(df: pd.DataFrame) -> bool:
    """거래량 급증 감지"""
    if "Vol_ratio" not in df.columns:
        return False
    vol_ratio = df["Vol_ratio"].iloc[-1]
    return pd.notna(vol_ratio) and vol_ratio > VOLUME_SURGE_RATIO


def screen_bollinger_breakout_low(df: pd.DataFrame) -> bool:
    """볼린저밴드 하단 이탈 (반등 매수 신호)"""
    if "BB_lower" not in df.columns:
        return False
    last = df.iloc[-1]
    return pd.notna(last["BB_lower"]) and last["Close"] < last["BB_lower"]


def screen_bollinger_breakout_high(df: pd.DataFrame) -> bool:
    """볼린저밴드 상단 돌파 (강한 상승 추세)"""
    if "BB_upper" not in df.columns:
        return False
    last = df.iloc[-1]
    return pd.notna(last["BB_upper"]) and last["Close"] > last["BB_upper"]


def screen_macd_golden_cross(df: pd.DataFrame) -> bool:
    """MACD 골든크로스: MACD가 시그널선을 상향 돌파"""
    if "MACD" not in df.columns or "MACD_signal" not in df.columns:
        return False
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        pd.notna(prev["MACD"]) and pd.notna(prev["MACD_signal"])
        and pd.notna(curr["MACD"]) and pd.notna(curr["MACD_signal"])
        and prev["MACD"] <= prev["MACD_signal"]
        and curr["MACD"] > curr["MACD_signal"]
    )


def screen_uptrend(df: pd.DataFrame) -> bool:
    """상승 추세: 가격 > MA20 > MA60"""
    if "MA20" not in df.columns or "MA60" not in df.columns:
        return False
    last = df.iloc[-1]
    return (
        pd.notna(last["MA20"]) and pd.notna(last["MA60"])
        and last["Close"] > last["MA20"] > last["MA60"]
    )


def screen_stochastic_oversold(df: pd.DataFrame) -> bool:
    """스토캐스틱 과매도 후 반전 신호"""
    if "Stoch_K" not in df.columns or "Stoch_D" not in df.columns:
        return False
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        pd.notna(prev["Stoch_K"]) and pd.notna(prev["Stoch_D"])
        and pd.notna(curr["Stoch_K"]) and pd.notna(curr["Stoch_D"])
        and prev["Stoch_K"] < 20
        and curr["Stoch_K"] > curr["Stoch_D"]
        and prev["Stoch_K"] <= prev["Stoch_D"]
    )


# 매수 관점 전략 모음
BUY_STRATEGIES = {
    "골든크로스 (MA20/60)": screen_golden_cross,
    "RSI 과매도 (<30)": screen_rsi_oversold,
    "거래량 급증": screen_volume_surge,
    "볼린저밴드 하단 이탈": screen_bollinger_breakout_low,
    "MACD 골든크로스": screen_macd_golden_cross,
    "상승 추세 (가격>MA20>MA60)": screen_uptrend,
    "스토캐스틱 과매도 반전": screen_stochastic_oversold,
}

# 매도/주의 관점 전략 모음
SELL_STRATEGIES = {
    "데드크로스 (MA20/60)": screen_dead_cross,
    "RSI 과매수 (>70)": screen_rsi_overbought,
    "볼린저밴드 상단 돌파": screen_bollinger_breakout_high,
}


def run_screening(df: pd.DataFrame) -> dict:
    """모든 전략을 실행하고 매칭된 신호를 반환"""
    signals = {"buy": [], "sell": []}

    for name, func in BUY_STRATEGIES.items():
        if func(df):
            signals["buy"].append(name)

    for name, func in SELL_STRATEGIES.items():
        if func(df):
            signals["sell"].append(name)

    return signals
