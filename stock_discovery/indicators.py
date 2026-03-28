"""기술적 분석 지표 계산 모듈"""

import pandas as pd
import numpy as np
from config import (
    RSI_PERIOD, MA_SHORT, MA_LONG,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)


def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI (Relative Strength Index) 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """이동평균선 계산 (5, 20, 60, 120일)"""
    for window in [5, MA_SHORT, MA_LONG, 120]:
        df[f"MA{window}"] = df["Close"].rolling(window=window).mean()
    return df


def calc_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """볼린저 밴드 계산"""
    mid = df["Close"].rolling(window=BOLLINGER_PERIOD).mean()
    std = df["Close"].rolling(window=BOLLINGER_PERIOD).std()
    df["BB_upper"] = mid + BOLLINGER_STD * std
    df["BB_mid"] = mid
    df["BB_lower"] = mid - BOLLINGER_STD * std
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    return df


def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD 계산"""
    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def calc_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """거래량 비율 (현재 거래량 / 평균 거래량)"""
    df["Vol_MA20"] = df["Volume"].rolling(window=window).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_MA20"]
    return df


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """스토캐스틱 계산"""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()
    return df


def calc_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """모든 기술적 지표 계산"""
    df = df.copy()
    df["RSI"] = calc_rsi(df["Close"])
    df = calc_moving_averages(df)
    df = calc_bollinger_bands(df)
    df = calc_macd(df)
    df = calc_volume_ratio(df)
    df = calc_stochastic(df)

    # 일간 수익률
    df["Return"] = df["Close"].pct_change()
    # 변동성 (20일)
    df["Volatility"] = df["Return"].rolling(window=20).std() * np.sqrt(252) * 100

    return df
