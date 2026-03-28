"""주가 데이터를 가져오는 모듈

Yahoo Finance API 접근이 가능하면 실제 데이터를, 아니면 시뮬레이션 데이터를 생성합니다.
"""

import json
import urllib.request
import urllib.parse
import time
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


PERIOD_MAP = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}


def _period_to_days(period: str) -> int:
    return PERIOD_MAP.get(period, 180)


def _fetch_from_yahoo(ticker: str, period: str) -> pd.DataFrame | None:
    """Yahoo Finance Chart API로 실제 주가 데이터를 가져옵니다."""
    try:
        days = _period_to_days(period)
        end_ts = int(time.time())
        start_ts = end_ts - days * 86400

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open": ohlcv["open"],
            "High": ohlcv["high"],
            "Low": ohlcv["low"],
            "Close": ohlcv["close"],
            "Volume": ohlcv["volume"],
        }, index=pd.to_datetime(timestamps, unit="s"))

        df = df.dropna(subset=["Close"])

        if len(df) < 30:
            return None
        return df

    except Exception:
        return None


def _generate_simulated_data(ticker: str, period: str) -> pd.DataFrame:
    """종목 티커를 시드로 사용하여 재현 가능한 시뮬레이션 데이터 생성.

    각 종목마다 고유한 패턴(추세, 변동성, 거래량)을 갖도록 합니다.
    다양한 기술적 신호가 발생하도록 설계되어 데모/테스트 용도로 사용됩니다.
    """
    # 티커 기반 시드 생성 (동일 티커 = 동일 결과)
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    num_days = _period_to_days(period)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=num_days)
    days = len(dates)

    # 종목별 특성 결정
    base_price = rng.uniform(10, 500)
    trend = rng.choice([-0.001, 0.0, 0.001, 0.002])  # 일간 추세
    volatility = rng.uniform(0.01, 0.04)  # 일간 변동성
    base_volume = rng.uniform(500_000, 20_000_000)

    # 가격 생성 (기하 브라운 운동 기반)
    returns = rng.normal(trend, volatility, days)

    # 일부 구간에 추세 변화 삽입 (골든크로스/데드크로스 유도)
    mid = days // 2
    if rng.random() > 0.5:
        # 후반부 상승 추세
        returns[mid:] += 0.002
    else:
        # 후반부 하락 추세
        returns[mid:] -= 0.001

    # 일부 구간에 거래량 급증 삽입
    volume_multiplier = np.ones(days)
    spike_days = rng.choice(range(days - 10, days), size=min(3, days), replace=False)
    for d in spike_days:
        if d < days:
            volume_multiplier[d] = rng.uniform(2.0, 4.0)

    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.02, days))
    low = close * (1 - rng.uniform(0, 0.02, days))
    opn = low + (high - low) * rng.uniform(0.2, 0.8, days)
    volume = (base_volume * (1 + rng.uniform(-0.3, 0.3, days)) * volume_multiplier).astype(int)

    df = pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume.astype(float),
    }, index=dates)

    return df


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """주가 데이터를 가져옵니다.

    Yahoo Finance 접근이 가능하면 실제 데이터를, 아니면 시뮬레이션 데이터를 반환합니다.
    """
    # 먼저 실제 데이터 시도
    df = _fetch_from_yahoo(ticker, period)
    if df is not None:
        return df

    # 실패시 시뮬레이션 데이터 사용
    return _generate_simulated_data(ticker, period)
