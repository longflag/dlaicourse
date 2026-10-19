"""
가격 데이터 수집.

Binance 공개 API 사용 — API 키 불필요, 전체 히스토리 제공, rate limit 여유.
받은 데이터는 로컬에 캐시하므로 백테스트를 반복해도 재다운로드하지 않는다.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from . import config

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

# data-api.binance.vision 은 인증/지역 제한 없는 공개 시세 전용 엔드포인트
HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]

_LIMIT = 1000  # klines 1회 최대 응답 개수
_MS_DAY = 86_400_000


def _get(path, params):
    """호스트 폴백 + 지수 백오프 재시도."""
    last_err = None
    for host in HOSTS:
        for attempt in range(4):
            try:
                resp = requests.get(host + path, params=params, timeout=20)
                if resp.status_code in (429, 418):
                    time.sleep(2**attempt * 5)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as err:
                last_err = err
                time.sleep(2**attempt)
    raise RuntimeError(f"가격 데이터 요청 실패: {last_err}")


def _fetch_klines(symbol, start_ms):
    """start_ms 이후의 일봉을 전부 가져온다 (1000개씩 페이지네이션)."""
    rows = []
    cursor = start_ms
    while True:
        batch = _get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1d", "startTime": cursor, "limit": _LIMIT},
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < _LIMIT:
            break
        cursor = batch[-1][0] + _MS_DAY
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame(columns=["close", "volume"])

    df = pd.DataFrame(rows).iloc[:, [0, 4, 5]]
    df.columns = ["open_time", "close", "volume"]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.set_index("date")[["close", "volume"]].sort_index()


def load_symbol(symbol, start=None, use_cache=True):
    """단일 심볼의 일봉 종가/거래량을 반환한다 (캐시 우선, 증분 갱신)."""
    start = start or config.DATA_START
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_1d.csv"

    cached = None
    if use_cache and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True).sort_index()
        # 어제까지의 데이터가 있으면 그대로 사용
        yesterday = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=1)
        if not cached.empty and cached.index[-1] >= yesterday:
            return cached

    if cached is not None and not cached.empty:
        fetch_from = int(cached.index[-1].timestamp() * 1000) + _MS_DAY
        fresh = _fetch_klines(symbol, fetch_from)
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        fetch_from = int(pd.Timestamp(start).timestamp() * 1000)
        combined = _fetch_klines(symbol, fetch_from)

    if not combined.empty:
        combined.to_csv(cache_file)
    return combined


def load_prices(assets=None, start=None, use_cache=True):
    """
    전 자산의 일별 종가를 하나의 DataFrame으로 반환한다.
    컬럼 = 자산 티커(BTC/ETH/...), 인덱스 = 날짜.
    상장 이전 구간은 NaN으로 남는다.
    """
    assets = assets or config.ASSETS
    series = {}
    for ticker, symbol in assets.items():
        df = load_symbol(symbol, start=start, use_cache=use_cache)
        if df.empty:
            continue
        series[ticker] = df["close"]

    if not series:
        raise RuntimeError("가격 데이터를 하나도 가져오지 못했습니다.")

    prices = pd.DataFrame(series).sort_index()
    # 거래소 점검 등으로 비는 날은 직전 종가로 채우되, 상장 전 구간은 NaN 유지
    return prices.ffill()
