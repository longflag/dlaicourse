"""
리밸런싱 일정 계산.

'오늘 거래해도 되는 날인가'를 판정하는 유일한 창구.
이 함수가 False를 주면 프로그램은 시그널을 계산하지 않는다.
"""

import pandas as pd

from . import config


def is_rebalance_date(date, cfg=config):
    """해당 날짜가 리밸런싱 예정일인가."""
    ts = pd.Timestamp(date).normalize()
    if cfg.REBALANCE_FREQ.upper() == "W":
        return ts.dayofweek == cfg.REBALANCE_DOW
    return ts.day == cfg.REBALANCE_DOM


def next_rebalance_date(after=None, cfg=config):
    """다음 리밸런싱 예정일."""
    ts = pd.Timestamp(after or pd.Timestamp.utcnow()).normalize()
    for offset in range(1, 400):
        candidate = ts + pd.Timedelta(days=offset)
        if is_rebalance_date(candidate, cfg):
            return candidate
    raise RuntimeError("다음 리밸런싱 날짜를 찾지 못했습니다. 설정을 확인하세요.")


def rebalance_dates(index, cfg=config):
    """주어진 날짜 인덱스 중 리밸런싱 예정일만 추린다."""
    return pd.DatetimeIndex([d for d in index if is_rebalance_date(d, cfg)])
