"""
백테스트 엔진.

규칙은 믿는 게 아니라 검증하는 것이다.
설계상 지켜지는 것들:
  - 미래참조 없음: t일 종가로 신호 산출 -> t+1일 종가에 체결
  - 수수료와 슬리피지를 실제 거래대금에 부과
  - 리밸런싱 밴드 적용 (불필요한 거래 억제)
  - 상장 이전 자산은 자동 제외
"""

import numpy as np
import pandas as pd

from . import config
from .schedule import rebalance_dates
from .strategy import target_weights


def run(prices, cfg=config, initial=10_000.0, start=None):
    """전략을 실행하고 일별 자산곡선과 거래 내역을 반환한다."""
    prices = prices.sort_index()
    if start is not None:
        prices = prices.loc[pd.Timestamp(start):]

    dates = prices.index
    returns = prices.pct_change()
    rebal = set(rebalance_dates(dates, cfg))

    positions = pd.Series(0.0, index=prices.columns)
    cash = float(initial)
    pending = None

    equity, exposures, trades = [], [], []

    for i, today in enumerate(dates):
        # 1) 보유 포지션에 당일 수익률 반영
        if i > 0:
            positions *= 1.0 + returns.loc[today].fillna(0.0)

        total = positions.sum() + cash

        # 2) 전일 신호를 오늘 종가에 체결 (미래참조 방지)
        if pending is not None and total > 0:
            current_w = positions / total
            drift = (pending - current_w).abs().max()
            if drift > cfg.REBALANCE_BAND:
                # 거래비용은 투자 가능 금액에서 먼저 차감한다.
                # 그러지 않으면 목표비중 100%일 때 현금이 마이너스가 된다(암묵적 레버리지).
                rate = cfg.FEE_RATE + cfg.SLIPPAGE
                investable = total
                for _ in range(2):  # 비용 <-> 거래규모 상호의존을 두 번 반복해 수렴시킴
                    target_value = pending * investable
                    traded = (target_value - positions).abs().sum()
                    cost = traded * rate
                    investable = total - cost
                target_value = pending * investable
                positions = target_value
                cash = investable - positions.sum()
                trades.append(
                    {
                        "date": today,
                        "turnover": traded / total,
                        "cost": cost,
                        "equity": total - cost,
                    }
                )
            pending = None

        # 3) 리밸런싱일이면 신호를 산출해 내일 체결 대기열에 올린다
        if today in rebal and i >= cfg.MIN_HISTORY_DAYS:
            weights, _ = target_weights(prices.loc[:today], cfg)
            pending = weights

        total = positions.sum() + cash
        equity.append(total)
        exposures.append(positions.sum() / total if total > 0 else 0.0)

    curve = pd.Series(equity, index=dates, name="equity")
    exposure = pd.Series(exposures, index=dates, name="exposure")
    return {
        "equity": curve,
        "exposure": exposure,
        "trades": pd.DataFrame(trades),
    }


# ─────────────────────────────────────────────────────────────
# 벤치마크
# ─────────────────────────────────────────────────────────────


def buy_and_hold(prices, ticker, initial=10_000.0):
    """단일 자산 단순 보유."""
    series = prices[ticker].dropna()
    return (series / series.iloc[0] * initial).rename(f"{ticker} 보유")


def equal_weight_hold(prices, cfg=config, initial=10_000.0):
    """4종목 동일비중, 매월 리밸런싱 (항상 풀 투자)."""
    dates = prices.index
    returns = prices.pct_change()
    rebal = set(rebalance_dates(dates, cfg))

    positions = pd.Series(0.0, index=prices.columns)
    cash = float(initial)
    values = []

    for i, today in enumerate(dates):
        if i > 0:
            positions *= 1.0 + returns.loc[today].fillna(0.0)
        total = positions.sum() + cash

        if today in rebal:
            live = [c for c in prices.columns if not np.isnan(prices.loc[today, c])]
            if live:
                target = pd.Series(0.0, index=prices.columns)
                target[live] = 1.0 / len(live)
                rate = cfg.FEE_RATE + cfg.SLIPPAGE
                investable = total
                for _ in range(2):
                    cost = (target * investable - positions).abs().sum() * rate
                    investable = total - cost
                positions = target * investable
                cash = investable - positions.sum()
        values.append(positions.sum() + cash)

    return pd.Series(values, index=dates, name="동일비중 보유")


def monthly_dca(prices, cfg=config, monthly=500.0):
    """매월 동일 금액 분할매수 (적립식). 투입 원금 대비 평가액을 추적."""
    dates = prices.index
    returns = prices.pct_change()
    rebal = set(rebalance_dates(dates, cfg))

    positions = pd.Series(0.0, index=prices.columns)
    invested = 0.0
    values, principal = [], []

    for i, today in enumerate(dates):
        if i > 0:
            positions *= 1.0 + returns.loc[today].fillna(0.0)
        if today in rebal:
            live = [c for c in prices.columns if not np.isnan(prices.loc[today, c])]
            if live:
                per = monthly / len(live) * (1 - cfg.FEE_RATE - cfg.SLIPPAGE)
                for c in live:
                    positions[c] += per
                invested += monthly
        values.append(positions.sum())
        principal.append(invested)

    return pd.Series(values, index=dates, name="적립식 DCA"), pd.Series(principal, index=dates)


# ─────────────────────────────────────────────────────────────
# 성과 지표
# ─────────────────────────────────────────────────────────────


def metrics(equity, exposure=None, trades=None):
    """자산곡선에서 핵심 성과 지표를 계산한다."""
    equity = equity.dropna()
    equity = equity[equity > 0]
    if len(equity) < 2:
        return {}

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if years > 0 else np.nan

    daily = equity.pct_change().dropna()
    vol = daily.std() * np.sqrt(365)
    sharpe = (daily.mean() * 365) / vol if vol > 0 else np.nan

    downside = daily[daily < 0].std() * np.sqrt(365)
    sortino = (daily.mean() * 365) / downside if downside > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    out = {
        "기간(년)": years,
        "총수익률": total_return,
        "CAGR": cagr,
        "연변동성": vol,
        "샤프": sharpe,
        "소르티노": sortino,
        "최대낙폭(MDD)": mdd,
        "칼마": calmar,
    }
    if exposure is not None:
        out["평균 투자비중"] = float(exposure.mean())
    if trades is not None and not trades.empty and years > 0:
        out["연간 거래횟수"] = len(trades) / years
        out["연간 회전율"] = trades["turnover"].sum() / years
        out["누적 거래비용"] = trades["cost"].sum()
    return out
