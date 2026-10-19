"""
전략 규칙 — 신호 산출과 목표 비중 계산.

여기에 있는 함수는 모두 '결정 시점까지의 데이터'만 입력받는다.
미래 데이터를 볼 수 없는 구조이므로 백테스트에 look-ahead bias가 생기지 않는다.
"""

import numpy as np
import pandas as pd

from . import config

TRADING_DAYS = 365  # 암호화폐는 연중무휴


def annualized_vol(prices, lookback=None):
    """최근 lookback일 로그수익률의 연환산 변동성."""
    lookback = lookback or config.VOL_LOOKBACK_DAYS
    rets = np.log(prices / prices.shift(1)).dropna()
    if len(rets) < max(20, lookback // 3):
        return np.nan
    return rets.tail(lookback).std() * np.sqrt(TRADING_DAYS)


def momentum_score(prices, lookbacks=None, weights=None):
    """여러 기간 수익률의 가중 평균. 단일 기간에 의존하지 않게 분산."""
    lookbacks = lookbacks or config.MOMENTUM_LOOKBACKS
    weights = weights or config.MOMENTUM_WEIGHTS

    total, wsum = 0.0, 0.0
    for lb, w in zip(lookbacks, weights):
        if len(prices) <= lb:
            continue
        ret = prices.iloc[-1] / prices.iloc[-1 - lb] - 1.0
        if np.isfinite(ret):
            total += w * ret
            wsum += w
    return total / wsum if wsum > 0 else np.nan


def passes_trend_filter(prices, ma_days=None, confirm_days=None):
    """가격이 장기 이동평균 위에 confirm_days 연속 머물렀는가."""
    ma_days = ma_days or config.TREND_MA_DAYS
    confirm_days = confirm_days or config.TREND_CONFIRM_DAYS

    if len(prices) < ma_days + confirm_days:
        return False
    ma = prices.rolling(ma_days).mean()
    recent = prices.iloc[-confirm_days:] > ma.iloc[-confirm_days:]
    return bool(recent.all())


def _apply_weight_cap(weights, cap):
    """개별 종목 상한을 적용하고 초과분을 나머지에 재분배."""
    w = weights.copy()
    for _ in range(10):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        room = (~over) & (w > 0)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    return w


def target_weights(prices, cfg=config, verbose=False):
    """
    결정 시점(prices의 마지막 행)에서의 목표 비중을 산출한다.

    반환: (pd.Series 자산->비중, 판정 근거 dict).
    비중 합계 < 1 이면 나머지는 현금(스테이블코인).
    """
    tickers = list(prices.columns)
    weights = pd.Series(0.0, index=tickers)
    diagnostics = {}

    # ── 1단계: 보유 자격 심사 (추세 필터 + 절대 모멘텀) ──
    eligible = []
    for t in tickers:
        series = prices[t].dropna()
        note = {}

        if len(series) < cfg.MIN_HISTORY_DAYS:
            note["status"] = "제외: 히스토리 부족"
            diagnostics[t] = note
            continue

        trend_ok = passes_trend_filter(series, cfg.TREND_MA_DAYS, cfg.TREND_CONFIRM_DAYS)
        mom = momentum_score(series, cfg.MOMENTUM_LOOKBACKS, cfg.MOMENTUM_WEIGHTS)
        mom_ok = np.isfinite(mom) and mom > cfg.MOMENTUM_MIN
        vol = annualized_vol(series, cfg.VOL_LOOKBACK_DAYS)

        ma = series.rolling(cfg.TREND_MA_DAYS).mean().iloc[-1]
        note.update(
            {
                "price": series.iloc[-1],
                "ma200": ma,
                "trend_ok": trend_ok,
                "momentum": mom,
                "mom_ok": mom_ok,
                "vol": vol,
            }
        )

        if trend_ok and mom_ok and np.isfinite(vol) and vol > 0:
            eligible.append(t)
            note["status"] = "보유"
        else:
            reasons = []
            if not trend_ok:
                reasons.append("200일선 아래")
            if not mom_ok:
                reasons.append("모멘텀 음수")
            note["status"] = "현금: " + ", ".join(reasons or ["변동성 산출 불가"])
        diagnostics[t] = note

    if not eligible:
        return weights, diagnostics

    # ── 2단계: 역변동성 가중 (변동성 큰 자산은 작게) ──
    vols = pd.Series({t: diagnostics[t]["vol"] for t in eligible})
    raw = 1.0 / vols
    raw = raw / raw.sum()

    # ── 3단계: 포트폴리오 변동성 타겟팅 ──
    # 암호화폐끼리는 상관계수가 높으므로 공분산으로 제대로 추정한다.
    rets = np.log(prices[eligible] / prices[eligible].shift(1)).tail(cfg.VOL_LOOKBACK_DAYS)
    cov = rets.dropna().cov() * TRADING_DAYS
    port_var = float(raw.values @ cov.values @ raw.values)
    port_vol = np.sqrt(port_var) if port_var > 0 else np.nan

    if np.isfinite(port_vol) and port_vol > 0:
        exposure = min(cfg.PORTFOLIO_VOL_TARGET / port_vol, cfg.MAX_TOTAL_EXPOSURE)
    else:
        exposure = cfg.MAX_TOTAL_EXPOSURE

    scaled = _apply_weight_cap(raw * exposure, cfg.MAX_ASSET_WEIGHT)
    weights.loc[eligible] = scaled.values

    for t in eligible:
        diagnostics[t]["weight"] = float(weights[t])
    diagnostics["_portfolio"] = {
        "estimated_vol": port_vol,
        "exposure": float(weights.sum()),
        "cash": float(1.0 - weights.sum()),
    }

    return weights, diagnostics
