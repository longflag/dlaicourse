"""포지션 사이징 및 리스크 관리 (Position Sizing & Risk Management)

수익률을 결정하는 것은 종목 선정보다 '얼마나 담느냐'와 '언제 멈추느냐'입니다.
이 모듈의 모든 규칙은 사전에 숫자로 정의되어 감정 개입 여지를 없앱니다.
"""

import numpy as np
import pandas as pd

from strategy import volatility


# ============================================================
# 비중 산출 방식
# ============================================================

def equal_weight(tickers: list) -> dict:
    """동일비중"""
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def inverse_volatility_weight(tickers: list, panel: pd.DataFrame,
                              window: int = 60) -> dict:
    """역변동성 비중: 변동성이 낮은 종목에 더 많이 배분.

    각 종목이 포트폴리오 전체 리스크에 기여하는 정도를 비슷하게 맞춰
    한 종목의 급락이 전체를 무너뜨리지 않게 합니다.
    """
    if not tickers:
        return {}

    inv_vols = {}
    for t in tickers:
        if t not in panel.columns:
            continue
        v = volatility(panel[t], window=window)
        if v is None or np.isnan(v) or v <= 0:
            v = 0.30  # 변동성 계산 불가 시 보수적 기본값
        inv_vols[t] = 1.0 / v

    total = sum(inv_vols.values())
    if total <= 0:
        return equal_weight(tickers)

    return {t: v / total for t, v in inv_vols.items()}


def apply_constraints(weights: dict, max_weight: float = 0.20,
                      cash_buffer: float = 0.0) -> dict:
    """비중 제약 적용.

    Args:
        max_weight: 종목당 최대 비중 (집중 위험 방지)
        cash_buffer: 항상 현금으로 남겨둘 비율
    """
    if not weights:
        return {}

    w = dict(weights)

    # 종목당 상한 적용 후 재정규화 (반복 수렴)
    for _ in range(10):
        total = sum(w.values())
        if total <= 0:
            return {}
        w = {t: v / total for t, v in w.items()}

        over = {t: v for t, v in w.items() if v > max_weight}
        if not over:
            break

        excess = sum(v - max_weight for v in over.values())
        under = {t: v for t, v in w.items() if v < max_weight}
        if not under:
            break

        under_total = sum(under.values())
        for t in w:
            if t in over:
                w[t] = max_weight
            elif under_total > 0:
                w[t] += excess * (w[t] / under_total)

    # 현금 버퍼 반영
    investable = 1.0 - cash_buffer
    total = sum(w.values())
    if total > 0:
        w = {t: v / total * investable for t, v in w.items()}

    return w


def size_positions(tickers: list, panel: pd.DataFrame,
                   scheme: str = "inverse_vol",
                   max_weight: float = 0.20,
                   cash_buffer: float = 0.0) -> dict:
    """선정된 종목에 대해 최종 비중을 산출합니다."""
    if scheme == "equal":
        raw = equal_weight(tickers)
    else:
        raw = inverse_volatility_weight(tickers, panel)

    return apply_constraints(raw, max_weight=max_weight,
                             cash_buffer=cash_buffer)


# ============================================================
# 리스크 관리 규칙
# ============================================================

class RiskManager:
    """사전 정의된 리스크 규칙을 집행합니다.

    규칙:
      1. 최대낙폭 서킷브레이커 - 낙폭이 한도를 넘으면 전액 현금화
      2. 재진입 조건 - 고점 대비 회복률이 기준을 넘어야 재투자
      3. 목표 변동성 - 포트폴리오 변동성이 목표를 넘으면 익스포저 축소

    이 규칙들은 '무서워서 팔았다'가 아니라 '규칙이라서 팔았다'로
    행동을 바꾸기 위한 장치입니다.
    """

    def __init__(self, max_drawdown_limit: float = 0.20,
                 recovery_threshold: float = 0.10,
                 target_volatility: float = None):
        """
        Args:
            max_drawdown_limit: 이 낙폭을 넘으면 현금화 (0.20 = -20%)
            recovery_threshold: 고점 대비 이 수준까지 회복하면 재진입
            target_volatility: 목표 연환산 변동성 (None이면 미적용)
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.recovery_threshold = recovery_threshold
        self.target_volatility = target_volatility

        self.peak_equity = None
        self.is_halted = False
        self.halt_count = 0

    def update(self, equity: float) -> bool:
        """자산 가치를 갱신하고 현재 투자 중단 여부를 반환합니다."""
        if self.peak_equity is None:
            self.peak_equity = equity

        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1

        if not self.is_halted:
            if drawdown <= -self.max_drawdown_limit:
                self.is_halted = True
                self.halt_count += 1
        else:
            # 고점 대비 recovery_threshold 이내로 회복하면 재진입
            if drawdown >= -self.recovery_threshold:
                self.is_halted = False

        return self.is_halted

    def scale_exposure(self, weights: dict, portfolio_vol: float) -> dict:
        """목표 변동성에 맞춰 전체 익스포저를 조절합니다."""
        if self.target_volatility is None or not weights:
            return weights
        if portfolio_vol is None or np.isnan(portfolio_vol) or portfolio_vol <= 0:
            return weights

        scale = min(1.0, self.target_volatility / portfolio_vol)
        return {t: w * scale for t, w in weights.items()}

    def reset(self):
        self.peak_equity = None
        self.is_halted = False
        self.halt_count = 0


def portfolio_volatility(weights: dict, panel: pd.DataFrame,
                         window: int = 60) -> float:
    """보유 비중 기준 포트폴리오의 최근 실현 변동성"""
    if not weights:
        return 0.0

    tickers = [t for t in weights if t in panel.columns]
    if not tickers:
        return 0.0

    returns = panel[tickers].pct_change().dropna().iloc[-window:]
    if len(returns) < 5:
        return np.nan

    w = np.array([weights[t] for t in tickers])
    port_returns = returns.values @ w
    return float(np.std(port_returns) * np.sqrt(252))
