"""백테스트 엔진 (Backtesting Engine)

규칙이 과거에 어떻게 작동했는지 검증합니다.
검증되지 않은 규칙은 규칙이 아니라 그냥 느낌입니다.

미래참조(look-ahead) 방지 설계:
  - 신호는 T일 종가까지의 데이터로만 계산합니다.
  - 체결은 T+1일 종가에 이루어집니다.
  ("장 마감 후 프로그램을 돌리고, 다음 날 주문을 낸다"는 실제 운용 흐름)

거래비용:
  - 회전율에 비례하여 수수료+슬리피지를 차감합니다.
"""

import sys

import numpy as np
import pandas as pd

from data_fetcher import fetch_stock_data
from portfolio import size_positions, RiskManager, portfolio_volatility


# ============================================================
# 가격 패널 구성
# ============================================================

def build_price_panel(stocks: dict, period: str = "5y",
                      verbose: bool = True) -> pd.DataFrame:
    """여러 종목의 종가를 하나의 DataFrame으로 결합합니다."""
    series = {}
    total = len(stocks)

    for i, (ticker, name) in enumerate(stocks.items(), 1):
        if verbose:
            sys.stdout.write(f"\r  데이터 수집 [{i}/{total}] {name}...")
            sys.stdout.flush()
        df = fetch_stock_data(ticker, period)
        if df is not None and len(df) > 0:
            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series[ticker] = s

    if verbose:
        print(f"\r  데이터 수집 완료: {len(series)}/{total}개 종목{' ' * 20}")

    if not series:
        return pd.DataFrame()

    panel = pd.DataFrame(series).sort_index()
    # 결측치는 직전 값으로 채우되, 상장 전 구간은 NaN 유지
    panel = panel.ffill()
    return panel


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> list:
    """리밸런싱 날짜(각 기간의 마지막 거래일)를 반환합니다.

    freq: 'M'=월간, 'Q'=분기, 'W'=주간, 'Y'=연간
    리밸런싱 주기가 길수록 거래비용과 감정 개입이 줄어듭니다.
    """
    def period_key(d):
        if freq == "W":
            iso = d.isocalendar()
            return (iso[0], iso[1])
        if freq == "Q":
            return (d.year, (d.month - 1) // 3)
        if freq == "Y":
            return (d.year,)
        return (d.year, d.month)

    # 각 기간의 마지막 거래일만 남깁니다.
    last_of_period = {}
    for d in index:
        last_of_period[period_key(d)] = d

    return sorted(last_of_period.values())


# ============================================================
# 백테스트 엔진
# ============================================================

class Backtester:
    """규칙 기반 전략의 과거 성과를 시뮬레이션합니다."""

    def __init__(self, panel: pd.DataFrame,
                 rebalance_freq: str = "M",
                 cost_bps: float = 25.0,
                 weight_scheme: str = "inverse_vol",
                 max_weight: float = 0.20,
                 risk_manager: RiskManager = None,
                 initial_capital: float = 10_000_000):
        """
        Args:
            panel: 종가 패널 (index=날짜, columns=티커)
            rebalance_freq: 리밸런싱 주기 (M/Q/W/Y)
            cost_bps: 편도 거래비용 (25bps = 0.25%, 수수료+세금+슬리피지)
            weight_scheme: 비중 산출 방식 (equal / inverse_vol)
            max_weight: 종목당 최대 비중
            risk_manager: 리스크 관리 규칙 (None이면 미적용)
            initial_capital: 초기 자본
        """
        self.panel = panel
        self.rebalance_freq = rebalance_freq
        self.cost_bps = cost_bps
        self.weight_scheme = weight_scheme
        self.max_weight = max_weight
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital

    def run(self, strategy) -> dict:
        """백테스트를 실행하고 결과를 반환합니다."""
        panel = self.panel
        if panel.empty:
            return {}

        warmup = strategy.warmup_days
        if len(panel) <= warmup + 20:
            return {}

        dates = panel.index
        rebal_dates = set(get_rebalance_dates(dates, self.rebalance_freq))

        if self.risk_manager:
            self.risk_manager.reset()

        cash = float(self.initial_capital)
        shares = {}           # ticker -> 보유 수량
        pending_target = None  # 다음 거래일에 체결할 목표 비중

        equity_curve = []
        weight_history = []
        turnover_history = []
        halt_days = 0
        total_cost = 0.0

        for i, date in enumerate(dates):
            prices = panel.loc[date]

            # --- 1) 전일 신호가 있으면 오늘 종가로 체결 ---
            if pending_target is not None:
                equity = cash + sum(
                    sh * prices.get(t, np.nan) for t, sh in shares.items()
                    if not np.isnan(prices.get(t, np.nan))
                )

                new_shares = {}
                for t, w in pending_target.items():
                    p = prices.get(t, np.nan)
                    if np.isnan(p) or p <= 0:
                        continue
                    new_shares[t] = (equity * w) / p

                # 회전율과 거래비용 계산
                traded_value = 0.0
                all_t = set(shares) | set(new_shares)
                for t in all_t:
                    p = prices.get(t, np.nan)
                    if np.isnan(p):
                        continue
                    traded_value += abs(new_shares.get(t, 0) - shares.get(t, 0)) * p

                cost = traded_value * (self.cost_bps / 10_000)
                total_cost += cost
                turnover_history.append(traded_value / equity if equity > 0 else 0)

                shares = new_shares
                invested = sum(
                    sh * prices.get(t, 0) for t, sh in shares.items()
                    if not np.isnan(prices.get(t, np.nan))
                )
                cash = equity - invested - cost
                pending_target = None

            # --- 2) 시가평가 ---
            equity = cash + sum(
                sh * prices.get(t, np.nan) for t, sh in shares.items()
                if not np.isnan(prices.get(t, np.nan))
            )
            equity_curve.append(equity)

            # --- 3) 워밍업 이전에는 신호를 내지 않음 ---
            if i < warmup:
                continue

            # --- 4) 리밸런싱일이면 신호 계산 (오늘까지의 데이터만 사용) ---
            if date in rebal_dates and i < len(dates) - 1:
                hist = panel.iloc[:i + 1]

                halted = False
                if self.risk_manager:
                    halted = self.risk_manager.update(equity)

                if halted:
                    pending_target = {}  # 전액 현금
                    halt_days += 1
                    weight_history.append((date, {}))
                    continue

                raw_weights = strategy.generate_weights(hist)
                selected = list(raw_weights.keys())

                if not selected:
                    pending_target = {}
                    weight_history.append((date, {}))
                    continue

                target = size_positions(
                    selected, hist,
                    scheme=self.weight_scheme,
                    max_weight=self.max_weight,
                )

                # 목표 변동성 조절
                if self.risk_manager and self.risk_manager.target_volatility:
                    pvol = portfolio_volatility(target, hist)
                    target = self.risk_manager.scale_exposure(target, pvol)

                pending_target = target
                weight_history.append((date, target))

        equity_series = pd.Series(equity_curve, index=dates)

        return {
            "strategy": strategy.name,
            "equity": equity_series,
            "weights": weight_history,
            "avg_turnover": float(np.mean(turnover_history)) if turnover_history else 0.0,
            "total_cost": total_cost,
            "halt_count": halt_days,
            "rebalance_count": len(weight_history),
            "warmup": warmup,
        }


# ============================================================
# 성과 지표
# ============================================================

def calc_metrics(equity: pd.Series, warmup: int = 0,
                 risk_free: float = 0.03) -> dict:
    """백테스트 성과 지표를 계산합니다."""
    e = equity.iloc[warmup:] if warmup > 0 else equity
    e = e.dropna()

    if len(e) < 2 or e.iloc[0] <= 0:
        return {}

    returns = e.pct_change().dropna()
    n_days = len(e)
    years = n_days / 252

    total_return = e.iloc[-1] / e.iloc[0] - 1
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free) / vol if vol > 0 else 0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
    sortino = (cagr - risk_free) / downside_vol if downside_vol > 0 else 0

    peak = e.cummax()
    dd = e / peak - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else 0

    # 월간 승률
    monthly = e.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0

    return {
        "총수익률": total_return,
        "CAGR": cagr,
        "변동성": vol,
        "샤프지수": sharpe,
        "소르티노": sortino,
        "최대낙폭": mdd,
        "칼마지수": calmar,
        "월간승률": win_rate,
        "기간(년)": years,
    }


def benchmark_equity(panel: pd.DataFrame, ticker: str,
                     initial_capital: float = 10_000_000) -> pd.Series:
    """단일 종목/지수 매수보유 기준선"""
    if ticker not in panel.columns:
        return pd.Series(dtype=float)
    s = panel[ticker].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return initial_capital * s / s.iloc[0]


def format_metrics_row(name: str, m: dict) -> dict:
    """지표를 표 출력용으로 포맷합니다."""
    if not m:
        return {"전략": name}
    return {
        "전략": name,
        "CAGR": f"{m['CAGR']:.1%}",
        "변동성": f"{m['변동성']:.1%}",
        "샤프": f"{m['샤프지수']:.2f}",
        "최대낙폭": f"{m['최대낙폭']:.1%}",
        "칼마": f"{m['칼마지수']:.2f}",
        "월승률": f"{m['월간승률']:.0%}",
        "총수익": f"{m['총수익률']:.1%}",
    }
