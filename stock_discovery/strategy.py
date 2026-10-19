"""규칙 기반 투자 전략 (Rule-Based Strategies)

모든 전략은 사전에 정의된 규칙만으로 목표 비중을 산출합니다.
재량적 판단(discretion), 뉴스, 감정은 일절 개입하지 않습니다.

백테스트 안전성 원칙:
  - 가격 기반 지표만 사용합니다. 재무/뉴스 데이터는 과거 시점 스냅샷을
    구할 수 없어 백테스트에 쓰면 미래참조(look-ahead) 편향이 생깁니다.
  - 신호는 의사결정 시점까지의 데이터로만 계산하고, 체결은 그 다음 날입니다.
"""

import numpy as np
import pandas as pd


# ============================================================
# 지표 헬퍼 (단일 종목 시계열 기준)
# ============================================================

def total_return(prices: pd.Series, lookback: int) -> float:
    """lookback 거래일 전 대비 수익률"""
    s = prices.dropna()
    if len(s) < lookback + 1:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-lookback - 1] - 1)


def momentum_12_1(prices: pd.Series) -> float:
    """12개월 모멘텀에서 최근 1개월을 제외한 값.

    최근 1개월은 단기 반전(short-term reversal) 효과가 강해 제외하는 것이
    학술적 표준입니다 (Jegadeesh & Titman).
    """
    s = prices.dropna()
    if len(s) < 253:
        return np.nan
    return float(s.iloc[-22] / s.iloc[-253] - 1)


def volatility(prices: pd.Series, window: int = 60) -> float:
    """연환산 변동성"""
    s = prices.dropna()
    if len(s) < window + 1:
        return np.nan
    r = s.pct_change().dropna().iloc[-window:]
    if len(r) < 2:
        return np.nan
    return float(r.std() * np.sqrt(252))


def above_ma(prices: pd.Series, window: int) -> bool:
    """현재가가 이동평균 위에 있는지 (추세 필터)"""
    s = prices.dropna()
    if len(s) < window:
        return False
    return bool(s.iloc[-1] > s.iloc[-window:].mean())


def max_drawdown(prices: pd.Series, window: int = 252) -> float:
    """최근 window 기간의 최대 낙폭"""
    s = prices.dropna().iloc[-window:]
    if len(s) < 2:
        return np.nan
    peak = s.cummax()
    return float((s / peak - 1).min())


# ============================================================
# 전략 베이스
# ============================================================

class Strategy:
    """모든 전략의 베이스 클래스.

    generate_weights()는 의사결정 시점까지의 가격 패널만 받아
    목표 비중 dict를 반환합니다. 합이 1.0 미만이면 나머지는 현금입니다.
    """

    name = "base"
    description = ""
    warmup_days = 252  # 신호 계산에 필요한 최소 거래일

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


# ============================================================
# 1. 절대 + 상대 듀얼 모멘텀 (Gary Antonacci)
# ============================================================

class DualMomentum(Strategy):
    """듀얼 모멘텀: 상대 모멘텀으로 종목을 고르고, 절대 모멘텀으로 하락장을 회피.

    규칙:
      1. 12-1개월 모멘텀 상위 top_n 종목을 선정 (상대 모멘텀)
      2. 그중 모멘텀이 0보다 큰 종목만 보유 (절대 모멘텀)
      3. 조건을 만족하는 종목이 없으면 전액 현금

    하락장에서 현금으로 빠지는 규칙이 최대낙폭을 크게 줄여줍니다.
    """

    name = "듀얼모멘텀"
    description = "상대 모멘텀 종목선정 + 절대 모멘텀 하락장 회피"
    warmup_days = 260

    def __init__(self, top_n: int = 10, min_momentum: float = 0.0):
        self.top_n = top_n
        self.min_momentum = min_momentum

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        scores = {}
        for ticker in panel.columns:
            m = momentum_12_1(panel[ticker])
            if not np.isnan(m):
                scores[ticker] = m

        if not scores:
            return {}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, m in ranked[:self.top_n] if m > self.min_momentum]

        if not selected:
            return {}  # 전액 현금

        w = 1.0 / len(selected)
        return {t: w for t in selected}


# ============================================================
# 2. 추세추종 (Meb Faber)
# ============================================================

class TrendFollowing(Strategy):
    """추세추종: 200일 이동평균 위에 있는 종목만 보유.

    규칙:
      1. 현재가 > 200일 이동평균인 종목만 후보
      2. 후보 중 모멘텀 상위 top_n 보유
      3. 후보가 없으면 전액 현금

    "추세를 거스르지 않는다"는 원칙을 기계적으로 구현한 것으로,
    가장 단순하면서도 낙폭 방어력이 검증된 규칙입니다.
    """

    name = "추세추종"
    description = "200일 이평선 위 종목만 보유, 아니면 현금"
    warmup_days = 260

    def __init__(self, top_n: int = 10, ma_window: int = 200):
        self.top_n = top_n
        self.ma_window = ma_window

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        candidates = {}
        for ticker in panel.columns:
            s = panel[ticker]
            if above_ma(s, self.ma_window):
                m = total_return(s, 126)  # 6개월 모멘텀
                if not np.isnan(m):
                    candidates[ticker] = m

        if not candidates:
            return {}

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:self.top_n]]
        w = 1.0 / len(selected)
        return {t: w for t in selected}


# ============================================================
# 3. 멀티팩터 (모멘텀 + 저변동성 + 추세)
# ============================================================

class MultiFactor(Strategy):
    """멀티팩터: 여러 검증된 팩터의 순위를 합성해 종목을 선정.

    규칙:
      1. 모멘텀(12-1) 순위 - 높을수록 좋음
      2. 변동성 순위 - 낮을수록 좋음 (저변동성 이상현상)
      3. 최대낙폭 순위 - 작을수록 좋음 (하방 안정성)
      4. 합성 순위 상위 top_n 보유, 단 추세 필터를 통과한 종목만

    단일 지표에 의존하지 않아 특정 국면에 대한 과적합 위험이 낮습니다.
    """

    name = "멀티팩터"
    description = "모멘텀+저변동성+하방안정성 합성 순위"
    warmup_days = 260

    def __init__(self, top_n: int = 10, use_trend_filter: bool = True,
                 w_momentum: float = 0.5, w_volatility: float = 0.25,
                 w_drawdown: float = 0.25):
        self.top_n = top_n
        self.use_trend_filter = use_trend_filter
        self.w_momentum = w_momentum
        self.w_volatility = w_volatility
        self.w_drawdown = w_drawdown

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        rows = []
        for ticker in panel.columns:
            s = panel[ticker]
            mom = momentum_12_1(s)
            vol = volatility(s)
            mdd = max_drawdown(s)
            if np.isnan(mom) or np.isnan(vol) or np.isnan(mdd):
                continue
            if self.use_trend_filter and not above_ma(s, 200):
                continue
            rows.append({"ticker": ticker, "mom": mom, "vol": vol, "mdd": mdd})

        if not rows:
            return {}

        f = pd.DataFrame(rows).set_index("ticker")

        # 순위화: 모멘텀은 클수록, 변동성/낙폭은 작을수록 좋음
        r_mom = f["mom"].rank(ascending=False)
        r_vol = f["vol"].rank(ascending=True)
        r_mdd = f["mdd"].rank(ascending=False)  # mdd는 음수이므로 0에 가까울수록 좋음

        composite = (self.w_momentum * r_mom
                     + self.w_volatility * r_vol
                     + self.w_drawdown * r_mdd)

        selected = composite.nsmallest(self.top_n).index.tolist()
        if not selected:
            return {}

        w = 1.0 / len(selected)
        return {t: w for t in selected}


# ============================================================
# 4. 동일비중 매수보유 (벤치마크)
# ============================================================

class EqualWeightBuyHold(Strategy):
    """전 종목 동일비중 매수 후 보유. 다른 전략의 기준선입니다."""

    name = "동일비중보유"
    description = "전 종목 동일비중 (벤치마크)"
    warmup_days = 21

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        valid = [t for t in panel.columns if panel[t].dropna().shape[0] > 0]
        if not valid:
            return {}
        w = 1.0 / len(valid)
        return {t: w for t in valid}


# ============================================================
# 5. 저변동성 (Low Volatility Anomaly)
# ============================================================

class LowVolatility(Strategy):
    """변동성이 가장 낮은 종목을 보유.

    저변동성 주식이 고변동성 주식보다 위험조정 수익이 높다는
    'low volatility anomaly'를 규칙화한 전략입니다.
    감정적으로는 지루하지만, 장기적으로 검증된 팩터입니다.
    """

    name = "저변동성"
    description = "변동성 하위 종목 보유 (저변동성 이상현상)"
    warmup_days = 130

    def __init__(self, top_n: int = 10, use_trend_filter: bool = True):
        self.top_n = top_n
        self.use_trend_filter = use_trend_filter

    def generate_weights(self, panel: pd.DataFrame) -> dict:
        vols = {}
        for ticker in panel.columns:
            s = panel[ticker]
            if self.use_trend_filter and not above_ma(s, 200):
                continue
            v = volatility(s, window=120)
            if not np.isnan(v) and v > 0:
                vols[ticker] = v

        if not vols:
            return {}

        ranked = sorted(vols.items(), key=lambda x: x[1])
        selected = [t for t, _ in ranked[:self.top_n]]
        w = 1.0 / len(selected)
        return {t: w for t in selected}


# 사용 가능한 전략 목록
STRATEGIES = {
    "dual_momentum": DualMomentum,
    "trend": TrendFollowing,
    "multifactor": MultiFactor,
    "lowvol": LowVolatility,
    "buyhold": EqualWeightBuyHold,
}
