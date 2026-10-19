#!/usr/bin/env python3
"""
합성 데이터로 전략/백테스트 엔진을 검증한다 (네트워크 불필요).

규칙이나 코드를 수정했다면 실제 돈을 넣기 전에 반드시 이걸 먼저 통과시킬 것.
    python test_engine.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from systematic import backtest, config, strategy
from systematic.schedule import is_rebalance_date, next_rebalance_date, rebalance_dates

rng = np.random.default_rng(42)
dates = pd.date_range("2017-01-01", "2026-09-01", freq="D")
n = len(dates)


def gbm(mu, sigma, n, start=100.0):
    shocks = rng.normal(mu / 365, sigma / np.sqrt(365), n)
    # 사이클성 부여: 강세/약세 구간 교대
    cycle = np.sin(np.arange(n) / 365 * 2 * np.pi / 1.5) * 0.0015
    return start * np.exp(np.cumsum(shocks + cycle))


prices = pd.DataFrame({
    "BTC": gbm(0.35, 0.60, n, 1000),
    "ETH": gbm(0.30, 0.80, n, 50),
    "XRP": gbm(0.05, 0.90, n, 0.5),
    "SOL": gbm(0.40, 1.10, n, 2),
}, index=dates)
# SOL은 2020-08 상장이라고 가정 -> 이전 구간 NaN
prices.loc[prices.index < "2020-08-01", "SOL"] = np.nan

print("=" * 60)
print("1. 스케줄 로직")
print("=" * 60)
assert is_rebalance_date("2026-03-01") is True
assert is_rebalance_date("2026-03-02") is False
nxt = next_rebalance_date("2026-03-02")
print(f"  2026-03-02 다음 리밸런싱 -> {nxt.date()}")
assert nxt == pd.Timestamp("2026-04-01")
rb = rebalance_dates(dates)
print(f"  전체 기간 리밸런싱일 수: {len(rb)} (예상 ~117)")
assert 110 <= len(rb) <= 120
print("  OK")

print()
print("=" * 60)
print("2. 지표 함수")
print("=" * 60)
s = prices["BTC"].dropna()
print(f"  연환산 변동성: {strategy.annualized_vol(s):.1%}")
print(f"  모멘텀 스코어: {strategy.momentum_score(s):+.1%}")
print(f"  추세 필터 통과: {strategy.passes_trend_filter(s)}")
assert np.isfinite(strategy.annualized_vol(s))
assert np.isfinite(strategy.momentum_score(s))
print("  OK")

print()
print("=" * 60)
print("3. 목표 비중 산출")
print("=" * 60)
for asof in ["2019-06-01", "2021-01-01", "2023-06-01", "2026-09-01"]:
    w, diag = strategy.target_weights(prices.loc[:asof])
    held = {k: f"{v:.1%}" for k, v in w.items() if v > 0}
    print(f"  {asof}: 투자 {w.sum():.0%} / 현금 {1-w.sum():.0%}  {held}")
    assert w.sum() <= config.MAX_TOTAL_EXPOSURE + 1e-9, "레버리지 발생!"
    assert (w >= -1e-12).all(), "음수 비중 발생!"
    assert (w <= config.MAX_ASSET_WEIGHT + 1e-9).all(), "종목 상한 초과!"
    # 상장 전 자산은 비중 0이어야 함
    if asof < "2020-08-01":
        assert w["SOL"] == 0.0, "상장 전 자산에 비중 배정됨!"
print("  OK (레버리지/음수/상한/상장전 모두 통과)")

print()
print("=" * 60)
print("4. 백테스트 엔진")
print("=" * 60)
res = backtest.run(prices, config, initial=10000.0)
eq = res["equity"]
print(f"  자산곡선 길이: {len(eq)}, 최종: ${eq.iloc[-1]:,.0f}")
print(f"  거래 횟수: {len(res['trades'])}")
print(f"  평균 투자비중: {res['exposure'].mean():.1%}")
assert len(eq) == len(prices)
assert (eq > 0).all(), "자산이 0 이하로 감!"
assert res["exposure"].max() <= 1.0 + 1e-9, "투자비중 100% 초과!"

m = backtest.metrics(eq, res["exposure"], res["trades"])
for k, v in m.items():
    print(f"    {k}: {v:,.3f}")
assert np.isfinite(m["CAGR"]) and np.isfinite(m["샤프"])
assert m["최대낙폭(MDD)"] <= 0
print("  OK")

print()
print("=" * 60)
print("5. 미래참조(look-ahead) 검증")
print("=" * 60)
# 데이터를 뒤쪽만 잘라내도 앞부분 자산곡선은 100% 동일해야 한다.
cut = prices.loc[:"2024-01-01"]
res_cut = backtest.run(cut, config, initial=10000.0)
common = res_cut["equity"].index
diff = (res["equity"].loc[common] - res_cut["equity"]).abs().max()
print(f"  전체 데이터 vs 잘린 데이터, 공통구간 최대 차이: {diff:.10f}")
assert diff < 1e-6, "미래 데이터가 과거 결정에 영향을 미침!"
print("  OK (미래참조 없음)")

print()
print("=" * 60)
print("6. 벤치마크")
print("=" * 60)
bh = backtest.buy_and_hold(prices, "BTC", 10000)
ew = backtest.equal_weight_hold(prices, config, 10000)
dca_v, dca_p = backtest.monthly_dca(prices, config, 500)
print(f"  BTC 보유 최종: ${bh.iloc[-1]:,.0f}  MDD {backtest.metrics(bh)['최대낙폭(MDD)']:.1%}")
print(f"  동일비중 최종: ${ew.iloc[-1]:,.0f}  MDD {backtest.metrics(ew)['최대낙폭(MDD)']:.1%}")
print(f"  DCA 투입 ${dca_p.iloc[-1]:,.0f} -> ${dca_v.iloc[-1]:,.0f}")
assert (ew > 0).all() and (bh > 0).all()
print("  OK")

print()
print("=" * 60)
print("7. 밴드가 거래를 실제로 억제하는가")
print("=" * 60)
orig = config.REBALANCE_BAND
config.REBALANCE_BAND = 0.0
res_nb = backtest.run(prices, config, initial=10000.0)
config.REBALANCE_BAND = orig
print(f"  밴드 5%: 거래 {len(res['trades'])}회, 비용 ${res['trades']['cost'].sum():,.0f}")
print(f"  밴드 0%: 거래 {len(res_nb['trades'])}회, 비용 ${res_nb['trades']['cost'].sum():,.0f}")
assert len(res["trades"]) < len(res_nb["trades"]), "밴드가 거래를 줄이지 못함!"
print("  OK")

print()
print("모든 검증 통과")
