"""기업 재무/실적 데이터 모듈 (Fundamental Analysis)

Yahoo Finance API로 실적 데이터를 가져오고, 실패 시 시뮬레이션 데이터를 생성합니다.
"""

import json
import hashlib
import urllib.request
import urllib.parse

import numpy as np


def _fetch_from_yahoo(ticker: str) -> dict | None:
    """Yahoo Finance quoteSummary API로 재무 데이터를 가져옵니다."""
    try:
        modules = "defaultKeyStatistics,financialData,earningsTrend"
        url = (
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
            f"{urllib.parse.quote(ticker)}?modules={modules}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        result = data["quoteSummary"]["result"][0]
        stats = result.get("defaultKeyStatistics", {})
        fin = result.get("financialData", {})

        def _val(d, key):
            v = d.get(key, {})
            if isinstance(v, dict):
                return v.get("raw")
            return v

        return {
            "per": _val(stats, "forwardPE") or _val(stats, "trailingPE"),
            "pbr": _val(stats, "priceToBook"),
            "eps_growth": (_val(stats, "earningsQuarterlyGrowth") or 0) * 100,
            "profit_margin": (_val(fin, "profitMargins") or 0) * 100,
            "revenue_growth": (_val(fin, "revenueGrowth") or 0) * 100,
            "roe": (_val(fin, "returnOnEquity") or 0) * 100,
            "debt_to_equity": _val(fin, "debtToEquity"),
            "dividend_yield": (_val(stats, "dividendYield") or 0) * 100,
            "current_ratio": _val(fin, "currentRatio"),
            "source": "yahoo",
        }
    except Exception:
        return None


def _generate_simulated(ticker: str) -> dict:
    """티커 기반 재현 가능한 시뮬레이션 재무 데이터 생성"""
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")

    # 섹터별 특성을 시드 기반으로 결정
    sector_type = rng.choice(["growth", "value", "stable", "turnaround"])

    if sector_type == "growth":
        per = rng.uniform(20, 80)
        eps_growth = rng.uniform(15, 60)
        revenue_growth = rng.uniform(10, 50)
        profit_margin = rng.uniform(5, 25)
        dividend_yield = rng.uniform(0, 1)
    elif sector_type == "value":
        per = rng.uniform(5, 15)
        eps_growth = rng.uniform(-5, 15)
        revenue_growth = rng.uniform(-3, 10)
        profit_margin = rng.uniform(8, 20)
        dividend_yield = rng.uniform(2, 6)
    elif sector_type == "stable":
        per = rng.uniform(12, 25)
        eps_growth = rng.uniform(3, 12)
        revenue_growth = rng.uniform(2, 8)
        profit_margin = rng.uniform(10, 30)
        dividend_yield = rng.uniform(1, 4)
    else:  # turnaround
        per = rng.uniform(30, 100) if rng.random() > 0.3 else rng.uniform(-50, -5)
        eps_growth = rng.uniform(-30, -5)
        revenue_growth = rng.uniform(-15, 5)
        profit_margin = rng.uniform(-5, 10)
        dividend_yield = rng.uniform(0, 1)

    return {
        "per": round(per, 1),
        "pbr": round(rng.uniform(0.5, 8.0), 2),
        "eps_growth": round(eps_growth, 1),
        "profit_margin": round(profit_margin, 1),
        "revenue_growth": round(revenue_growth, 1),
        "roe": round(rng.uniform(-5, 35), 1),
        "debt_to_equity": round(rng.uniform(10, 250), 1),
        "dividend_yield": round(dividend_yield, 2),
        "current_ratio": round(rng.uniform(0.5, 4.0), 2),
        "source": "simulation",
    }


def fetch_fundamentals(ticker: str) -> dict:
    """기업 재무 데이터를 가져옵니다. Yahoo Finance 실패 시 시뮬레이션 사용."""
    result = _fetch_from_yahoo(ticker)
    if result is not None:
        return result
    return _generate_simulated(ticker)


def evaluate_fundamentals(data: dict) -> dict:
    """재무 데이터를 평가하여 점수(0~100)와 판정을 반환합니다."""
    score = 50

    # PER 평가
    per = data.get("per")
    if per is not None and per > 0:
        if per < 10:
            score += 12  # 저평가
        elif per < 20:
            score += 8   # 적정~저평가
        elif per < 35:
            score += 0   # 적정
        elif per < 60:
            score -= 5   # 고평가
        else:
            score -= 10  # 심한 고평가
    elif per is not None and per < 0:
        score -= 8  # 적자

    # EPS 성장률
    eps_g = data.get("eps_growth", 0)
    if eps_g > 30:
        score += 12
    elif eps_g > 15:
        score += 8
    elif eps_g > 5:
        score += 4
    elif eps_g < -10:
        score -= 8
    elif eps_g < 0:
        score -= 4

    # 매출 성장률
    rev_g = data.get("revenue_growth", 0)
    if rev_g > 20:
        score += 8
    elif rev_g > 10:
        score += 5
    elif rev_g > 0:
        score += 2
    elif rev_g < -5:
        score -= 5

    # 이익률
    margin = data.get("profit_margin", 0)
    if margin > 20:
        score += 8
    elif margin > 10:
        score += 4
    elif margin < 0:
        score -= 8

    # ROE
    roe = data.get("roe", 0)
    if roe > 20:
        score += 5
    elif roe > 10:
        score += 3
    elif roe < 0:
        score -= 5

    # 부채비율
    de = data.get("debt_to_equity")
    if de is not None:
        if de < 50:
            score += 5   # 재무 안정
        elif de > 200:
            score -= 8   # 과다 부채

    score = max(0, min(100, score))

    # 판정
    if score >= 70:
        label = "우량"
    elif score >= 55:
        label = "양호"
    elif score >= 40:
        label = "보통"
    else:
        label = "주의"

    return {"score": score, "label": label}
