"""뉴스 감성 분석 모듈 (News Sentiment Analysis)

Yahoo Finance에서 뉴스 헤드라인을 가져와 키워드 기반 감성 분석을 수행합니다.
외부 NLP 라이브러리 없이 내장 키워드 사전으로 동작합니다.
"""

import json
import hashlib
import urllib.request
import urllib.parse

import numpy as np


# 감성 분석 키워드 사전
POSITIVE_KEYWORDS = {
    # 영문
    "surge", "surges", "surging", "soar", "soars", "rally", "rallies",
    "beat", "beats", "exceeds", "record", "high", "upgrade", "upgrades",
    "growth", "profit", "gains", "gain", "bullish", "optimistic",
    "strong", "outperform", "buy", "boom", "breakout", "recovery",
    "raise", "raises", "dividend", "approval", "launch", "innovation",
    "partnership", "deal", "expansion", "revenue",
    # 한글
    "급등", "상승", "상한가", "신고가", "호실적", "흑자", "성장",
    "매출증가", "영업이익", "순이익", "목표가상향", "매수",
    "호재", "수주", "계약", "인수", "합병", "배당", "실적개선",
    "턴어라운드", "반등", "돌파", "기대", "긍정", "호조",
}

NEGATIVE_KEYWORDS = {
    # 영문
    "crash", "crashes", "plunge", "plunges", "drop", "drops",
    "miss", "misses", "loss", "losses", "downgrade", "downgrades",
    "lawsuit", "recall", "cut", "cuts", "layoff", "layoffs",
    "bearish", "sell", "decline", "declining", "risk", "warning",
    "debt", "bankruptcy", "fraud", "investigation", "fine", "penalty",
    "weak", "underperform", "slump", "crisis", "default",
    # 한글
    "급락", "하락", "하한가", "적자", "감소", "부진", "손실",
    "목표가하향", "매도", "악재", "소송", "리콜", "감원",
    "구조조정", "하향", "위기", "부도", "사기", "조사",
    "벌금", "제재", "하락세", "약세", "폭락",
}

# 시뮬레이션용 헤드라인 템플릿
_POSITIVE_TEMPLATES = [
    "{name}, 분기 실적 시장 예상 상회",
    "{name} 신규 사업 진출로 성장 기대감",
    "{name} 목표주가 상향 조정 - 증권사 리포트",
    "글로벌 수요 확대로 {name} 수혜 전망",
    "{name} 전략적 파트너십 체결 발표",
    "{name} reports strong quarterly earnings beat",
    "Analysts upgrade {name} on growth outlook",
    "{name} announces strategic partnership deal",
]

_NEGATIVE_TEMPLATES = [
    "{name}, 실적 부진으로 주가 압박",
    "{name} 업황 둔화 우려 확대",
    "{name} 비용 증가로 이익률 하락 전망",
    "경쟁 심화로 {name} 시장점유율 하락",
    "{name} 구조조정 계획 발표",
    "{name} misses revenue expectations in Q{q}",
    "Concerns grow over {name} debt levels",
    "{name} faces regulatory investigation",
]

_NEUTRAL_TEMPLATES = [
    "{name}, 정기 주주총회 개최 예정",
    "{name} 신제품 라인업 공개",
    "{name} 인사 변동 발표",
    "{name} announces new product lineup for {year}",
    "{name} CEO discusses company strategy at conference",
]


def _analyze_text_sentiment(text: str) -> float:
    """텍스트의 감성 점수를 계산합니다 (-1.0 ~ +1.0)"""
    text_lower = text.lower()
    words = set(text_lower.split())

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in words or kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in words or kw in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def _fetch_from_yahoo(ticker: str) -> dict | None:
    """Yahoo Finance에서 뉴스 헤드라인을 가져옵니다."""
    try:
        url = (
            f"https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={urllib.parse.quote(ticker)}&newsCount=10&quotesCount=0"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        news = data.get("news", [])
        if not news:
            return None

        headlines = [item.get("title", "") for item in news[:10]]
        sentiments = [_analyze_text_sentiment(h) for h in headlines]

        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        return {
            "sentiment_score": round(float(avg_sentiment), 3),
            "headline_count": len(headlines),
            "top_headlines": headlines[:3],
            "source": "yahoo",
        }
    except Exception:
        return None


def _generate_simulated(ticker: str, name: str) -> dict:
    """시뮬레이션 뉴스 감성 데이터 생성"""
    seed = int(hashlib.md5(f"news_{ticker}".encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    sentiment_score = rng.uniform(-0.7, 0.7)
    headline_count = rng.randint(3, 10)

    headlines = []
    for _ in range(3):
        r = rng.random()
        if r < 0.4:
            tmpl = rng.choice(_POSITIVE_TEMPLATES)
        elif r < 0.7:
            tmpl = rng.choice(_NEGATIVE_TEMPLATES)
        else:
            tmpl = rng.choice(_NEUTRAL_TEMPLATES)
        headlines.append(tmpl.format(
            name=name, q=rng.randint(1, 5), year=2025 + rng.randint(0, 2)
        ))

    return {
        "sentiment_score": round(float(sentiment_score), 3),
        "headline_count": headline_count,
        "top_headlines": headlines,
        "source": "simulation",
    }


def fetch_news_sentiment(ticker: str, name: str) -> dict:
    """뉴스 감성 분석 데이터를 가져옵니다."""
    result = _fetch_from_yahoo(ticker)
    if result is not None:
        return result
    return _generate_simulated(ticker, name)


def evaluate_sentiment(data: dict) -> dict:
    """감성 데이터를 평가하여 점수(0~100)와 라벨을 반환합니다."""
    score_raw = data.get("sentiment_score", 0)

    # -1~+1 을 0~100으로 매핑
    score = int(50 + score_raw * 50)

    # 헤드라인 수가 많고 긍정적이면 보너스
    if data.get("headline_count", 0) >= 5 and score_raw > 0.3:
        score += 5

    score = max(0, min(100, score))

    if score_raw > 0.3:
        label = "긍정"
    elif score_raw > 0.1:
        label = "약긍정"
    elif score_raw > -0.1:
        label = "중립"
    elif score_raw > -0.3:
        label = "약부정"
    else:
        label = "부정"

    return {"score": score, "label": label}
