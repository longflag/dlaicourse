"""뉴스 감성 분석 모듈 (News Sentiment Analysis)

뉴스 헤드라인의 감성을 분석하되, 다음을 고려합니다:
- 출처 신뢰도: 공시/실적 vs 루머/추측/광고성 기사 구분
- 뉴스 과열 감지: 긍정 뉴스가 몰리면 오히려 주의 (피크 신호)
- 역발상(Contrarian): "소문에 사서 뉴스에 팔아라" 로직
- 감성 신뢰도: 불확실한 뉴스는 가중치 자체를 낮춤
"""

import json
import hashlib
import urllib.request
import urllib.parse

import numpy as np


# ============================================================
# 감성 키워드 사전 (가중치 포함)
# ============================================================

# 신뢰도 높은 긍정 키워드 (공시, 실적 기반 팩트)
STRONG_POSITIVE = {
    # 영문 - 실적/공시 기반
    "beat", "beats", "exceeds", "record earnings", "profit",
    "dividend", "approval", "fda approved", "buyback",
    "revenue growth", "outperform",
    # 한글 - 공시/실적 기반
    "호실적", "흑자", "영업이익", "순이익", "배당", "자사주매입",
    "수주", "계약체결", "실적개선", "흑자전환", "공시",
}

# 일반 긍정 키워드
MILD_POSITIVE = {
    # 영문
    "surge", "surges", "surging", "soar", "rally", "rallies",
    "gains", "gain", "bullish", "strong", "boom", "breakout",
    "recovery", "raise", "launch", "innovation", "partnership",
    "deal", "expansion", "upgrade", "upgrades", "growth",
    # 한글
    "급등", "상승", "상한가", "신고가", "성장", "매출증가",
    "목표가상향", "매수", "호재", "인수", "합병",
    "턴어라운드", "반등", "돌파", "호조",
}

# 신뢰도 높은 부정 키워드 (팩트 기반)
STRONG_NEGATIVE = {
    # 영문
    "bankruptcy", "fraud", "sec investigation", "default",
    "recall", "lawsuit", "layoffs", "restatement",
    "miss", "misses", "loss", "losses", "downgrade",
    # 한글
    "적자", "부도", "사기", "횡령", "분식회계", "상장폐지",
    "리콜", "소송", "감원", "구조조정", "하향", "손실",
    "목표가하향", "매도",
}

# 일반 부정 키워드
MILD_NEGATIVE = {
    # 영문
    "crash", "plunge", "drop", "drops", "decline", "declining",
    "bearish", "sell", "risk", "warning", "debt", "weak",
    "underperform", "slump", "crisis", "cut", "cuts", "fine",
    "penalty", "investigation",
    # 한글
    "급락", "하락", "하한가", "감소", "부진", "악재",
    "벌금", "제재", "하락세", "약세", "폭락", "위기", "조사",
}

# ============================================================
# 불확실/추측성 키워드 → 신뢰도 감점
# ============================================================
UNRELIABLE_KEYWORDS = {
    # 영문 - 추측/루머/의견
    "rumor", "rumour", "rumors", "rumours", "could", "might",
    "may", "possibly", "speculate", "speculation", "unconfirmed",
    "sources say", "allegedly", "according to sources",
    "opinion", "analyst says", "predicted", "expected",
    "potential", "if", "whether",
    # 한글 - 추측/루머
    "루머", "소문", "추정", "전망", "예상", "가능성", "관측",
    "것으로 보인다", "것으로 알려", "소식통", "미확인",
    "~할 수도", "~될 듯", "~할 것", "의견", "전문가",
    "찌라시", "카더라", "~설",
}

# 광고성/홍보성 키워드 → 신뢰도 대폭 감점
PROMOTIONAL_KEYWORDS = {
    # 영문
    "sponsored", "advertisement", "promoted", "paid",
    "must buy", "hot stock", "guaranteed", "moon",
    "to the moon", "millionaire", "100x", "10x",
    "don't miss", "act now", "limited time",
    # 한글
    "광고", "협찬", "홍보", "추천종목", "급등예상",
    "대박", "폭등예감", "지금이 기회", "놓치지마",
    "수익보장", "무료리딩",
}


# ============================================================
# 시뮬레이션용 헤드라인 템플릿 (출처/신뢰도 포함)
# ============================================================
_FACTUAL_POSITIVE = [
    "[공시] {name}, 분기 영업이익 전년비 25% 증가",
    "[실적] {name} Q{q} 매출·이익 컨센서스 상회",
    "[공시] {name}, 자사주 매입 프로그램 발표",
    "[IR] {name} announces record quarterly revenue",
    "[공시] {name}, 대규모 수주 계약 체결",
]

_SPECULATIVE_POSITIVE = [
    "{name}, 신사업 진출 기대감에 강세 전망 - 증권사 의견",
    "전문가 \"{name} 목표주가 상향 가능성\" 전망",
    "소식통 \"{name} 인수합병 추진 중\" 루머",
    "Sources say {name} could announce major partnership",
    "{name} 관련주 급등... 테마 수혜 기대",
]

_FACTUAL_NEGATIVE = [
    "[공시] {name}, 분기 실적 시장 예상 하회",
    "[공시] {name} 구조조정 계획 공식 발표",
    "[뉴스] {name} 제품 리콜 결정",
    "[공시] {name} reports quarterly loss, cuts guidance",
    "[뉴스] {name} faces regulatory investigation",
]

_SPECULATIVE_NEGATIVE = [
    "{name} 업황 둔화 우려 확대 - 애널리스트 의견",
    "전문가 \"{name} 추가 하락 가능성\" 전망",
    "{name} 경쟁 심화 관측... 시장점유율 하락 추정",
    "Analysts speculate {name} may cut dividend",
    "소식통 \"{name} 대규모 감원 가능성\"",
]

_NEUTRAL_TEMPLATES = [
    "[공시] {name}, 정기 주주총회 개최 예정",
    "{name} 신제품 라인업 공개",
    "{name} 인사 변동 발표",
    "{name} announces new product lineup for {year}",
    "{name} CEO discusses company strategy at conference",
]


def _calc_headline_credibility(text: str) -> float:
    """개별 헤드라인의 신뢰도를 계산합니다 (0.0 ~ 1.0).

    공시/실적 기반 → 높은 신뢰도
    루머/추측성 → 낮은 신뢰도
    광고/홍보성 → 매우 낮은 신뢰도
    """
    text_lower = text.lower()
    credibility = 0.7  # 기본 신뢰도

    # 공시/실적 태그가 있으면 높은 신뢰도
    official_tags = ["[공시]", "[실적]", "[ir]", "[sec filing]",
                     "earnings report", "quarterly results", "공시"]
    if any(tag in text_lower for tag in official_tags):
        credibility = 1.0

    # 불확실/추측성 키워드가 있으면 감점
    unreliable_count = sum(1 for kw in UNRELIABLE_KEYWORDS if kw in text_lower)
    credibility -= unreliable_count * 0.15

    # 광고/홍보성이면 대폭 감점
    promo_count = sum(1 for kw in PROMOTIONAL_KEYWORDS if kw in text_lower)
    if promo_count > 0:
        credibility -= promo_count * 0.3

    return max(0.1, min(1.0, credibility))


def _analyze_text_sentiment(text: str) -> tuple[float, float]:
    """텍스트의 감성 점수(-1.0~+1.0)와 신뢰도(0.0~1.0)를 반환합니다."""
    text_lower = text.lower()
    words = set(text_lower.split())

    # 가중치 적용: 강한 키워드 = 2점, 약한 키워드 = 1점
    pos_score = sum(2 for kw in STRONG_POSITIVE if kw in words or kw in text_lower)
    pos_score += sum(1 for kw in MILD_POSITIVE if kw in words or kw in text_lower)
    neg_score = sum(2 for kw in STRONG_NEGATIVE if kw in words or kw in text_lower)
    neg_score += sum(1 for kw in MILD_NEGATIVE if kw in words or kw in text_lower)

    total = pos_score + neg_score
    if total == 0:
        sentiment = 0.0
    else:
        sentiment = (pos_score - neg_score) / total

    credibility = _calc_headline_credibility(text)

    return sentiment, credibility


def _detect_news_overheating(sentiments: list, credibilities: list) -> dict:
    """뉴스 과열 상태를 감지합니다.

    "소문에 사서 뉴스에 팔아라" 원칙 적용:
    - 긍정 뉴스가 과도하게 집중 → 피크 가능성 (contrarian 경고)
    - 부정 뉴스가 과도하게 집중 → 바닥 가능성 (contrarian 기회)
    """
    if not sentiments:
        return {"is_overheated": False, "contrarian_signal": None, "adjustment": 0}

    avg_sent = np.mean(sentiments)
    pos_ratio = sum(1 for s in sentiments if s > 0.2) / len(sentiments)
    neg_ratio = sum(1 for s in sentiments if s < -0.2) / len(sentiments)
    avg_cred = np.mean(credibilities) if credibilities else 0.7

    result = {
        "is_overheated": False,
        "contrarian_signal": None,
        "adjustment": 0,
        "reason": "",
    }

    # 긍정 뉴스 과열: 80% 이상이 긍정이고 평균 감성이 매우 높음
    if pos_ratio >= 0.8 and avg_sent > 0.5:
        result["is_overheated"] = True
        result["contrarian_signal"] = "sell"
        result["adjustment"] = -20  # 점수 하향 조정
        result["reason"] = "긍정 뉴스 과열 (피크 주의)"

    # 부정 뉴스 과열: 80% 이상이 부정이고 평균 감성이 매우 낮음
    elif neg_ratio >= 0.8 and avg_sent < -0.5:
        result["is_overheated"] = True
        result["contrarian_signal"] = "buy"
        result["adjustment"] = +15  # 점수 상향 조정 (역발상 매수)
        result["reason"] = "부정 뉴스 과집중 (바닥 반전 가능)"

    # 신뢰도가 전반적으로 낮은 경우
    if avg_cred < 0.4:
        result["adjustment"] = int(result["adjustment"] * 0.5)  # 조정 효과도 반감
        result["reason"] += " [낮은 신뢰도로 효과 축소]" if result["reason"] else "뉴스 신뢰도 낮음"

    return result


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
        results = [_analyze_text_sentiment(h) for h in headlines]
        sentiments = [r[0] for r in results]
        credibilities = [r[1] for r in results]

        # 신뢰도 가중 평균 감성
        weighted_sum = sum(s * c for s, c in zip(sentiments, credibilities))
        weight_total = sum(credibilities)
        avg_sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
        avg_credibility = np.mean(credibilities)

        # 과열 감지
        overheating = _detect_news_overheating(sentiments, credibilities)

        return {
            "sentiment_score": round(float(avg_sentiment), 3),
            "credibility": round(float(avg_credibility), 2),
            "headline_count": len(headlines),
            "top_headlines": headlines[:3],
            "overheating": overheating,
            "source": "yahoo",
        }
    except Exception:
        return None


def _generate_simulated(ticker: str, name: str) -> dict:
    """시뮬레이션 뉴스 감성 데이터 생성 (신뢰도/과열 포함)"""
    seed = int(hashlib.md5(f"news_{ticker}".encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    # 시뮬레이션 뉴스 유형 결정
    news_scenario = rng.choice([
        "normal", "positive_overheated", "negative_overheated",
        "mixed", "low_credibility",
    ], p=[0.4, 0.15, 0.15, 0.2, 0.1])

    headlines = []
    sentiments = []
    credibilities = []
    headline_count = rng.randint(3, 10)

    for i in range(min(5, headline_count)):
        if news_scenario == "positive_overheated":
            if rng.random() < 0.7:
                tmpl = rng.choice(_SPECULATIVE_POSITIVE)
                cred = rng.uniform(0.3, 0.6)
            else:
                tmpl = rng.choice(_FACTUAL_POSITIVE)
                cred = rng.uniform(0.8, 1.0)
            sent = rng.uniform(0.3, 0.8)

        elif news_scenario == "negative_overheated":
            if rng.random() < 0.7:
                tmpl = rng.choice(_SPECULATIVE_NEGATIVE)
                cred = rng.uniform(0.3, 0.6)
            else:
                tmpl = rng.choice(_FACTUAL_NEGATIVE)
                cred = rng.uniform(0.8, 1.0)
            sent = rng.uniform(-0.8, -0.3)

        elif news_scenario == "low_credibility":
            tmpl = rng.choice(_SPECULATIVE_POSITIVE + _SPECULATIVE_NEGATIVE)
            cred = rng.uniform(0.1, 0.4)
            sent = rng.uniform(-0.5, 0.5)

        elif news_scenario == "mixed":
            all_templates = (_FACTUAL_POSITIVE + _SPECULATIVE_POSITIVE +
                             _FACTUAL_NEGATIVE + _SPECULATIVE_NEGATIVE +
                             _NEUTRAL_TEMPLATES)
            tmpl = rng.choice(all_templates)
            cred = rng.uniform(0.4, 0.9)
            sent = rng.uniform(-0.5, 0.5)

        else:  # normal
            r = rng.random()
            if r < 0.35:
                tmpl = rng.choice(_FACTUAL_POSITIVE)
                cred = rng.uniform(0.7, 1.0)
                sent = rng.uniform(0.2, 0.6)
            elif r < 0.6:
                tmpl = rng.choice(_FACTUAL_NEGATIVE)
                cred = rng.uniform(0.7, 1.0)
                sent = rng.uniform(-0.6, -0.2)
            elif r < 0.8:
                tmpl = rng.choice(_NEUTRAL_TEMPLATES)
                cred = rng.uniform(0.6, 0.9)
                sent = rng.uniform(-0.1, 0.1)
            else:
                tmpl = rng.choice(_SPECULATIVE_POSITIVE + _SPECULATIVE_NEGATIVE)
                cred = rng.uniform(0.3, 0.6)
                sent = rng.uniform(-0.4, 0.4)

        headline = tmpl.format(name=name, q=rng.randint(1, 5), year=2025 + rng.randint(0, 2))
        headlines.append(headline)
        sentiments.append(sent)
        credibilities.append(cred)

    # 신뢰도 가중 평균
    if credibilities:
        weighted_sum = sum(s * c for s, c in zip(sentiments, credibilities))
        weight_total = sum(credibilities)
        avg_sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
        avg_credibility = float(np.mean(credibilities))
    else:
        avg_sentiment = 0.0
        avg_credibility = 0.5

    overheating = _detect_news_overheating(sentiments, credibilities)

    return {
        "sentiment_score": round(float(avg_sentiment), 3),
        "credibility": round(avg_credibility, 2),
        "headline_count": headline_count,
        "top_headlines": headlines[:3],
        "overheating": overheating,
        "source": "simulation",
    }


def fetch_news_sentiment(ticker: str, name: str) -> dict:
    """뉴스 감성 분석 데이터를 가져옵니다."""
    result = _fetch_from_yahoo(ticker)
    if result is not None:
        return result
    return _generate_simulated(ticker, name)


def evaluate_sentiment(data: dict) -> dict:
    """감성 데이터를 종합 평가하여 점수(0~100)와 라벨을 반환합니다.

    신뢰도가 낮은 뉴스는 감성 점수의 영향력을 줄이고,
    과열 상태가 감지되면 역발상(contrarian) 조정을 적용합니다.
    """
    score_raw = data.get("sentiment_score", 0)
    credibility = data.get("credibility", 0.7)
    overheating = data.get("overheating", {})

    # 기본 점수: 감성 * 신뢰도 가중
    # 신뢰도가 낮으면 감성 효과가 줄어듦 (중립 50점 쪽으로 수렴)
    adjusted_sentiment = score_raw * credibility
    score = int(50 + adjusted_sentiment * 50)

    # 과열 조정 (contrarian)
    if overheating.get("is_overheated"):
        score += overheating.get("adjustment", 0)

    score = max(0, min(100, score))

    # 라벨 결정 (과열 상태 반영)
    if overheating.get("contrarian_signal") == "sell":
        label = "과열주의"
    elif overheating.get("contrarian_signal") == "buy":
        label = "역발상매수"
    elif credibility < 0.4:
        label = "신뢰도낮음"
    elif adjusted_sentiment > 0.3:
        label = "긍정"
    elif adjusted_sentiment > 0.1:
        label = "약긍정"
    elif adjusted_sentiment > -0.1:
        label = "중립"
    elif adjusted_sentiment > -0.3:
        label = "약부정"
    else:
        label = "부정"

    return {
        "score": score,
        "label": label,
        "credibility": credibility,
        "overheating_reason": overheating.get("reason", ""),
    }
