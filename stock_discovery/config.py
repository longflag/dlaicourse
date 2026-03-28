"""주식종목 발굴 프로그램 설정"""

# 분석 대상 종목 리스트 (KOSPI/KOSDAQ 대표 종목 + 미국 주요 종목)
# 한국 종목은 티커 뒤에 .KS (KOSPI) 또는 .KQ (KOSDAQ) 를 붙입니다.
KOREAN_STOCKS = {
    # KOSPI 대표 종목
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "006400.KS": "삼성SDI",
    "051910.KS": "LG화학",
    "003670.KS": "포스코퓨처엠",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "096770.KS": "SK이노베이션",
    "034730.KS": "SK",
    "028260.KS": "삼성물산",
    "012330.KS": "현대모비스",
    "066570.KS": "LG전자",
    "003550.KS": "LG",
    "032830.KS": "삼성생명",
    "015760.KS": "한국전력",
    "033780.KS": "KT&G",
    # KOSDAQ 대표 종목
    "247540.KQ": "에코프로비엠",
    "086520.KQ": "에코프로",
    "377300.KQ": "카카오페이",
    "263750.KQ": "펄어비스",
    "293490.KQ": "카카오게임즈",
}

US_STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "MA": "Mastercard",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "BAC": "Bank of America",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "INTC": "Intel",
}

# 기술적 분석 기본 설정
DEFAULT_PERIOD = "6mo"  # 데이터 조회 기간
RSI_PERIOD = 14
RSI_OVERSOLD = 30       # RSI 과매도 기준
RSI_OVERBOUGHT = 70     # RSI 과매수 기준
MA_SHORT = 20           # 단기 이동평균
MA_LONG = 60            # 장기 이동평균
VOLUME_SURGE_RATIO = 2.0  # 거래량 급증 배수 (평균 대비)
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
