"""주식종목 발굴 프로그램 설정"""

# 분석 대상 종목 리스트 (KOSPI/KOSDAQ 대표 종목 + 미국 주요 종목)
# 한국 종목은 티커 뒤에 .KS (KOSPI) 또는 .KQ (KOSDAQ) 를 붙입니다.
KOREAN_STOCKS = {
    # KOSPI 시가총액 상위 종목
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
    # KOSPI 추가 종목
    "005490.KS": "POSCO홀딩스",
    "000810.KS": "삼성화재",
    "086790.KS": "하나금융지주",
    "316140.KS": "우리금융지주",
    "010130.KS": "고려아연",
    "009150.KS": "삼성전기",
    "018260.KS": "삼성에스디에스",
    "010950.KS": "S-Oil",
    "011170.KS": "롯데케미칼",
    "051900.KS": "LG생활건강",
    "030200.KS": "KT",
    "017670.KS": "SK텔레콤",
    "036570.KS": "엔씨소프트",
    "011200.KS": "HMM",
    "034020.KS": "두산에너빌리티",
    "003490.KS": "대한항공",
    "004020.KS": "현대제철",
    "010140.KS": "한솔제지",
    "000100.KS": "유한양행",
    "068270.KS": "셀트리온",
    "207940.KS": "삼성바이오로직스",
    "326030.KS": "SK바이오팜",
    "180640.KS": "한진칼",
    "267250.KS": "HD현대",
    "329180.KS": "HD현대중공업",
    "042700.KS": "한미반도체",
    "352820.KS": "하이브",
    "259960.KS": "크래프톤",
    "373220.KS": "LG에너지솔루션",
    "011790.KS": "SKC",
    # KOSDAQ 대표 종목
    "247540.KQ": "에코프로비엠",
    "086520.KQ": "에코프로",
    "377300.KQ": "카카오페이",
    "263750.KQ": "펄어비스",
    "293490.KQ": "카카오게임즈",
    "035900.KQ": "JYP Ent.",
    "041510.KQ": "에스엠",
    "112040.KQ": "위메이드",
    "253450.KQ": "스튜디오드래곤",
    "145020.KQ": "휴젤",
    "196170.KQ": "알테오젠",
    "328130.KQ": "루닛",
    "322000.KQ": "HD현대에너지솔루션",
    "058470.KQ": "리노공업",
    "403870.KQ": "HPSP",
    "357780.KQ": "솔브레인",
    "039030.KQ": "이오테크닉스",
    "095340.KQ": "ISC",
    "298380.KQ": "에이비엘바이오",
    "006580.KQ": "대양전기공업",
    "036930.KQ": "주성엔지니어링",
}

US_STOCKS = {
    # 빅테크 / Magnificent 7
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    # 반도체
    "AMD": "AMD",
    "INTC": "Intel",
    "AVGO": "Broadcom",
    "QCOM": "Qualcomm",
    "MU": "Micron",
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA Corp",
    "MRVL": "Marvell",
    "ARM": "ARM Holdings",
    "TSM": "TSMC",
    # 소프트웨어 / 클라우드
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "ORCL": "Oracle",
    "NOW": "ServiceNow",
    "SNOW": "Snowflake",
    "PLTR": "Palantir",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike",
    "DDOG": "Datadog",
    "NET": "Cloudflare",
    # AI / 로보틱스
    "AI": "C3.ai",
    "PATH": "UiPath",
    "SMCI": "Super Micro Computer",
    # 금융
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "V": "Visa",
    "MA": "Mastercard",
    "AXP": "American Express",
    # 헬스케어
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
    "LLY": "Eli Lilly",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "BMY": "Bristol-Myers Squibb",
    "AMGN": "Amgen",
    # 소비재
    "WMT": "Walmart",
    "COST": "Costco",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "MCD": "McDonald's",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "HD": "Home Depot",
    "LOW": "Lowe's",
    # 산업재 / 에너지
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "GE": "GE Aerospace",
    "LMT": "Lockheed Martin",
    "RTX": "RTX Corp",
    # 미디어 / 엔터
    "NFLX": "Netflix",
    "DIS": "Disney",
    "SPOT": "Spotify",
    "RBLX": "Roblox",
    # ETF (시장 전체 흐름 참고)
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ 100 ETF",
    "SOXX": "반도체 ETF",
    "ARKK": "ARK Innovation ETF",
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
