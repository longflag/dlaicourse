# 주식종목 발굴 프로그램 (Stock Discovery Program)

기술적 분석 지표를 활용하여 한국(KOSPI/KOSDAQ) 및 미국 주식 시장에서 매수/매도 유망 종목을 자동으로 발굴하는 프로그램입니다.

## 주요 기능

### 기술적 분석 지표
- **RSI** (Relative Strength Index): 과매수/과매도 판단
- **이동평균선** (MA5, MA20, MA60, MA120): 추세 판단
- **볼린저밴드**: 가격 밴드 이탈 감지
- **MACD**: 추세 전환 감지
- **스토캐스틱**: 모멘텀 반전 감지
- **거래량 분석**: 거래량 급증 감지

### 스크리닝 전략

**매수 신호:**
- 골든크로스 (MA20이 MA60 상향돌파)
- RSI 과매도 (RSI < 30)
- 거래량 급증 (20일 평균 대비 2배 이상)
- 볼린저밴드 하단 이탈
- MACD 골든크로스
- 상승 추세 (가격 > MA20 > MA60)
- 스토캐스틱 과매도 반전

**매도/주의 신호:**
- 데드크로스 (MA20이 MA60 하향돌파)
- RSI 과매수 (RSI > 70)
- 볼린저밴드 상단 돌파

## 설치

```bash
cd stock_discovery
pip install -r requirements.txt
```

## 사용법

```bash
# 전체 종목 분석 (한국 + 미국)
python main.py

# 한국 종목만 분석
python main.py --market kr

# 미국 종목만 분석
python main.py --market us

# 특정 종목 상세 분석
python main.py --ticker 005930.KS    # 삼성전자
python main.py --ticker AAPL          # Apple

# 매수 신호 종목만 필터링
python main.py --strategy buy

# 상위 10개 종목만 표시
python main.py --top 10

# 1년치 데이터 기반 분석
python main.py --period 1y
```

## 종합 점수 기준

- 기본 점수 50점에서 시작
- 매수 신호 하나당 +8점
- 매도 신호 하나당 -10점
- 이동평균 정배열 보너스 +10점
- RSI/MACD/거래량 등 추가 가감점
- 최종 점수 범위: 0~100점

## 주의사항

- 본 프로그램은 투자 참고용이며, 투자 판단의 책임은 본인에게 있습니다.
- 기술적 분석만으로는 완전한 투자 판단이 어렵습니다.
- 실시간 데이터가 아닌 일봉 기준 분석입니다.
