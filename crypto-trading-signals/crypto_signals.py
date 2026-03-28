#!/usr/bin/env python3
"""
암호화폐 매매 시그널 분석기
Bitcoin, Ethereum, Ripple, Solana의 매수/매도 시점을 기술적 분석과
종합 시황을 기반으로 단기/중기/장기 추세를 분석합니다.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
import json
import sys
import time

console = Console()

# CoinGecko API (무료, 키 불필요)
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COINS = {
    "bitcoin": {"symbol": "BTC", "name": "비트코인"},
    "ethereum": {"symbol": "ETH", "name": "이더리움"},
    "ripple": {"symbol": "XRP", "name": "리플"},
    "solana": {"symbol": "SOL", "name": "솔라나"},
}

# 기간 설정 (일 수)
PERIODS = {
    "short": {"days": 30, "label": "단기 (30일)", "ma_fast": 5, "ma_slow": 10},
    "mid": {"days": 90, "label": "중기 (90일)", "ma_fast": 20, "ma_slow": 50},
    "long": {"days": 365, "label": "장기 (365일)", "ma_fast": 50, "ma_slow": 200},
}


def api_request_with_retry(url, params=None, max_retries=5):
    """Rate limit 대응 재시도 로직이 포함된 API 요청"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80초
                console.print(f"  [yellow]API 제한 감지, {wait}초 대기 중... (시도 {attempt+1}/{max_retries})[/yellow]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                console.print(f"  [yellow]요청 실패, {wait}초 후 재시도... ({e})[/yellow]")
                time.sleep(wait)
            else:
                console.print(f"  [red]요청 최종 실패: {e}[/red]")
                return None
    return None


def fetch_market_data(coin_id, days):
    """CoinGecko에서 시장 데이터를 가져옵니다."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data = api_request_with_retry(url, params)
    if data is None:
        return None
    try:
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        if "total_volumes" in data:
            vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
            vol_df.set_index("timestamp", inplace=True)
            df = df.join(vol_df, how="left")

        return df
    except (KeyError, ValueError) as e:
        console.print(f"[red]데이터 파싱 실패 ({coin_id}): {e}[/red]")
        return None


def fetch_current_prices():
    """현재 가격 및 시장 데이터를 가져옵니다."""
    ids = ",".join(COINS.keys())
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ids,
        "order": "market_cap_desc",
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d,30d",
    }
    data = api_request_with_retry(url, params)
    if data is None:
        return {}
    return {item["id"]: item for item in data}


def fetch_global_data():
    """글로벌 시장 데이터를 가져옵니다."""
    url = f"{COINGECKO_BASE}/global"
    data = api_request_with_retry(url)
    if data is None:
        return {}
    return data.get("data", {})


def fetch_fear_greed_index():
    """공포-탐욕 지수를 가져옵니다."""
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [{}])[0]
        return {
            "value": int(data.get("value", 50)),
            "classification": data.get("value_classification", "N/A"),
        }
    except Exception:
        return {"value": 50, "classification": "N/A"}


# ─── 기술적 지표 계산 ───


def calc_rsi(prices, period=14):
    """RSI (상대강도지수) 계산"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(prices, period=20, std_dev=2):
    """볼린저 밴드 계산"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calc_stochastic(prices, k_period=14, d_period=3):
    """스토캐스틱 오실레이터 계산"""
    low_min = prices.rolling(window=k_period).min()
    high_max = prices.rolling(window=k_period).max()
    denom = high_max - low_min
    k = 100 * (prices - low_min) / denom.replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def calc_obv(prices, volumes):
    """OBV (거래량 균형) 계산"""
    if volumes is None or volumes.empty:
        return pd.Series(dtype=float)
    direction = np.sign(prices.diff())
    obv = (direction * volumes).cumsum()
    return obv


def calc_adx(prices, period=14):
    """ADX (평균방향지수) - 가격만으로 근사 계산"""
    diff = prices.diff()
    plus_dm = diff.where(diff > 0, 0.0)
    minus_dm = (-diff).where(diff < 0, 0.0)
    atr = prices.diff().abs().rolling(window=period).mean()
    atr = atr.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.rolling(window=period).mean()
    return adx


# ─── 시그널 분석 ───


def analyze_signals(df, period_config):
    """종합 기술적 분석을 수행하고 시그널을 생성합니다."""
    prices = df["price"]
    volumes = df.get("volume")

    if len(prices) < 30:
        return {"signal": "데이터 부족", "score": 0, "details": []}

    ma_fast = period_config["ma_fast"]
    ma_slow = period_config["ma_slow"]

    signals = []
    score = 0  # -100 (강력 매도) ~ +100 (강력 매수)

    current_price = prices.iloc[-1]

    # 1) 이동평균선 분석 (가중치: 25)
    if len(prices) >= ma_slow:
        sma_fast = prices.rolling(window=ma_fast).mean().iloc[-1]
        sma_slow = prices.rolling(window=ma_slow).mean().iloc[-1]
        ema_fast = prices.ewm(span=ma_fast, adjust=False).mean().iloc[-1]

        if sma_fast > sma_slow:
            ma_score = 15
            if current_price > sma_fast:
                ma_score = 25
                signals.append(("이동평균선", "강력 매수", f"단기MA({ma_fast}) > 장기MA({ma_slow}), 가격 > 단기MA"))
            else:
                signals.append(("이동평균선", "매수", f"단기MA({ma_fast}) > 장기MA({ma_slow})"))
        else:
            ma_score = -15
            if current_price < sma_fast:
                ma_score = -25
                signals.append(("이동평균선", "강력 매도", f"단기MA({ma_fast}) < 장기MA({ma_slow}), 가격 < 단기MA"))
            else:
                signals.append(("이동평균선", "매도", f"단기MA({ma_fast}) < 장기MA({ma_slow})"))
        score += ma_score
    else:
        fast_available = min(ma_fast, len(prices) - 1)
        if fast_available >= 3:
            sma = prices.rolling(window=fast_available).mean().iloc[-1]
            if current_price > sma:
                score += 10
                signals.append(("이동평균선", "매수", f"가격 > MA({fast_available})"))
            else:
                score -= 10
                signals.append(("이동평균선", "매도", f"가격 < MA({fast_available})"))

    # 2) RSI 분석 (가중치: 20)
    rsi = calc_rsi(prices)
    if not rsi.empty and not np.isnan(rsi.iloc[-1]):
        rsi_val = rsi.iloc[-1]
        if rsi_val < 30:
            rsi_score = 20
            signals.append(("RSI", "강력 매수 (과매도)", f"RSI = {rsi_val:.1f}"))
        elif rsi_val < 40:
            rsi_score = 10
            signals.append(("RSI", "매수", f"RSI = {rsi_val:.1f}"))
        elif rsi_val > 70:
            rsi_score = -20
            signals.append(("RSI", "강력 매도 (과매수)", f"RSI = {rsi_val:.1f}"))
        elif rsi_val > 60:
            rsi_score = -10
            signals.append(("RSI", "매도", f"RSI = {rsi_val:.1f}"))
        else:
            rsi_score = 0
            signals.append(("RSI", "중립", f"RSI = {rsi_val:.1f}"))
        score += rsi_score

    # 3) MACD 분석 (가중치: 20)
    macd_line, signal_line, histogram = calc_macd(prices)
    if not histogram.empty and len(histogram) >= 2:
        h_curr = histogram.iloc[-1]
        h_prev = histogram.iloc[-2]
        if not (np.isnan(h_curr) or np.isnan(h_prev)):
            if h_curr > 0 and h_prev <= 0:
                macd_score = 20
                signals.append(("MACD", "강력 매수 (골든크로스)", "MACD 히스토그램 양전환"))
            elif h_curr > 0 and h_curr > h_prev:
                macd_score = 10
                signals.append(("MACD", "매수", "MACD 히스토그램 상승 중"))
            elif h_curr < 0 and h_prev >= 0:
                macd_score = -20
                signals.append(("MACD", "강력 매도 (데드크로스)", "MACD 히스토그램 음전환"))
            elif h_curr < 0 and h_curr < h_prev:
                macd_score = -10
                signals.append(("MACD", "매도", "MACD 히스토그램 하락 중"))
            else:
                macd_score = 0
                signals.append(("MACD", "중립", f"히스토그램 = {h_curr:.2f}"))
            score += macd_score

    # 4) 볼린저 밴드 분석 (가중치: 15)
    upper, mid, lower = calc_bollinger(prices)
    if not upper.empty and not np.isnan(upper.iloc[-1]):
        bb_upper = upper.iloc[-1]
        bb_lower = lower.iloc[-1]
        bb_mid = mid.iloc[-1]
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (current_price - bb_lower) / bb_range
        else:
            bb_position = 0.5

        if current_price <= bb_lower:
            bb_score = 15
            signals.append(("볼린저밴드", "강력 매수", f"가격이 하단밴드 도달 (위치: {bb_position:.1%})"))
        elif bb_position < 0.3:
            bb_score = 8
            signals.append(("볼린저밴드", "매수", f"가격이 하단밴드 근접 (위치: {bb_position:.1%})"))
        elif current_price >= bb_upper:
            bb_score = -15
            signals.append(("볼린저밴드", "강력 매도", f"가격이 상단밴드 도달 (위치: {bb_position:.1%})"))
        elif bb_position > 0.7:
            bb_score = -8
            signals.append(("볼린저밴드", "매도", f"가격이 상단밴드 근접 (위치: {bb_position:.1%})"))
        else:
            bb_score = 0
            signals.append(("볼린저밴드", "중립", f"밴드 중앙 부근 (위치: {bb_position:.1%})"))
        score += bb_score

    # 5) 스토캐스틱 분석 (가중치: 10)
    k, d = calc_stochastic(prices)
    if not k.empty and not np.isnan(k.iloc[-1]):
        k_val = k.iloc[-1]
        d_val = d.iloc[-1] if not np.isnan(d.iloc[-1]) else k_val
        if k_val < 20 and k_val > d_val:
            stoch_score = 10
            signals.append(("스토캐스틱", "강력 매수", f"%K={k_val:.1f}, %D={d_val:.1f} (과매도 반등)"))
        elif k_val < 30:
            stoch_score = 5
            signals.append(("스토캐스틱", "매수", f"%K={k_val:.1f} (과매도 구간)"))
        elif k_val > 80 and k_val < d_val:
            stoch_score = -10
            signals.append(("스토캐스틱", "강력 매도", f"%K={k_val:.1f}, %D={d_val:.1f} (과매수 하락)"))
        elif k_val > 70:
            stoch_score = -5
            signals.append(("스토캐스틱", "매도", f"%K={k_val:.1f} (과매수 구간)"))
        else:
            stoch_score = 0
            signals.append(("스토캐스틱", "중립", f"%K={k_val:.1f}, %D={d_val:.1f}"))
        score += stoch_score

    # 6) 거래량 분석 (가중치: 10)
    if volumes is not None and not volumes.empty and len(volumes) >= 10:
        vol_avg = volumes.rolling(window=10).mean().iloc[-1]
        vol_current = volumes.iloc[-1]
        if vol_avg > 0:
            vol_ratio = vol_current / vol_avg
            price_change = prices.iloc[-1] / prices.iloc[-2] - 1 if len(prices) >= 2 else 0

            if vol_ratio > 1.5 and price_change > 0:
                vol_score = 10
                signals.append(("거래량", "매수", f"거래량 급증 + 가격 상승 (거래량 비율: {vol_ratio:.1f}x)"))
            elif vol_ratio > 1.5 and price_change < 0:
                vol_score = -10
                signals.append(("거래량", "매도", f"거래량 급증 + 가격 하락 (거래량 비율: {vol_ratio:.1f}x)"))
            elif vol_ratio < 0.5:
                vol_score = 0
                signals.append(("거래량", "주의", f"거래량 감소 (거래량 비율: {vol_ratio:.1f}x)"))
            else:
                vol_score = 0
                signals.append(("거래량", "중립", f"거래량 보통 (거래량 비율: {vol_ratio:.1f}x)"))
            score += vol_score

    # 최종 시그널 결정
    if score >= 50:
        final_signal = "강력 매수"
    elif score >= 25:
        final_signal = "매수"
    elif score >= 10:
        final_signal = "약한 매수"
    elif score <= -50:
        final_signal = "강력 매도"
    elif score <= -25:
        final_signal = "매도"
    elif score <= -10:
        final_signal = "약한 매도"
    else:
        final_signal = "관망 (중립)"

    return {"signal": final_signal, "score": score, "details": signals}


def get_signal_color(signal):
    """시그널에 따른 색상 반환"""
    if "강력 매수" in signal:
        return "bold green"
    elif "매수" in signal:
        return "green"
    elif "강력 매도" in signal:
        return "bold red"
    elif "매도" in signal:
        return "red"
    elif "주의" in signal:
        return "yellow"
    return "white"


def display_market_overview(current_data, global_data, fng):
    """시장 종합 현황을 표시합니다."""
    console.print()
    console.rule("[bold cyan]암호화폐 종합 매매 시그널 분석기[/bold cyan]")
    console.print(f"[dim]분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print()

    # 글로벌 시장 현황
    global_table = Table(title="글로벌 시장 현황", box=box.ROUNDED)
    global_table.add_column("항목", style="cyan")
    global_table.add_column("값", justify="right")

    if global_data:
        total_mcap = global_data.get("total_market_cap", {}).get("usd", 0)
        total_vol = global_data.get("total_volume", {}).get("usd", 0)
        btc_dom = global_data.get("market_cap_percentage", {}).get("btc", 0)
        mcap_change = global_data.get("market_cap_change_percentage_24h_usd", 0)

        global_table.add_row("총 시가총액", f"${total_mcap/1e12:.2f}T")
        global_table.add_row("24시간 거래량", f"${total_vol/1e9:.1f}B")
        global_table.add_row("BTC 도미넌스", f"{btc_dom:.1f}%")
        change_color = "green" if mcap_change >= 0 else "red"
        global_table.add_row("24h 시총 변화", f"[{change_color}]{mcap_change:+.2f}%[/{change_color}]")

    # 공포-탐욕 지수
    fng_val = fng["value"]
    if fng_val <= 25:
        fng_color = "red"
        fng_comment = "극도의 공포 -> 매수 기회 가능"
    elif fng_val <= 45:
        fng_color = "yellow"
        fng_comment = "공포 -> 신중한 매수 고려"
    elif fng_val <= 55:
        fng_color = "white"
        fng_comment = "중립"
    elif fng_val <= 75:
        fng_color = "green"
        fng_comment = "탐욕 -> 신중한 매도 고려"
    else:
        fng_color = "bold red"
        fng_comment = "극도의 탐욕 -> 매도 기회 가능"

    global_table.add_row(
        "공포-탐욕 지수",
        f"[{fng_color}]{fng_val} ({fng['classification']})[/{fng_color}]",
    )
    global_table.add_row("시장 심리 해석", f"[{fng_color}]{fng_comment}[/{fng_color}]")

    console.print(global_table)
    console.print()

    # 현재 가격 테이블
    price_table = Table(title="현재 시세", box=box.ROUNDED)
    price_table.add_column("코인", style="cyan")
    price_table.add_column("현재가", justify="right")
    price_table.add_column("1시간", justify="right")
    price_table.add_column("24시간", justify="right")
    price_table.add_column("7일", justify="right")
    price_table.add_column("30일", justify="right")
    price_table.add_column("시가총액", justify="right")

    for coin_id, info in COINS.items():
        data = current_data.get(coin_id, {})
        if not data:
            continue
        price = data.get("current_price", 0)

        def fmt_pct(val):
            if val is None:
                return "[dim]N/A[/dim]"
            color = "green" if val >= 0 else "red"
            return f"[{color}]{val:+.2f}%[/{color}]"

        h1 = data.get("price_change_percentage_1h_in_currency")
        h24 = data.get("price_change_percentage_24h_in_currency")
        d7 = data.get("price_change_percentage_7d_in_currency")
        d30 = data.get("price_change_percentage_30d_in_currency")
        mcap = data.get("market_cap", 0)

        price_table.add_row(
            f"{info['name']} ({info['symbol']})",
            f"${price:,.2f}" if price >= 1 else f"${price:,.4f}",
            fmt_pct(h1),
            fmt_pct(h24),
            fmt_pct(d7),
            fmt_pct(d30),
            f"${mcap/1e9:.1f}B",
        )

    console.print(price_table)
    console.print()


def display_coin_analysis(coin_id, info, current_data, all_results):
    """개별 코인 분석 결과를 표시합니다."""
    coin_data = current_data.get(coin_id, {})
    price = coin_data.get("current_price", 0)
    price_str = f"${price:,.2f}" if price >= 1 else f"${price:,.4f}"

    console.rule(f"[bold yellow]{info['name']} ({info['symbol']}) - {price_str}[/bold yellow]")

    for period_key, period_config in PERIODS.items():
        result = all_results.get(f"{coin_id}_{period_key}")
        if not result:
            continue

        signal = result["signal"]
        score_val = result["score"]
        details = result["details"]

        signal_color = get_signal_color(signal)

        # 점수 바 생성
        bar_len = 40
        normalized = (score_val + 100) / 200  # 0~1
        filled = int(normalized * bar_len)
        bar = ""
        for i in range(bar_len):
            if i < filled:
                if i < bar_len * 0.3:
                    bar += "[red]\u2588[/red]"
                elif i < bar_len * 0.7:
                    bar += "[yellow]\u2588[/yellow]"
                else:
                    bar += "[green]\u2588[/green]"
            else:
                bar += "[dim]\u2591[/dim]"

        console.print(f"\n  [{signal_color}]{period_config['label']}[/{signal_color}]")
        console.print(f"  시그널: [{signal_color}]{signal}[/{signal_color}] (점수: {score_val:+d}/100)")
        console.print(f"  {bar}")

        # 세부 지표
        detail_table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        detail_table.add_column("지표", style="cyan", width=14)
        detail_table.add_column("판정", width=18)
        detail_table.add_column("상세", style="dim")

        for indicator, verdict, desc in details:
            v_color = get_signal_color(verdict)
            detail_table.add_row(indicator, f"[{v_color}]{verdict}[/{v_color}]", desc)

        console.print(detail_table)


def generate_strategy(all_results, fng, global_data):
    """종합 투자 전략을 생성합니다."""
    console.print()
    console.rule("[bold magenta]종합 투자 전략 리포트[/bold magenta]")
    console.print()

    fng_val = fng["value"]

    for coin_id, info in COINS.items():
        scores = {}
        for period_key in PERIODS:
            result = all_results.get(f"{coin_id}_{period_key}")
            if result:
                scores[period_key] = result

        if not scores:
            continue

        short_score = scores.get("short", {}).get("score", 0)
        mid_score = scores.get("mid", {}).get("score", 0)
        long_score = scores.get("long", {}).get("score", 0)

        # 가중 종합 점수 (단기 30%, 중기 40%, 장기 30%)
        weighted = short_score * 0.3 + mid_score * 0.4 + long_score * 0.3

        # 공포-탐욕 지수 보정
        if fng_val <= 25:
            weighted += 10  # 극도의 공포 시 매수 가산
        elif fng_val >= 75:
            weighted -= 10  # 극도의 탐욕 시 매도 가산

        # 시장 변화율 보정
        if global_data:
            mcap_change = global_data.get("market_cap_change_percentage_24h_usd", 0)
            if mcap_change is not None:
                weighted += mcap_change * 0.5

        # 전략 결정
        strategies = []
        if weighted >= 40:
            action = "적극 매수"
            color = "bold green"
            strategies.append("- 분할 매수 전략: 현재가 기준 3회 분할 매수 권장")
            strategies.append("- 단기 목표 수익률: +10~15%")
        elif weighted >= 20:
            action = "매수"
            color = "green"
            strategies.append("- 소량 분할 매수 권장 (총 투자금의 30~50%)")
            strategies.append("- 추가 하락 시 추가 매수 대기")
        elif weighted >= 5:
            action = "소량 매수 / 관망"
            color = "yellow"
            strategies.append("- 소량 시험 매수 또는 관망 권장")
            strategies.append("- 명확한 추세 형성 후 진입 고려")
        elif weighted <= -40:
            action = "적극 매도"
            color = "bold red"
            strategies.append("- 보유분의 70~100% 매도 권장")
            strategies.append("- 손절가 설정 필수")
        elif weighted <= -20:
            action = "매도"
            color = "red"
            strategies.append("- 보유분의 50% 이상 매도 고려")
            strategies.append("- 반등 시 추가 매도 계획")
        elif weighted <= -5:
            action = "소량 매도 / 관망"
            color = "yellow"
            strategies.append("- 리스크 관리 차원 일부 매도 고려")
            strategies.append("- 지지선 확인 후 대응")
        else:
            action = "관망"
            color = "white"
            strategies.append("- 현재 포지션 유지")
            strategies.append("- 시장 방향성 확인 후 진입")

        # 추세 일관성 분석
        signs = [
            1 if short_score > 0 else (-1 if short_score < 0 else 0),
            1 if mid_score > 0 else (-1 if mid_score < 0 else 0),
            1 if long_score > 0 else (-1 if long_score < 0 else 0),
        ]
        if all(s > 0 for s in signs):
            strategies.append("- [green]모든 기간 상승 추세 일치 -> 신뢰도 높음[/green]")
        elif all(s < 0 for s in signs):
            strategies.append("- [red]모든 기간 하락 추세 일치 -> 신뢰도 높음[/red]")
        elif signs[0] != signs[2] and signs[0] != 0 and signs[2] != 0:
            strategies.append("- [yellow]단기/장기 추세 불일치 -> 전환 구간 가능성, 신중한 접근 필요[/yellow]")

        panel_text = f"[{color}]종합 판정: {action}[/{color}] (가중 점수: {weighted:+.1f})\n"
        panel_text += f"단기: {short_score:+d} | 중기: {mid_score:+d} | 장기: {long_score:+d}\n\n"
        panel_text += "\n".join(strategies)

        console.print(Panel(panel_text, title=f"{info['name']} ({info['symbol']}) 전략", border_style=color))

    # 리스크 경고
    console.print()
    console.print(
        Panel(
            "[bold yellow]투자 경고[/bold yellow]\n"
            "이 프로그램은 기술적 분석 기반의 참고 도구이며, 투자 조언이 아닙니다.\n"
            "암호화폐는 변동성이 매우 높으며 원금 손실 위험이 있습니다.\n"
            "반드시 본인의 판단과 책임하에 투자하시기 바랍니다.\n"
            "분할 매수/매도, 손절가 설정 등 리스크 관리를 항상 실천하세요.",
            border_style="yellow",
        )
    )


def main():
    console.print("[bold cyan]데이터를 불러오는 중...[/bold cyan]")

    # 1단계: 기본 데이터 조회
    current_data = fetch_current_prices()
    time.sleep(4)  # API rate limit 대응
    global_data = fetch_global_data()
    time.sleep(4)
    fng = fetch_fear_greed_index()

    if not current_data:
        console.print("[red]시장 데이터를 가져올 수 없습니다. 네트워크를 확인하세요.[/red]")
        sys.exit(1)

    # 2단계: 각 코인별 기간별 데이터 조회 및 분석
    all_results = {}
    for coin_id in COINS:
        for period_key, period_config in PERIODS.items():
            console.print(f"  [dim]{COINS[coin_id]['name']} {period_config['label']} 분석 중...[/dim]")
            df = fetch_market_data(coin_id, period_config["days"])
            if df is not None and len(df) > 0:
                result = analyze_signals(df, period_config)
                all_results[f"{coin_id}_{period_key}"] = result
            time.sleep(6)  # rate limit (CoinGecko 무료: ~10-30 req/min)

    # 3단계: 결과 출력
    display_market_overview(current_data, global_data, fng)

    for coin_id, info in COINS.items():
        display_coin_analysis(coin_id, info, current_data, all_results)

    generate_strategy(all_results, fng, global_data)


if __name__ == "__main__":
    main()
