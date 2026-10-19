#!/usr/bin/env python3
"""
리밸런싱 실행기 — cron에 걸어두고 잊어버리는 용도.

이 프로그램의 가장 중요한 기능은 '대부분의 날에 아무것도 하지 않는 것'이다.
리밸런싱일이 아니면 시그널을 계산하지도, 보여주지도 않는다.
볼 수 없으면 흔들릴 수도 없다.

사용법:
    python run_rebalance.py              # 오늘이 리밸런싱일이면 주문 산출
    python run_rebalance.py --force      # 일정 무시하고 조회만 (주문 미출력)
    python run_rebalance.py --dry-run    # 보유현황 파일을 갱신하지 않음
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from systematic import config, data
from systematic.schedule import is_rebalance_date, next_rebalance_date
from systematic.strategy import target_weights

console = Console()
HOLDINGS_FILE = Path(__file__).resolve().parent / "holdings.json"
LOG_FILE = Path(__file__).resolve().parent / "rebalance_log.jsonl"


def load_holdings():
    """보유 현황을 읽는다. 없으면 초기 템플릿을 만든다."""
    if not HOLDINGS_FILE.exists():
        template = {
            "cash_usd": 10000.0,
            "units": {ticker: 0.0 for ticker in config.ASSETS},
            "updated_at": None,
        }
        HOLDINGS_FILE.write_text(json.dumps(template, indent=2, ensure_ascii=False))
        console.print(
            Panel(
                f"보유 현황 파일을 생성했습니다: [cyan]{HOLDINGS_FILE.name}[/cyan]\n"
                "실제 보유 수량과 현금을 입력한 뒤 다시 실행하세요.",
                border_style="yellow",
            )
        )
        sys.exit(0)
    return json.loads(HOLDINGS_FILE.read_text())


def save_holdings(holdings):
    holdings["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    HOLDINGS_FILE.write_text(json.dumps(holdings, indent=2, ensure_ascii=False))


def append_log(record):
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def show_diagnostics(diagnostics):
    """각 자산이 왜 보유/현금인지 근거를 보여준다."""
    table = Table(title="규칙 판정", box=box.ROUNDED)
    table.add_column("자산", style="cyan")
    table.add_column("현재가", justify="right")
    table.add_column("200일선", justify="right")
    table.add_column("추세", justify="center")
    table.add_column("모멘텀", justify="right")
    table.add_column("변동성", justify="right")
    table.add_column("판정")

    for ticker in config.ASSETS:
        note = diagnostics.get(ticker)
        if not note or "price" not in note:
            table.add_row(ticker, "-", "-", "-", "-", "-", note.get("status", "-") if note else "-")
            continue

        trend_mark = "[green]위[/green]" if note["trend_ok"] else "[red]아래[/red]"
        mom = note["momentum"]
        mom_color = "green" if mom > 0 else "red"
        status_color = "green" if note["status"] == "보유" else "dim"

        table.add_row(
            ticker,
            f"${note['price']:,.2f}",
            f"${note['ma200']:,.2f}",
            trend_mark,
            f"[{mom_color}]{mom:+.1%}[/{mom_color}]",
            f"{note['vol']:.0%}",
            f"[{status_color}]{note['status']}[/{status_color}]",
        )
    console.print(table)


def build_orders(weights, holdings, prices_today):
    """목표 비중과 현재 보유를 비교해 실제 주문 목록을 만든다."""
    units = holdings.get("units", {})
    cash = float(holdings.get("cash_usd", 0.0))

    values = {t: units.get(t, 0.0) * prices_today[t] for t in config.ASSETS if t in prices_today}
    total = sum(values.values()) + cash
    if total <= 0:
        return None, 0.0, {}, 0.0

    current_w = {t: v / total for t, v in values.items()}
    max_drift = max(abs(weights.get(t, 0.0) - current_w.get(t, 0.0)) for t in config.ASSETS)

    orders = []
    for ticker in config.ASSETS:
        if ticker not in prices_today:
            continue
        target_value = weights.get(ticker, 0.0) * total
        delta = target_value - values.get(ticker, 0.0)
        if abs(delta) < config.MIN_ORDER_USD:
            continue
        orders.append(
            {
                "ticker": ticker,
                "side": "매수" if delta > 0 else "매도",
                "usd": abs(delta),
                "units": abs(delta) / prices_today[ticker],
                "price": prices_today[ticker],
                "current_w": current_w.get(ticker, 0.0),
                "target_w": weights.get(ticker, 0.0),
            }
        )
    return orders, total, current_w, max_drift


def main():
    parser = argparse.ArgumentParser(description="규칙 기반 리밸런싱 실행기")
    parser.add_argument("--force", action="store_true", help="일정 무시하고 조회만 (주문 미출력)")
    parser.add_argument("--dry-run", action="store_true", help="보유 현황 파일을 갱신하지 않음")
    parser.add_argument("--date", help="특정 날짜 기준으로 실행 (YYYY-MM-DD, 테스트용)")
    args = parser.parse_args()

    today = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp.utcnow().normalize()
    scheduled = is_rebalance_date(today)

    # ── 거래일이 아니면 여기서 끝낸다. 이게 이 프로그램의 핵심 기능이다. ──
    if not scheduled and not args.force:
        nxt = next_rebalance_date(today)
        days = (nxt - today).days
        console.print()
        console.print(
            Panel(
                f"오늘은 거래일이 아닙니다.  [bold]아무것도 하지 마세요.[/bold]\n\n"
                f"다음 리밸런싱: [cyan]{nxt.date()}[/cyan]  ({days}일 후)\n\n"
                "[dim]시세를 확인할 이유가 없습니다. 그 사이 무슨 일이 일어나든\n"
                "규칙은 다음 리밸런싱 날에 알아서 반영합니다.[/dim]",
                title="정기 점검",
                border_style="blue",
            )
        )
        console.print()
        return

    console.print("[dim]가격 데이터 갱신 중...[/dim]")
    prices = data.load_prices()
    prices = prices.loc[:today]

    if prices.empty:
        console.print("[red]가격 데이터를 가져오지 못했습니다.[/red]")
        sys.exit(1)

    weights, diagnostics = target_weights(prices)
    last_row = prices.iloc[-1]
    prices_today = {t: float(last_row[t]) for t in prices.columns if pd.notna(last_row[t])}

    console.print()
    console.rule(f"[bold cyan]리밸런싱 {today.date()}[/bold cyan]")
    console.print()
    show_diagnostics(diagnostics)

    port = diagnostics.get("_portfolio", {})
    exposure = port.get("exposure", 0.0)
    console.print(
        f"\n목표 투자비중 [bold]{exposure:.0%}[/bold]  /  현금 [bold]{1 - exposure:.0%}[/bold]"
        + (f"  [dim](추정 포트폴리오 변동성 {port['estimated_vol']:.0%})[/dim]"
           if port.get("estimated_vol") else "")
    )

    if args.force and not scheduled:
        nxt = next_rebalance_date(today)
        console.print()
        console.print(
            Panel(
                f"[yellow]조회 모드입니다. 오늘은 거래일이 아니므로 주문을 출력하지 않습니다.[/yellow]\n"
                f"다음 리밸런싱: [cyan]{nxt.date()}[/cyan]",
                border_style="yellow",
            )
        )
        return

    # ── 실제 주문 산출 ──
    holdings = load_holdings()
    orders, total, current_w, max_drift = build_orders(weights, holdings, prices_today)

    if total <= 0:
        console.print("[red]보유 자산과 현금이 모두 0입니다. holdings.json을 확인하세요.[/red]")
        sys.exit(1)

    console.print(f"\n평가 총액: [bold]${total:,.2f}[/bold]")

    if max_drift <= config.REBALANCE_BAND:
        console.print()
        console.print(
            Panel(
                f"목표 비중과의 최대 괴리가 [bold]{max_drift:.1%}[/bold]로 "
                f"밴드({config.REBALANCE_BAND:.0%}) 이내입니다.\n"
                "[bold]거래하지 않습니다.[/bold] 수수료를 아끼는 것도 수익입니다.",
                title="거래 없음",
                border_style="green",
            )
        )
        append_log({"date": today, "action": "no_trade", "max_drift": max_drift, "total": total})
        return

    order_table = Table(title="실행할 주문", box=box.ROUNDED)
    order_table.add_column("자산", style="cyan")
    order_table.add_column("구분")
    order_table.add_column("수량", justify="right")
    order_table.add_column("금액", justify="right")
    order_table.add_column("현재비중", justify="right")
    order_table.add_column("목표비중", justify="right")

    for o in sorted(orders, key=lambda x: -x["usd"]):
        color = "green" if o["side"] == "매수" else "red"
        order_table.add_row(
            o["ticker"],
            f"[{color}]{o['side']}[/{color}]",
            f"{o['units']:.6f}",
            f"${o['usd']:,.2f}",
            f"{o['current_w']:.1%}",
            f"{o['target_w']:.1%}",
        )
    console.print()
    console.print(order_table)

    est_cost = sum(o["usd"] for o in orders) * (config.FEE_RATE + config.SLIPPAGE)
    console.print(f"[dim]예상 거래비용: ${est_cost:,.2f}[/dim]")

    console.print()
    console.print(
        Panel(
            "위 주문을 거래소에서 실행한 뒤, [cyan]holdings.json[/cyan]의 보유 수량을 갱신하세요.\n"
            "[dim]--dry-run 없이 실행하면 목표 상태로 자동 기록됩니다.[/dim]",
            border_style="cyan",
        )
    )

    if not args.dry_run:
        new_units = {}
        for ticker in config.ASSETS:
            if ticker in prices_today:
                new_units[ticker] = weights.get(ticker, 0.0) * total / prices_today[ticker]
            else:
                new_units[ticker] = holdings.get("units", {}).get(ticker, 0.0)
        holdings["units"] = new_units
        holdings["cash_usd"] = total * (1.0 - float(weights.sum())) - est_cost
        save_holdings(holdings)
        console.print(f"[dim]{HOLDINGS_FILE.name} 갱신 완료[/dim]")

    append_log(
        {
            "date": today,
            "action": "rebalance",
            "total": total,
            "weights": {k: float(v) for k, v in weights.items()},
            "orders": [{k: v for k, v in o.items()} for o in orders],
        }
    )


if __name__ == "__main__":
    main()
