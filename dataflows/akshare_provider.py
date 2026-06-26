"""
AkShare data provider for TradingAgents.

Provides A-share market data via AkShare (free, open-source).
Implements the same function signatures as y_finance.py / tushare_provider.py
so it plugs directly into the vendor routing in interface.py.

Data sources: East Money, Sina Finance, etc. (aggregated by AkShare).
"""

import re
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from typing import Annotated, Optional


# ---------------------------------------------------------------------------
# Rate-limit error (for fallback logic in interface.py)
# ---------------------------------------------------------------------------


class AkShareRateLimitError(Exception):
    """Raised when AkShare data source rate limit is hit."""
    pass


def _safe_call(func, *args, **kwargs):
    """Wrap an AkShare call; detect rate-limit / anti-crawl errors."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = str(e)
        if any(kw in msg for kw in ("频率", "rate", "限制", "访问过于频繁", "429", "Too Many")):
            raise AkShareRateLimitError(msg) from e
        raise


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------


def _to_ak_symbol(symbol: str) -> str:
    """Convert various symbol formats to AkShare 6-digit code.

    AkShare stock_zh_a_hist expects pure 6-digit codes like '000001'.
    """
    symbol = symbol.strip().upper()

    # Already 6 digits
    if re.match(r"^\d{6}$", symbol):
        return symbol

    # Tushare format: 000001.SZ → 000001
    m = re.match(r"^(\d{6})\.(SZ|SH|BJ)$", symbol)
    if m:
        return m.group(1)

    raise ValueError(
        f"Cannot convert '{symbol}' to AkShare symbol. "
        "Use A-share codes like '000001' or '000001.SZ'."
    )


def _to_ak_market(symbol: str) -> str:
    """Determine market (sz/sh/bj) from symbol for APIs that need it."""
    symbol = symbol.strip().upper()

    # Explicit exchange suffix
    if ".SZ" in symbol:
        return "sz"
    if ".SH" in symbol:
        return "sh"
    if ".BJ" in symbol:
        return "bj"

    # Guess from leading digit
    code = _to_ak_symbol(symbol)
    first = code[0]
    if first in ("0", "3"):
        return "sz"
    elif first in ("6", "9"):
        return "sh"
    elif first in ("4", "8"):
        return "bj"
    return "sz"


def _fmt_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' → 'YYYYMMDD'."""
    return date_str.replace("-", "")


# ---------------------------------------------------------------------------
# 1. Core stock data  (maps to get_stock_data)
# ---------------------------------------------------------------------------


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve daily OHLCV data from AkShare (East Money)."""

    code = _to_ak_symbol(symbol)

    df = _safe_call(
        ak.stock_zh_a_hist,
        symbol=code,
        period="daily",
        start_date=_fmt_date(start_date),
        end_date=_fmt_date(end_date),
        adjust="qfq",
    )

    if df is None or df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Rename Chinese columns to English
    col_map = {
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "Pct_Change",
        "涨跌额": "Change",
        "换手率": "Turnover",
    }
    df = df.rename(columns=col_map)

    cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount", "Pct_Change", "Turnover"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)

    header = f"# Stock data for {code} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Source: AkShare (East Money) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 2. Technical indicators  (maps to get_indicators)
# ---------------------------------------------------------------------------


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate technical indicators from AkShare daily data using stockstats."""
    from stockstats import wrap

    code = _to_ak_symbol(symbol)

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days + 250)

    df = _safe_call(
        ak.stock_zh_a_hist,
        symbol=code,
        period="daily",
        start_date=_fmt_date(start_dt.strftime("%Y-%m-%d")),
        end_date=_fmt_date(curr_date),
        adjust="qfq",
    )

    if df is None or df.empty:
        return f"No data for {symbol} to compute indicator '{indicator}'"

    df = df.rename(columns={
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    ss = wrap(df)
    try:
        ss[indicator]
    except Exception as e:
        return f"Indicator '{indicator}' not supported or calculation failed: {e}"

    window_start = end_dt - timedelta(days=look_back_days)
    mask = (df["Date"] >= window_start) & (df["Date"] <= end_dt)
    result_df = df.loc[mask].copy()

    ind_string = ""
    for _, row in result_df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        val = row.get(indicator, "N/A")
        if pd.isna(val):
            val = "N/A"
        ind_string += f"{date_str}: {val}\n"

    return (
        f"## {indicator} values for {code} "
        f"from {window_start.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        f"{ind_string}\n"
        f"Source: AkShare + stockstats"
    )


# ---------------------------------------------------------------------------
# 3. Fundamentals  (maps to get_fundamentals)
# ---------------------------------------------------------------------------


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Retrieve company fundamentals from AkShare (East Money)."""

    code = _to_ak_symbol(ticker)
    lines = []

    # Basic info
    try:
        info = _safe_call(ak.stock_individual_info_em, symbol=code)
        if info is not None and not info.empty:
            for _, row in info.iterrows():
                item = row.get("item", "")
                value = row.get("value", "")
                if item and value:
                    lines.append(f"{item}: {value}")
    except Exception:
        pass

    # Financial summary from THS
    try:
        fina = _safe_call(
            ak.stock_financial_abstract_ths,
            symbol=code,
            indicator="按报告期",
        )
        if fina is not None and not fina.empty:
            # Get latest 2 periods
            latest = fina.head(2)
            lines.append("\n--- Financial Summary (latest periods) ---")
            for _, row in latest.iterrows():
                period = row.get("报告期", "")
                lines.append(f"\nPeriod: {period}")
                for col in fina.columns:
                    if col == "报告期":
                        continue
                    val = row.get(col)
                    if val is not None and str(val) != "False" and str(val) != "nan":
                        lines.append(f"  {col}: {val}")
    except Exception:
        pass

    # Fund flow (latest)
    try:
        market = _to_ak_market(ticker)
        fund = _safe_call(ak.stock_individual_fund_flow, stock=code, market=market)
        if fund is not None and not fund.empty:
            latest_fund = fund.iloc[0]
            lines.append("\n--- Fund Flow (latest) ---")
            for col in fund.columns:
                if col == "日期":
                    lines.append(f"  Date: {latest_fund[col]}")
                else:
                    val = latest_fund.get(col)
                    if val is not None and not pd.isna(val):
                        lines.append(f"  {col}: {val}")
    except Exception:
        pass

    header = f"# Company Fundamentals for {code}\n"
    header += f"# Source: AkShare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(lines) if lines else f"No fundamentals data found for {ticker}"


# ---------------------------------------------------------------------------
# 4. Balance sheet  (maps to get_balance_sheet)
# ---------------------------------------------------------------------------


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve balance sheet from AkShare (Sina Finance)."""

    code = _to_ak_symbol(ticker)

    try:
        df = _safe_call(
            ak.stock_financial_report_sina,
            stock=code,
            symbol="资产负债表",
        )
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {e}"

    if df is None or df.empty:
        return f"No balance sheet data found for {ticker}"

    # Filter by report type
    if "类型" in df.columns:
        df = df[df["类型"] == "合并期末"]

    if freq.lower() == "annual":
        df = df[df["报告日"].astype(str).str.endswith("1231")]

    # Filter by curr_date
    if curr_date:
        cutoff = _fmt_date(curr_date)
        df = df[df["报告日"].astype(str) <= cutoff]

    df = df.head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Balance Sheet for {code} ({freq})\n"
    header += f"# Source: AkShare (Sina) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 5. Cash flow  (maps to get_cashflow)
# ---------------------------------------------------------------------------


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve cash flow statement from AkShare (Sina Finance)."""

    code = _to_ak_symbol(ticker)

    try:
        df = _safe_call(
            ak.stock_financial_report_sina,
            stock=code,
            symbol="现金流量表",
        )
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {e}"

    if df is None or df.empty:
        return f"No cash flow data found for {ticker}"

    if "类型" in df.columns:
        df = df[df["类型"] == "合并期末"]

    if freq.lower() == "annual":
        df = df[df["报告日"].astype(str).str.endswith("1231")]

    if curr_date:
        cutoff = _fmt_date(curr_date)
        df = df[df["报告日"].astype(str) <= cutoff]

    df = df.head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Cash Flow for {code} ({freq})\n"
    header += f"# Source: AkShare (Sina) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 6. Income statement  (maps to get_income_statement)
# ---------------------------------------------------------------------------


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve income statement from AkShare (Sina Finance)."""

    code = _to_ak_symbol(ticker)

    try:
        df = _safe_call(
            ak.stock_financial_report_sina,
            stock=code,
            symbol="利润表",
        )
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {e}"

    if df is None or df.empty:
        return f"No income statement data found for {ticker}"

    if "类型" in df.columns:
        df = df[df["类型"] == "合并期末"]

    if freq.lower() == "annual":
        df = df[df["报告日"].astype(str).str.endswith("1231")]

    if curr_date:
        cutoff = _fmt_date(curr_date)
        df = df[df["报告日"].astype(str) <= cutoff]

    df = df.head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Income Statement for {code} ({freq})\n"
    header += f"# Source: AkShare (Sina) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 7. News  (maps to get_news)
# ---------------------------------------------------------------------------


def get_news(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Retrieve stock-specific news from AkShare (East Money)."""

    code = _to_ak_symbol(ticker)

    try:
        df = _safe_call(ak.stock_news_em, symbol=code)
    except AkShareRateLimitError:
        raise
    except Exception:
        df = None

    if df is None or df.empty:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    # Filter by date range if possible
    if "发布时间" in df.columns:
        try:
            df["_dt"] = pd.to_datetime(df["发布时间"], errors="coerce")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)
            df = df[(df["_dt"] >= start_dt) & (df["_dt"] < end_dt)]
        except Exception:
            pass

    df = df.head(20)

    news_str = ""
    for _, row in df.iterrows():
        title = row.get("新闻标题", row.get("title", "No title"))
        content = row.get("新闻内容", row.get("content", ""))
        pub_time = row.get("发布时间", "")
        source = row.get("文章来源", "")
        link = row.get("新闻链接", "")

        news_str += f"### {title}"
        if source:
            news_str += f" (source: {source})"
        news_str += "\n"
        if pub_time:
            news_str += f"Date: {pub_time}\n"
        if content:
            news_str += f"{str(content)[:300]}\n"
        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"

    return f"## {code} News, from {start_date} to {end_date}:\n\n{news_str}"


# ---------------------------------------------------------------------------
# 8. Global news  (maps to get_global_news)
# ---------------------------------------------------------------------------


def get_global_news(
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Max articles"] = 10,
) -> str:
    """Retrieve global/macro news from AkShare (East Money)."""

    try:
        df = _safe_call(ak.stock_info_global_em)
    except AkShareRateLimitError:
        raise
    except Exception:
        df = None

    if df is None or df.empty:
        return f"No global news found for {curr_date}"

    df = df.head(limit)

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)

    news_str = ""
    for _, row in df.iterrows():
        title = row.get("标题", "No title")
        summary = row.get("摘要", "")
        pub_time = row.get("发布时间", "")
        link = row.get("链接", "")

        news_str += f"### {title}\n"
        if pub_time:
            news_str += f"Date: {pub_time}\n"
        if summary:
            news_str += f"{str(summary)[:300]}\n"
        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"

    return (
        f"## Global Market News, "
        f"from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n{news_str}"
    )


# ---------------------------------------------------------------------------
# 9. Insider / block transactions  (maps to get_insider_transactions)
# ---------------------------------------------------------------------------


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Retrieve block trades from AkShare (East Money)."""

    code = _to_ak_symbol(ticker)

    # Try daily block trade summary
    df = None
    try:
        df = _safe_call(ak.stock_dzjy_mrtj, start_date="20230101", end_date="20251231")
        if df is not None and not df.empty and "证券代码" in df.columns:
            df = df[df["证券代码"] == code]
    except Exception:
        df = None

    if df is None or df.empty:
        return f"No insider/block transactions found for {ticker}"

    df = df.head(20)

    csv_string = df.to_csv(index=False)
    header = f"# Block Transactions for {code}\n"
    header += f"# Source: AkShare (East Money) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string