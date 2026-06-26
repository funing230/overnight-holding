"""
Tushare data provider for TradingAgents.

Provides A-share market data via Tushare Pro API with a custom endpoint.
Implements the same function signatures as y_finance.py / alpha_vantage.py
so it plugs directly into the vendor routing in interface.py.
"""

import os
import re
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated, Optional

# ---------------------------------------------------------------------------
# Tushare Pro API singleton
# ---------------------------------------------------------------------------

_pro = None


def _load_env_file() -> None:
    """Load a project-local .env file when python-dotenv is available.

    OpenClaw tool executions do not automatically inherit per-project .env
    values, while this project stores data-provider tokens there.  Keeping this
    small local loader makes scripts work from the repo without requiring users
    to export TUSHARE_TOKEN in every shell.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass

    # Minimal fallback for environments without python-dotenv.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_pro():
    """Lazy-init Tushare Pro API instance."""
    global _pro
    if _pro is None:
        import tushare as ts

        _load_env_file()
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "TUSHARE_TOKEN environment variable is not set. "
                "Set it in the project .env file or run: "
                "export TUSHARE_TOKEN='your-token-here'"
            )
        api_url = os.getenv("TUSHARE_API_URL", "").strip()
        _pro = ts.pro_api(token)
        _pro._DataApi__token = token
        # The official tushare SDK defaults to http://api.waditu.com/dataapi.
        # Only override when explicitly requested; api.tushare.pro is the raw
        # HTTP endpoint and does not behave like the SDK DataApi endpoint.
        if api_url:
            _pro._DataApi__http_url = api_url
    return _pro


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

# Mapping of well-known US tickers to A-share ts_codes (extend as needed)
_US_TO_A_SHARE = {
    "BABA": "09988.HK",
}


def _to_ts_code(symbol: str) -> str:
    """Convert various symbol formats to Tushare ts_code.

    Accepted inputs:
      - Already qualified: '000001.SZ', '600000.SH' → pass through
      - Pure digits: '000001' → guess exchange by leading digit
      - US-style ticker: 'BABA' → lookup table or raise
    """
    symbol = symbol.strip().upper()

    # Already in ts_code format
    if re.match(r"^\d{6}\.(SZ|SH|BJ)$", symbol):
        return symbol

    # Pure 6-digit code
    if re.match(r"^\d{6}$", symbol):
        first = symbol[0]
        if first in ("0", "3"):
            return f"{symbol}.SZ"
        elif first in ("6", "9"):
            return f"{symbol}.SH"
        elif first in ("4", "8"):
            return f"{symbol}.BJ"
        return f"{symbol}.SZ"

    # Lookup table for foreign tickers
    if symbol in _US_TO_A_SHARE:
        return _US_TO_A_SHARE[symbol]

    raise ValueError(
        f"Cannot convert '{symbol}' to Tushare ts_code. "
        "Use A-share codes like '000001.SZ' or '600000.SH'."
    )


def _fmt_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' → 'YYYYMMDD'."""
    return date_str.replace("-", "")


def _parse_date(date_str: str) -> str:
    """Convert 'YYYYMMDD' → 'YYYY-MM-DD'."""
    if len(date_str) == 8 and "-" not in date_str:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


# ---------------------------------------------------------------------------
# Rate-limit error (mirrors AlphaVantageRateLimitError for fallback logic)
# ---------------------------------------------------------------------------


class TushareRateLimitError(Exception):
    """Raised when Tushare API rate limit is hit."""
    pass


class TusharePermissionError(TushareRateLimitError):
    """Raised when Tushare denies access due to insufficient permissions.

    Subclassing TushareRateLimitError preserves existing fallback behavior in
    interface routing while allowing tests/callers to distinguish permission
    failures from transient throttling.
    """
    pass


def _safe_call(func, *args, **kwargs):
    """Wrap a Tushare call and normalize common external API failures."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = str(e)
        lower_msg = msg.lower()
        if any(kw in msg for kw in ("权限", "访问权限", "permission")):
            raise TusharePermissionError(msg) from e
        if any(kw in msg for kw in ("每分钟", "每小时", "最多访问", "超限", "频率")) or "rate limit" in lower_msg:
            raise TushareRateLimitError(msg) from e
        raise


# ---------------------------------------------------------------------------
# 1. Core stock data  (maps to get_stock_data)
# ---------------------------------------------------------------------------


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve daily OHLCV data from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(symbol)

    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=_fmt_date(start_date),
        end_date=_fmt_date(end_date),
    )

    if df is None or df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Rename columns to match yfinance convention
    df = df.rename(columns={
        "trade_date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
        "amount": "Amount",
        "pct_chg": "Pct_Change",
    })
    df["Date"] = df["Date"].apply(_parse_date)
    df = df.sort_values("Date")

    cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount", "Pct_Change"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)

    header = f"# Stock data for {ts_code} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

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
    """Calculate technical indicators from Tushare daily data using stockstats."""
    from stockstats import wrap

    pro = _get_pro()
    ts_code = _to_ts_code(symbol)

    # Fetch enough history for indicator warm-up
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days + 250)

    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=_fmt_date(start_dt.strftime("%Y-%m-%d")),
        end_date=_fmt_date(curr_date),
    )

    if df is None or df.empty:
        return f"No data for {symbol} to compute indicator '{indicator}'"

    # Prepare for stockstats (requires lowercase column names)
    df = df.rename(columns={"vol": "volume"})
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)

    ss = wrap(df)
    try:
        ss[indicator]  # trigger calculation
    except Exception as e:
        return f"Indicator '{indicator}' not supported or calculation failed: {e}"

    # Copy computed indicator back to the plain DataFrame to avoid
    # StockDataFrame's __getitem__ intercepting column access.
    ind_col = indicator
    if ind_col not in ss.columns:
        candidates = [c for c in ss.columns if c.startswith(indicator)]
        if candidates:
            ind_col = candidates[0]
        else:
            return f"Indicator '{indicator}' computed but column not found in: {list(ss.columns)}"

    df[ind_col] = ss[ind_col].values

    # Filter to look-back window
    window_start = end_dt - timedelta(days=look_back_days)
    mask = (df["date"] >= window_start) & (df["date"] <= end_dt)
    result_df = df.loc[mask]

    ind_string = ""
    for _, row in result_df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        val = row.get(ind_col, "N/A")
        if pd.isna(val):
            val = "N/A"
        else:
            val = f"{val:.4f}" if isinstance(val, float) else str(val)
        ind_string += f"{date_str}: {val}\n"

    return (
        f"## {indicator} values for {ts_code} "
        f"from {window_start.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        f"{ind_string}\n"
        f"Source: Tushare + stockstats"
    )


# ---------------------------------------------------------------------------
# 3. Fundamentals  (maps to get_fundamentals)
# ---------------------------------------------------------------------------


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Retrieve company fundamentals from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    # Basic info
    basic = _safe_call(pro.stock_basic, ts_code=ts_code,
                       fields="ts_code,symbol,name,area,industry,market,list_date")

    # Financial indicators (latest)
    end_date = _fmt_date(curr_date) if curr_date else _fmt_date(datetime.now().strftime("%Y-%m-%d"))
    start_fina = str(int(end_date[:4]) - 2) + "0101"
    fina = _safe_call(
        pro.fina_indicator,
        ts_code=ts_code,
        start_date=start_fina,
        end_date=end_date,
        fields=(
            "ts_code,ann_date,end_date,eps,bps,roe,roe_waa,roa,"
            "current_ratio,quick_ratio,debt_to_assets,"
            "netprofit_yoy,or_yoy,grossprofit_margin,netprofit_margin"
        ),
    )

    # Daily basic (PE/PB/MV)
    daily_basic = _safe_call(
        pro.daily_basic,
        ts_code=ts_code,
        start_date=str(int(end_date[:4])) + "0101",
        end_date=end_date,
    )

    lines = []

    # Company info
    if basic is not None and not basic.empty:
        row = basic.iloc[0]
        lines.append(f"Name: {row.get('name', 'N/A')}")
        lines.append(f"Symbol: {row.get('symbol', 'N/A')}")
        lines.append(f"Industry: {row.get('industry', 'N/A')}")
        lines.append(f"Area: {row.get('area', 'N/A')}")
        lines.append(f"Market: {row.get('market', 'N/A')}")
        lines.append(f"List Date: {_parse_date(str(row.get('list_date', 'N/A')))}")

    # Latest valuation
    if daily_basic is not None and not daily_basic.empty:
        latest = daily_basic.sort_values("trade_date", ascending=False).iloc[0]
        for label, col in [
            ("PE (TTM)", "pe_ttm"),
            ("PB", "pb"),
            ("PS (TTM)", "ps_ttm"),
            ("Total MV (万元)", "total_mv"),
            ("Circ MV (万元)", "circ_mv"),
            ("Turnover Rate", "turnover_rate"),
            ("Volume Ratio", "volume_ratio"),
        ]:
            val = latest.get(col)
            if val is not None and not pd.isna(val):
                lines.append(f"{label}: {val}")

    # Latest financial indicators
    if fina is not None and not fina.empty:
        latest_fina = fina.sort_values("end_date", ascending=False).iloc[0]
        lines.append(f"\n--- Financial Indicators (period ending {_parse_date(str(latest_fina.get('end_date', '')))}) ---")
        for label, col in [
            ("EPS", "eps"),
            ("BPS (Book Value Per Share)", "bps"),
            ("ROE (%)", "roe"),
            ("ROE Weighted (%)", "roe_waa"),
            ("ROA (%)", "roa"),
            ("Current Ratio", "current_ratio"),
            ("Quick Ratio", "quick_ratio"),
            ("Debt to Assets (%)", "debt_to_assets"),
            ("Net Profit YoY (%)", "netprofit_yoy"),
            ("Revenue YoY (%)", "or_yoy"),
            ("Gross Margin (%)", "grossprofit_margin"),
            ("Net Margin (%)", "netprofit_margin"),
        ]:
            val = latest_fina.get(col)
            if val is not None and not pd.isna(val):
                lines.append(f"{label}: {val}")

    header = f"# Company Fundamentals for {ts_code}\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(lines) if lines else f"No fundamentals data found for {ticker}"


# ---------------------------------------------------------------------------
# 4. Balance sheet  (maps to get_balance_sheet)
# ---------------------------------------------------------------------------


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve balance sheet from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    end = _fmt_date(curr_date) if curr_date else _fmt_date(datetime.now().strftime("%Y-%m-%d"))
    start = str(int(end[:4]) - 3) + "0101"

    df = _safe_call(pro.balancesheet, ts_code=ts_code, start_date=start, end_date=end)

    if df is None or df.empty:
        return f"No balance sheet data found for {ticker}"

    # Filter by report type (1=合并报表)
    if "report_type" in df.columns:
        df = df[df["report_type"] == "1"]

    if freq.lower() == "annual":
        df = df[df["end_date"].str.endswith("1231")]

    df = df.sort_values("end_date", ascending=False).head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Balance Sheet for {ts_code} ({freq})\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 5. Cash flow  (maps to get_cashflow)
# ---------------------------------------------------------------------------


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve cash flow statement from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    end = _fmt_date(curr_date) if curr_date else _fmt_date(datetime.now().strftime("%Y-%m-%d"))
    start = str(int(end[:4]) - 3) + "0101"

    df = _safe_call(pro.cashflow, ts_code=ts_code, start_date=start, end_date=end)

    if df is None or df.empty:
        return f"No cash flow data found for {ticker}"

    if "report_type" in df.columns:
        df = df[df["report_type"] == "1"]

    if freq.lower() == "annual":
        df = df[df["end_date"].str.endswith("1231")]

    df = df.sort_values("end_date", ascending=False).head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Cash Flow for {ts_code} ({freq})\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 6. Income statement  (maps to get_income_statement)
# ---------------------------------------------------------------------------


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Retrieve income statement from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    end = _fmt_date(curr_date) if curr_date else _fmt_date(datetime.now().strftime("%Y-%m-%d"))
    start = str(int(end[:4]) - 3) + "0101"

    df = _safe_call(pro.income, ts_code=ts_code, start_date=start, end_date=end)

    if df is None or df.empty:
        return f"No income statement data found for {ticker}"

    if "report_type" in df.columns:
        df = df[df["report_type"] == "1"]

    if freq.lower() == "annual":
        df = df[df["end_date"].str.endswith("1231")]

    df = df.sort_values("end_date", ascending=False).head(8)

    csv_string = df.to_csv(index=False)
    header = f"# Income Statement for {ts_code} ({freq})\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# 7. News  (maps to get_news)
# ---------------------------------------------------------------------------


def get_news(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Retrieve news for a stock from Tushare.

    Falls back to a simple message if the news API is rate-limited
    (Tushare free tier: 2 calls/hour for news).
    """
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    try:
        df = _safe_call(
            pro.news,
            src="sina",
            start_date=_fmt_date(start_date),
            end_date=_fmt_date(end_date),
        )
    except TushareRateLimitError:
        raise  # let route_to_vendor fallback
    except Exception:
        df = None

    if df is None or df.empty:
        return f"No news found for {ticker} between {start_date} and {end_date} (Tushare)"

    # Filter by ticker keyword in title/content if possible
    name = ""
    try:
        basic = pro.stock_basic(ts_code=ts_code, fields="name")
        if basic is not None and not basic.empty:
            name = basic.iloc[0]["name"]
    except Exception:
        pass

    if name and "title" in df.columns:
        mask = df["title"].str.contains(name, na=False)
        filtered = df[mask]
        if not filtered.empty:
            df = filtered

    df = df.head(20)

    news_str = ""
    for _, row in df.iterrows():
        title = row.get("title", "No title")
        content = row.get("content", "")
        dt = row.get("datetime", "")
        news_str += f"### {title}\n"
        if dt:
            news_str += f"Date: {dt}\n"
        if content:
            news_str += f"{content[:300]}\n"
        news_str += "\n"

    return f"## {ts_code} News, from {start_date} to {end_date}:\n\n{news_str}"


# ---------------------------------------------------------------------------
# 8. Global news  (maps to get_global_news)
# ---------------------------------------------------------------------------


def get_global_news(
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Max articles"] = 10,
) -> str:
    """Retrieve global/macro news from Tushare."""
    pro = _get_pro()

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)

    try:
        df = _safe_call(
            pro.news,
            src="sina",
            start_date=_fmt_date(start_dt.strftime("%Y-%m-%d")),
            end_date=_fmt_date(curr_date),
        )
    except TushareRateLimitError:
        raise
    except Exception:
        df = None

    if df is None or df.empty:
        return f"No global news found for {curr_date} (Tushare)"

    df = df.head(limit)

    news_str = ""
    for _, row in df.iterrows():
        title = row.get("title", "No title")
        content = row.get("content", "")
        dt = row.get("datetime", "")
        news_str += f"### {title}\n"
        if dt:
            news_str += f"Date: {dt}\n"
        if content:
            news_str += f"{content[:300]}\n"
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
    """Retrieve block trades / insider transactions from Tushare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    # Try block_trade first (大宗交易)
    df = None
    try:
        df = _safe_call(pro.block_trade, ts_code=ts_code, start_date=start, end_date=end)
    except Exception:
        pass

    if df is None or df.empty:
        # Fallback: stk_holdertrade (股东增减持)
        try:
            df = _safe_call(pro.stk_holdertrade, ts_code=ts_code, start_date=start, end_date=end)
        except Exception:
            pass

    if df is None or df.empty:
        return f"No insider/block transactions found for {ticker}"

    csv_string = df.to_csv(index=False)
    header = f"# Insider/Block Transactions for {ts_code}\n"
    header += f"# Source: Tushare | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string
