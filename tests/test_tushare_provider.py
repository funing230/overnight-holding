"""
Unit tests for tushare_provider.py

Tests are split into two groups:
  1. Pure unit tests (no network) — symbol conversion, date helpers, error handling
  2. Integration tests (marked with @pytest.mark.integration) — real API calls

Run unit tests only:   pytest tests/test_tushare_provider.py -m "not integration"
Run all tests:         pytest tests/test_tushare_provider.py
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from dataflows.tushare_provider import (
    _to_ts_code,
    _fmt_date,
    _parse_date,
    _safe_call,
    TushareRateLimitError,
    TusharePermissionError,
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_global_news,
    get_insider_transactions,
)


# ===================================================================
# 1. Pure unit tests — no network
# ===================================================================


class TestSymbolConversion:
    """Test _to_ts_code symbol conversion."""

    def test_passthrough_sz(self):
        assert _to_ts_code("000001.SZ") == "000001.SZ"

    def test_passthrough_sh(self):
        assert _to_ts_code("600000.SH") == "600000.SH"

    def test_passthrough_bj(self):
        assert _to_ts_code("430047.BJ") == "430047.BJ"

    def test_pure_digits_sz(self):
        assert _to_ts_code("000001") == "000001.SZ"
        assert _to_ts_code("300750") == "300750.SZ"

    def test_pure_digits_sh(self):
        assert _to_ts_code("600000") == "600000.SH"
        assert _to_ts_code("688981") == "688981.SH"

    def test_pure_digits_bj(self):
        assert _to_ts_code("430047") == "430047.BJ"
        assert _to_ts_code("830799") == "830799.BJ"

    def test_case_insensitive(self):
        assert _to_ts_code("000001.sz") == "000001.SZ"

    def test_whitespace_stripped(self):
        assert _to_ts_code("  000001.SZ  ") == "000001.SZ"

    def test_unknown_ticker_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _to_ts_code("AAPL")

    def test_known_us_ticker(self):
        assert _to_ts_code("BABA") == "09988.HK"


class TestDateHelpers:
    """Test date format conversion helpers."""

    def test_fmt_date(self):
        assert _fmt_date("2024-01-15") == "20240115"

    def test_fmt_date_no_dash(self):
        assert _fmt_date("20240115") == "20240115"

    def test_parse_date(self):
        assert _parse_date("20240115") == "2024-01-15"

    def test_parse_date_already_formatted(self):
        assert _parse_date("2024-01-15") == "2024-01-15"


class TestSafeCall:
    """Test _safe_call rate-limit detection."""

    def test_normal_call(self):
        result = _safe_call(lambda: 42)
        assert result == 42

    def test_rate_limit_chinese(self):
        def raise_rate():
            raise Exception("抱歉，您每小时最多访问该接口2次")

        with pytest.raises(TushareRateLimitError):
            _safe_call(raise_rate)

    def test_permission_error(self):
        def raise_perm():
            raise Exception("权限不足，请升级")

        with pytest.raises(TusharePermissionError):
            _safe_call(raise_perm)

    def test_other_error_passthrough(self):
        def raise_other():
            raise ValueError("some other error")

        with pytest.raises(ValueError, match="some other error"):
            _safe_call(raise_other)


class TestGetStockDataMocked:
    """Test get_stock_data with mocked Tushare API."""

    @patch("dataflows.tushare_provider._get_pro")
    def test_returns_csv(self, mock_pro):
        mock_api = MagicMock()
        mock_api.daily.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240102", "20240103"],
            "open": [9.39, 9.19],
            "high": [9.42, 9.22],
            "low": [9.21, 9.15],
            "close": [9.21, 9.20],
            "pre_close": [9.39, 9.21],
            "change": [-0.18, -0.01],
            "pct_chg": [-1.92, -0.11],
            "vol": [1158366.0, 733610.0],
            "amount": [1075742.0, 673673.0],
        })
        mock_pro.return_value = mock_api

        result = get_stock_data("000001.SZ", "2024-01-02", "2024-01-03")

        assert "000001.SZ" in result
        assert "2024-01-02" in result
        assert "Tushare" in result
        assert "9.21" in result
        mock_api.daily.assert_called_once()

    @patch("dataflows.tushare_provider._get_pro")
    def test_empty_data(self, mock_pro):
        mock_api = MagicMock()
        mock_api.daily.return_value = pd.DataFrame()
        mock_pro.return_value = mock_api

        result = get_stock_data("000001.SZ", "2024-01-02", "2024-01-03")
        assert "No data found" in result


class TestGetFundamentalsMocked:
    """Test get_fundamentals with mocked Tushare API."""

    @patch("dataflows.tushare_provider._get_pro")
    def test_returns_fundamentals(self, mock_pro):
        mock_api = MagicMock()
        mock_api.stock_basic.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "symbol": ["000001"],
            "name": ["平安银行"],
            "area": ["深圳"],
            "industry": ["银行"],
            "market": ["主板"],
            "list_date": ["19910403"],
        })
        mock_api.fina_indicator.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240315"],
            "end_date": ["20231231"],
            "eps": [2.25],
            "bps": [20.74],
            "roe": [10.24],
            "roe_waa": [11.38],
            "roa": [0.72],
            "current_ratio": [None],
            "quick_ratio": [None],
            "debt_to_assets": [91.55],
            "netprofit_yoy": [2.06],
            "or_yoy": [-8.45],
            "grossprofit_margin": [None],
            "netprofit_margin": [None],
        })
        mock_api.daily_basic.return_value = pd.DataFrame({
            "trade_date": ["20240110"],
            "close": [9.09],
            "pe_ttm": [3.64],
            "pb": [0.45],
            "ps_ttm": [1.04],
            "total_mv": [17639979.0],
            "circ_mv": [17639642.0],
            "turnover_rate": [0.44],
            "volume_ratio": [0.78],
        })
        mock_pro.return_value = mock_api

        result = get_fundamentals("000001.SZ", "2024-01-10")

        assert "平安银行" in result
        assert "银行" in result
        assert "EPS" in result
        assert "ROE" in result
        assert "Tushare" in result


class TestFinancialStatementsMocked:
    """Test balance sheet, cashflow, income statement with mocked API."""

    def _make_mock_pro(self):
        mock_api = MagicMock()
        sample_df = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240315"],
            "end_date": ["20231231"],
            "report_type": ["1"],
            "total_assets": [1000000.0],
        })
        mock_api.balancesheet.return_value = sample_df.copy()
        mock_api.cashflow.return_value = sample_df.copy()
        mock_api.income.return_value = sample_df.copy()
        return mock_api

    @patch("dataflows.tushare_provider._get_pro")
    def test_balance_sheet(self, mock_pro):
        mock_pro.return_value = self._make_mock_pro()
        result = get_balance_sheet("000001.SZ", "quarterly", "2024-01-10")
        assert "Balance Sheet" in result
        assert "000001.SZ" in result

    @patch("dataflows.tushare_provider._get_pro")
    def test_cashflow(self, mock_pro):
        mock_pro.return_value = self._make_mock_pro()
        result = get_cashflow("000001.SZ", "quarterly", "2024-01-10")
        assert "Cash Flow" in result

    @patch("dataflows.tushare_provider._get_pro")
    def test_income_statement(self, mock_pro):
        mock_pro.return_value = self._make_mock_pro()
        result = get_income_statement("000001.SZ", "quarterly", "2024-01-10")
        assert "Income Statement" in result

    @patch("dataflows.tushare_provider._get_pro")
    def test_annual_filter(self, mock_pro):
        mock_api = MagicMock()
        mock_api.balancesheet.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20240315", "20231025"],
            "end_date": ["20231231", "20230930"],
            "report_type": ["1", "1"],
        })
        mock_pro.return_value = mock_api
        result = get_balance_sheet("000001.SZ", "annual", "2024-01-10")
        assert "20231231" in result
        assert "20230930" not in result


class TestInsiderTransactionsMocked:
    """Test get_insider_transactions with mocked API."""

    @patch("dataflows.tushare_provider._get_pro")
    def test_block_trade(self, mock_pro):
        mock_api = MagicMock()
        mock_api.block_trade.return_value = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": ["20231127"],
            "price": [9.50],
            "vol": [100000],
            "amount": [950000],
            "buyer": ["机构A"],
            "seller": ["机构B"],
        })
        mock_pro.return_value = mock_api

        result = get_insider_transactions("000001.SZ")
        assert "000001.SZ" in result
        assert "机构A" in result

    @patch("dataflows.tushare_provider._get_pro")
    def test_empty_fallback(self, mock_pro):
        mock_api = MagicMock()
        mock_api.block_trade.return_value = pd.DataFrame()
        mock_api.stk_holdertrade.return_value = pd.DataFrame()
        mock_pro.return_value = mock_api

        result = get_insider_transactions("000001.SZ")
        assert "No insider" in result


class TestNewsMocked:
    """Test news functions with mocked API."""

    @patch("dataflows.tushare_provider._get_pro")
    def test_get_news_rate_limit(self, mock_pro):
        mock_api = MagicMock()
        mock_api.news.side_effect = Exception("抱歉，您每小时最多访问该接口2次")
        mock_api.stock_basic.return_value = pd.DataFrame({"name": ["平安银行"]})
        mock_pro.return_value = mock_api

        with pytest.raises(TushareRateLimitError):
            get_news("000001.SZ", "2024-01-01", "2024-01-10")

    @patch("dataflows.tushare_provider._get_pro")
    def test_get_global_news_empty(self, mock_pro):
        mock_api = MagicMock()
        mock_api.news.return_value = pd.DataFrame()
        mock_pro.return_value = mock_api

        result = get_global_news("2024-01-10")
        assert "No global news" in result


# ===================================================================
# 2. Integration tests — real API calls (skip in CI)
# ===================================================================


def _skip_if_tushare_external_constraint(exc: Exception) -> None:
    msg = str(exc).lower()
    if isinstance(exc, (TushareRateLimitError, TusharePermissionError)):
        pytest.skip(f"Tushare integration unavailable/limited: {exc}")
    transient_markers = (
        "connection",
        "timeout",
        "remote",
        "reset",
        "refused",
        "max retries exceeded",
    )
    if any(marker in msg for marker in transient_markers):
        pytest.skip(f"Tushare integration transient failure: {exc}")
    raise exc


@pytest.mark.integration
class TestTushareIntegration:
    """Integration tests that hit the real Tushare API.

    Run with: pytest tests/test_tushare_provider.py -m integration
    """

    def test_get_stock_data_real(self):
        try:
            result = get_stock_data("000001.SZ", "2024-01-01", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "000001.SZ" in result
        assert "Open" in result
        assert "Close" in result
        lines = result.strip().split("\n")
        assert len(lines) > 3  # header + at least some data rows

    def test_get_fundamentals_real(self):
        try:
            result = get_fundamentals("000001.SZ", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "平安银行" in result
        assert "EPS" in result

    def test_get_balance_sheet_real(self):
        try:
            result = get_balance_sheet("000001.SZ", "quarterly", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "Balance Sheet" in result

    def test_get_cashflow_real(self):
        try:
            result = get_cashflow("000001.SZ", "quarterly", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "Cash Flow" in result

    def test_get_income_statement_real(self):
        try:
            result = get_income_statement("000001.SZ", "quarterly", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "Income Statement" in result

    def test_get_insider_transactions_real(self):
        try:
            result = get_insider_transactions("000001.SZ")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        # May or may not have data, just ensure no crash
        assert isinstance(result, str)

    def test_symbol_conversion_in_call(self):
        """Test that pure digit symbols work end-to-end."""
        try:
            result = get_stock_data("000001", "2024-01-01", "2024-01-10")
        except Exception as exc:
            _skip_if_tushare_external_constraint(exc)
        assert "000001.SZ" in result

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError):
            get_stock_data("AAPL", "2024-01-01", "2024-01-10")


# ===================================================================
# 3. Interface routing test
# ===================================================================


class TestInterfaceRouting:
    """Test vendor routing registrations stay aligned with current interface design."""

    def test_tushare_in_vendor_list(self):
        from dataflows.interface import VENDOR_LIST
        assert "tushare" in VENDOR_LIST

    def test_vendor_registration_matches_current_design(self):
        from dataflows.interface import VENDOR_METHODS

        tushare_methods = {
            "get_stock_data",
            "get_indicators",
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        }
        local_only_methods = {
            "get_overnight_candidates",
            "get_overnight_candidate_summary",
            "get_overnight_candidate_payload",
        }

        for method in tushare_methods:
            assert "tushare" in VENDOR_METHODS[method], f"tushare missing from {method}"

        for method in local_only_methods:
            assert set(VENDOR_METHODS[method]) == {"local"}, f"expected local-only routing for {method}"

    def test_route_to_tushare(self):
        """Test routing with tushare as configured vendor."""
        from dataflows.config import set_config
        from dataflows.interface import route_to_vendor

        set_config({
            "data_vendors": {
                "core_stock_apis": "tushare",
                "technical_indicators": "tushare",
                "fundamental_data": "tushare",
                "news_data": "tushare",
            }
        })

        # This will actually call the real API
        result = route_to_vendor("get_stock_data", "000001.SZ", "2024-01-01", "2024-01-05")
        assert "000001.SZ" in result or "Open" in result or "Date" in result
        assert "Tushare" in result or "Close" in result
