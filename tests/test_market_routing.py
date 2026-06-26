"""
Unit tests for market detection and vendor routing in interface.py.
"""

import pytest
from unittest.mock import patch, MagicMock, call

from dataflows.interface import (
    detect_market,
    _is_empty_result,
    route_to_vendor,
    VENDOR_MARKETS,
    VENDOR_METHODS,
)


# ===================================================================
# Tests — detect_market
# ===================================================================


class TestDetectMarket:
    """Test ticker → market detection."""

    def test_sz_suffix(self):
        assert detect_market("000001.SZ") == "cn"

    def test_sh_suffix(self):
        assert detect_market("600000.SH") == "cn"

    def test_bj_suffix(self):
        assert detect_market("830001.BJ") == "cn"

    def test_pure_6_digit(self):
        assert detect_market("000001") == "cn"

    def test_6_digit_sh(self):
        assert detect_market("600519") == "cn"

    def test_us_ticker(self):
        assert detect_market("AAPL") == "us"

    def test_us_ticker_short(self):
        assert detect_market("GM") == "us"

    def test_us_ticker_with_dot(self):
        assert detect_market("BRK.B") == "us"

    def test_us_index(self):
        assert detect_market("^GSPC") == "us"

    def test_us_index_nasdaq(self):
        assert detect_market("^IXIC") == "us"

    def test_empty_string(self):
        assert detect_market("") == "global"

    def test_lowercase_cn(self):
        assert detect_market("000001.sz") == "cn"

    def test_lowercase_us(self):
        assert detect_market("aapl") == "us"


# ===================================================================
# Tests — _is_empty_result
# ===================================================================


class TestIsEmptyResult:
    """Test empty/fake-success detection."""

    def test_none_is_empty(self):
        assert _is_empty_result(None) is True

    def test_no_data_found_message(self):
        assert _is_empty_result("No data found for symbol '000001.SZ'") is True

    def test_no_news_found(self):
        assert _is_empty_result("No news found for AAPL between 2025-01-01 and 2025-03-31") is True

    def test_csv_header_only(self):
        assert _is_empty_result("# Header\nDate,Open,High,Low,Close\n") is True

    def test_valid_csv_data(self):
        csv = "Date,Open,High,Low,Close\n2025-01-01,10.0,11.0,9.5,10.5\n"
        assert _is_empty_result(csv) is False

    def test_valid_text_report(self):
        report = "The stock showed strong momentum.\nPrice increased 5% this week.\nVolume was above average.\n"
        assert _is_empty_result(report) is False

    def test_multiline_report(self):
        report = "# Report\nLine 1\nLine 2\nLine 3\n"
        assert _is_empty_result(report) is False


# ===================================================================
# Tests — Market-aware vendor filtering
#
# We patch VENDOR_METHODS to use mock functions so no real API calls
# are made, and patch VENDOR_LIMITERS to skip rate limiting delays.
# ===================================================================


def _mock_vendor_methods(method_name, vendor_mocks):
    """Build a patched VENDOR_METHODS dict for a single method."""
    original = dict(VENDOR_METHODS)
    patched = dict(original)
    patched[method_name] = dict(vendor_mocks)
    return patched


class TestMarketAwareRouting:
    """Test that route_to_vendor filters vendors by market."""

    @patch("dataflows.interface.VENDOR_LIMITERS", {})
    @patch("dataflows.interface.load_file_cache", return_value=None)
    @patch("dataflows.interface.get_config")
    def test_cn_ticker_routes_to_tushare(self, mock_config, *_):
        mock_tushare = MagicMock(return_value="Date,Open\n2025-01-01,10.0\n")
        mock_yfinance = MagicMock(return_value="Date,Open\n2025-01-01,150.0\n")

        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "tushare,yfinance"},
            "tool_vendors": {},
        }

        patched = _mock_vendor_methods("get_stock_data", {
            "tushare": mock_tushare,
            "yfinance": mock_yfinance,
        })

        with patch.dict("dataflows.interface.VENDOR_METHODS", patched):
            result = route_to_vendor("get_stock_data", "000001.SZ", "2025-01-01", "2025-03-31")

        mock_tushare.assert_called_once()
        mock_yfinance.assert_not_called()
        assert "10.0" in result

    @patch("dataflows.interface.VENDOR_LIMITERS", {})
    @patch("dataflows.interface.load_file_cache", return_value=None)
    @patch("dataflows.interface.get_config")
    def test_us_ticker_routes_to_yfinance(self, mock_config, *_):
        mock_tushare = MagicMock(return_value="Date,Open\n2025-01-01,10.0\n")
        mock_yfinance = MagicMock(return_value="Date,Open\n2025-01-01,150.0\n")

        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance,tushare"},
            "tool_vendors": {},
        }

        patched = _mock_vendor_methods("get_stock_data", {
            "tushare": mock_tushare,
            "yfinance": mock_yfinance,
        })

        with patch.dict("dataflows.interface.VENDOR_METHODS", patched):
            result = route_to_vendor("get_stock_data", "AAPL", "2025-01-01", "2025-03-31")

        mock_yfinance.assert_called_once()
        mock_tushare.assert_not_called()
        assert "150.0" in result

    @patch("dataflows.interface.VENDOR_LIMITERS", {})
    @patch("dataflows.interface.load_file_cache", return_value=None)
    @patch("dataflows.interface.get_config")
    def test_cn_fallback_stays_in_cn_vendors(self, mock_config, *_):
        from dataflows.tushare_provider import TushareRateLimitError

        mock_tushare = MagicMock(side_effect=TushareRateLimitError("rate limited"))
        mock_akshare = MagicMock(return_value="Date,Open\n2025-01-01,10.0\n")
        mock_yfinance = MagicMock(return_value="Date,Open\n2025-01-01,999.0\n")

        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "tushare"},
            "tool_vendors": {},
        }

        patched = _mock_vendor_methods("get_stock_data", {
            "tushare": mock_tushare,
            "akshare": mock_akshare,
            "yfinance": mock_yfinance,
        })

        with patch.dict("dataflows.interface.VENDOR_METHODS", patched):
            result = route_to_vendor("get_stock_data", "600519.SH", "2025-01-01", "2025-03-31")

        mock_akshare.assert_called_once()
        mock_yfinance.assert_not_called()
        assert "10.0" in result


class TestEmptyResultFallback:
    """Test that empty results trigger fallback to next vendor."""

    @patch("dataflows.interface.VENDOR_LIMITERS", {})
    @patch("dataflows.interface.load_file_cache", return_value=None)
    @patch("dataflows.interface.get_config")
    def test_empty_result_triggers_fallback(self, mock_config, *_):
        mock_tushare = MagicMock(return_value="No data found for 000001.SZ")
        mock_akshare = MagicMock(return_value="Date,Open\n2025-01-01,10.0\n")

        mock_config.return_value = {
            "data_vendors": {"core_stock_apis": "tushare,akshare"},
            "tool_vendors": {},
        }

        patched = _mock_vendor_methods("get_stock_data", {
            "tushare": mock_tushare,
            "akshare": mock_akshare,
        })

        with patch.dict("dataflows.interface.VENDOR_METHODS", patched):
            result = route_to_vendor("get_stock_data", "000001.SZ", "2025-01-01", "2025-03-31")

        mock_akshare.assert_called_once()
        assert "10.0" in result


class TestGlobalNewsNoFilter:
    """Test that get_global_news is market-agnostic."""

    @patch("dataflows.interface.VENDOR_LIMITERS", {})
    @patch("dataflows.interface.get_config")
    def test_global_news_not_filtered_by_market(self, mock_config):
        mock_yf_news = MagicMock(return_value="Global news report...\nMarket summary\nMore details\n")

        mock_config.return_value = {
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {},
        }

        patched = _mock_vendor_methods("get_global_news", {
            "yfinance": mock_yf_news,
        })

        with patch.dict("dataflows.interface.VENDOR_METHODS", patched):
            result = route_to_vendor("get_global_news", "2025-03-28", 7, 5)

        mock_yf_news.assert_called_once()
        assert "Global news" in result
