"""Stub: y_finance provider (US stocks — not used in A-share overnight pipeline)."""
import logging
logger = logging.getLogger(__name__)

def get_YFin_data_online(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_stock_stats_indicators_window(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_fundamentals(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_balance_sheet(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_cashflow(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_income_statement(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")

def get_insider_transactions(*args, **kwargs):
    raise NotImplementedError("y_finance not available — US stocks only")
