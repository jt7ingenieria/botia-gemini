import pytest
from src.execution import TradeManager
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

@pytest.fixture
def mock_trading_system():
    mock_system = MagicMock()
    mock_system.current_balance = 10000.0
    mock_system.max_balance = 10000.0
    mock_system.commission_rate = 0.001
    mock_system.atr = 10.0
    mock_system.volatility = 0.01
    mock_system.losing_streak = 0
    mock_system.winning_streak = 0
    mock_system.model_version = 1
    mock_system.trade_history = []
    mock_system.active_trade = None
    mock_system.hedge_trade = None
    mock_system.equity_curve = [10000.0]
    
    mock_system.risk_manager = MagicMock()
    mock_system.risk_manager.calculate_position_size.return_value = (0.01, {'features': [0.5, 0.0, 1.0, 0.01, 0.001], 'kelly_size': 0.01, 'actual_position_size': 0.01})
    
    mock_system._calculate_trailing_stop = MagicMock(return_value=None)
    mock_system._check_reversal = MagicMock(return_value=False)
    
    return mock_system

@pytest.fixture
def trade_manager(mock_trading_system):
    return TradeManager(mock_trading_system)

@pytest.fixture
def sample_current_data():
    return pd.Series({'close': 1000.0, 'open': 990.0, 'high': 1010.0, 'low': 980.0, 'volume': 1000}, name=100)

def test_open_trade(trade_manager, mock_trading_system, sample_current_data):
    trade_manager.open_trade(sample_current_data, 1, sample_current_data.name)
    
    assert mock_trading_system.active_trade is not None
    assert mock_trading_system.active_trade['direction'] == 1
    assert mock_trading_system.current_balance < 10000.0
    mock_trading_system.risk_manager.calculate_position_size.assert_called_once()

def test_manage_open_trades_close_trade(trade_manager, mock_trading_system):
    mock_trading_system.active_trade = {
        'entry_index': 50, 'entry_price': 990.0, 'direction': 1, 'size': 100.0,
        'position_size': 0.01, 'sl': 980.0, 'tp': 1010.0, 'trailing_stop': None,
        'commission': 1.0, 'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01, 'model_version': 1
    }
    mock_trading_system.current_balance = 9999.0
    mock_trading_system.equity_curve = [10000.0, 9999.0]
    mock_trading_system.max_balance = 10000.0

    with patch.object(trade_manager, '_close_trade') as mock_close_trade:
        dummy_data = pd.Series({'close': 1010.0})
        trade_manager.manage_open_trades(dummy_data, 100)
        mock_close_trade.assert_called_once_with(1010.0, "take_profit")

def test_manage_open_trades_open_hedge(trade_manager, mock_trading_system, sample_current_data):
    mock_trading_system.active_trade = {
        'entry_index': 50, 'entry_price': 1000.0, 'direction': 1, 'size': 1000.0,
        'position_size': 0.1, 'sl': 900.0, 'tp': 1100.0, 'trailing_stop': None,
        'commission': 1.0, 'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01
    }
    mock_trading_system.hedge_trade = None
    mock_trading_system._check_reversal.return_value = True

    with patch.object(trade_manager, 'open_hedge_trade') as mock_open_hedge:
        trade_manager.manage_open_trades(sample_current_data, sample_current_data.name)
        mock_open_hedge.assert_called_once_with(sample_current_data, -1, sample_current_data.name)

def test_close_trade(trade_manager, mock_trading_system):
    mock_trading_system.active_trade = {
        'entry_index': 50, 'entry_price': 990.0, 'direction': 1, 'size': 100.0,
        'position_size': 0.01, 'sl': 980.0, 'tp': 1010.0, 'trailing_stop': None,
        'commission': 1.0, 'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01, 'model_version': 1
    }
    mock_trading_system.current_balance = 9999.0
    mock_trading_system.equity_curve = [10000.0] * 51
    mock_trading_system.max_balance = 10000.0

    trade_manager._close_trade(1005.0, "take_profit")
    
    assert mock_trading_system.active_trade is None
    assert len(mock_trading_system.trade_history) == 1
    assert mock_trading_system.trade_history[0]['pl'] > 0
    assert mock_trading_system.winning_streak == 1
    assert mock_trading_system.losing_streak == 0

def test_open_hedge_trade(trade_manager, mock_trading_system, sample_current_data):
    mock_trading_system.active_trade = { 'size': 1000.0 }
    mock_trading_system.atr = 10.0
    mock_trading_system.current_balance = 9999.0

    trade_manager.open_hedge_trade(sample_current_data, -1, 101)
    
    assert mock_trading_system.hedge_trade is not None
    assert mock_trading_system.hedge_trade['direction'] == -1
    assert mock_trading_system.current_balance < 9999.0

def test_close_hedge_trade(trade_manager, mock_trading_system):
    mock_trading_system.hedge_trade = {
        'entry_index': 101, 'entry_price': 1000.0, 'direction': -1, 'size': 500.0,
        'sl': 1030.0, 'tp': 990.0, 'commission': 0.5
    }
    mock_trading_system.current_balance = 9998.5
    mock_trading_system.trade_history = []

    trade_manager._close_hedge_trade(995.0)
    
    assert mock_trading_system.hedge_trade is None
    assert len(mock_trading_system.trade_history) == 1
    assert mock_trading_system.trade_history[0]['pl'] > 0

