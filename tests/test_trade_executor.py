import pytest
from unittest.mock import Mock
from src.trade_executor import TradeExecutor

class MockRiskManager:
    def __init__(self):
        self.atr = 1.0  # Dummy ATR
        self.volatility = 0.01  # Dummy Volatility

    def dynamic_stop_loss(self, *args, **kwargs):
        return 90.0  # Dummy SL

    def dynamic_take_profit(self, *args, **kwargs):
        return 110.0  # Dummy TP

    def update_with_trade_result(self, *args, **kwargs):
        pass  # Mock this method

def test_trade_executor_initialization():
    risk_model = MockRiskManager()
    executor = TradeExecutor(initial_balance=10000, commission_rate=0.001, max_drawdown=0.1, risk_model=risk_model)
    assert executor.initial_balance == 10000
    assert executor.current_balance == 10000
    assert executor.commission_rate == 0.001
    assert executor.max_drawdown == 0.1
    assert executor.max_balance == 10000
    assert executor.trade_history == []
    assert executor.active_trade is None
    assert executor.hedge_trade is None
    assert executor.losing_streak == 0
    assert executor.winning_streak == 0

def test_trade_executor_open_trade():
    risk_model = MockRiskManager()
    executor = TradeExecutor(initial_balance=10000, commission_rate=0.001, max_drawdown=0.1, risk_model=risk_model)
    
    current_data = {'close': 100.0}
    direction = 1
    index = 10
    position_size = 0.01
    metadata = {'test': 'data'}
    atr = 1.0
    volatility = 0.01

    executor._open_trade(current_data, direction, index, position_size, metadata, atr, volatility)

    assert executor.active_trade is not None
    assert executor.active_trade['entry_price'] == 100.0
    assert executor.active_trade['direction'] == 1
    assert executor.active_trade['sl'] == 90.0
    assert executor.active_trade['tp'] == 110.0
    assert executor.current_balance < 10000 # Commission deducted

def test_trade_executor_close_trade():
    risk_model = MockRiskManager()
    executor = TradeExecutor(initial_balance=10000, commission_rate=0.001, max_drawdown=0.1, risk_model=risk_model)
    
    # Open a trade first
    current_data = {'close': 100.0}
    direction = 1
    index = 10
    position_size = 0.01
    metadata = {'test': 'data'}
    atr = 1.0
    volatility = 0.01
    executor._open_trade(current_data, direction, index, position_size, metadata, atr, volatility)

    initial_balance_after_open = executor.current_balance
    
    # Close the trade
    exit_price = 105.0 # Profitable close
    reason = "TP Hit"
    equity_curve_len = 20
    trade_record = executor._close_trade(exit_price, reason, equity_curve_len)

    assert executor.active_trade is None
    assert len(executor.trade_history) == 1
    assert executor.trade_history[0]['pl'] > 0
    assert executor.current_balance > initial_balance_after_open
    assert trade_record['close_reason'] == reason

def test_trade_executor_close_losing_trade():
    risk_model = MockRiskManager()
    executor = TradeExecutor(initial_balance=10000, commission_rate=0.001, max_drawdown=0.1, risk_model=risk_model)
    
    # Open a trade first
    current_data = {'close': 100.0}
    direction = 1
    index = 10
    position_size = 0.01
    metadata = {'test': 'data'}
    atr = 1.0
    volatility = 0.01
    executor._open_trade(current_data, direction, index, position_size, metadata, atr, volatility)

    initial_balance_after_open = executor.current_balance
    
    # Close the trade at a loss
    exit_price = 95.0 # Losing close
    reason = "SL Hit"
    equity_curve_len = 20
    trade_record = executor._close_trade(exit_price, reason, equity_curve_len)

    assert executor.active_trade is None
    assert len(executor.trade_history) == 1
    assert executor.trade_history[0]['pl'] < 0
    assert executor.current_balance < initial_balance_after_open
    assert trade_record['close_reason'] == reason