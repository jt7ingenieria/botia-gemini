import pytest
import numpy as np
from src.risk_management import AdvancedRiskManager
from unittest.mock import MagicMock, patch

@pytest.fixture
def risk_manager():
    return AdvancedRiskManager()

@pytest.fixture
def sample_trade_history():
    # Generar 20 operaciones de ejemplo
    history = []
    for i in range(20):
        pl = 10 if i % 2 == 0 else -5
        history.append({'pl': pl, 'features': [0.5, 0.1, 0.9, 0.02, 0.002], 'kelly_size': 0.01, 'actual_position_size': 0.01})
    return history

def test_risk_manager_init(risk_manager):
    assert risk_manager.q_network is not None
    assert risk_manager.scaler is not None

def test_train_risk_model(risk_manager, sample_trade_history):
    # Simulate a few updates to trigger training
    for i in range(risk_manager.min_training_samples + 5):
        risk_manager.update_risk_model(
            state=np.random.rand(risk_manager.config.get("STATE_DIM", 8)),
            action=np.random.randint(0, risk_manager.config.get("ACTION_DIM", 3)),
            reward=np.random.rand(),
            next_state=np.random.rand(risk_manager.config.get("STATE_DIM", 8)),
            done=False
        )
    
    # Assert that _replay was called (which is where the actual training happens)
    # This is an internal method, so mocking it might be better for unit testing.
    # For now, we'll just check if the memory has grown and epsilon has decayed.
    assert len(risk_manager.memory) > risk_manager.min_training_samples
    assert risk_manager.epsilon < risk_manager.config.get("INITIAL_EPSILON", 1.0)

def test_calculate_position_size(risk_manager, sample_trade_history):
    # Mock the internal get_risk_action to control the action
    with patch.object(risk_manager, 'get_risk_action', return_value=1): # Assume balanced action
        # Mock the scaler transform method
        with patch.object(risk_manager.scaler, 'transform', return_value=np.array([[0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0]])):
            # Ensure scaler is fitted for the mock to work
            risk_manager.scaler_fitted = True
            
            position_size, metadata = risk_manager.calculate_position_size(
                trade_history=sample_trade_history,
                current_balance=10000,
                max_balance=10000,
                atr=10,
                volatility=0.01,
                losing_streak=0,
                max_drawdown=0.10,
                portfolio_correlation=None,
                market_trend=None
            )
            assert isinstance(position_size, float)
            assert position_size >= 0
            assert 'state' in metadata
            assert 'action' in metadata
            assert 'position_size' in metadata
            assert 'max_position' in metadata

def test_dynamic_stop_loss(risk_manager):
    entry_price = 100
    current_price = 100
    atr = 1.0
    volatility = 0.01
    
    sl_long = risk_manager.dynamic_stop_loss(entry_price, current_price, atr, volatility, "long")
    assert isinstance(sl_long, float)
    assert sl_long < entry_price

    sl_short = risk_manager.dynamic_stop_loss(entry_price, current_price, atr, volatility, "short")
    assert isinstance(sl_short, float)
    assert sl_short > entry_price

def test_dynamic_take_profit(risk_manager):
    entry_price = 100
    current_price = 100
    atr = 1.0
    volatility = 0.01
    
    tp_long = risk_manager.dynamic_take_profit(entry_price, current_price, atr, volatility, "long")
    assert isinstance(tp_long, float)
    assert tp_long > entry_price

    tp_short = risk_manager.dynamic_take_profit(entry_price, current_price, atr, volatility, "short")
    assert isinstance(tp_short, float)
    assert tp_short < entry_price

def test_update_with_trade_result(risk_manager):
    # Mock the internal _create_next_state and _calculate_reward
    with (patch.object(risk_manager, '_create_next_state', return_value=np.zeros(risk_manager.config.get("STATE_DIM", 8))),
          patch.object(risk_manager, '_calculate_reward', return_value=0.1),
          patch.object(risk_manager, 'update_risk_model') as mock_update_risk_model):
        
        risk_manager.current_state = np.ones(risk_manager.config.get("STATE_DIM", 8))
        risk_manager.last_action = 1
        
        trade_result = {'pl': 10, 'closed': True}
        next_state_data = {}
        
        result = risk_manager.update_with_trade_result(trade_result, next_state_data)
        assert result is True
        mock_update_risk_model.assert_called_once()
        assert risk_manager.current_state is None
        assert risk_manager.last_action is None
