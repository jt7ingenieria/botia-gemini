import pytest
import numpy as np
from src.risk_management import RiskManager
from unittest.mock import MagicMock, patch

@pytest.fixture
def risk_manager():
    return RiskManager()

@pytest.fixture
def sample_trade_history():
    # Generar 20 operaciones de ejemplo
    history = []
    for i in range(20):
        pl = 10 if i % 2 == 0 else -5
        history.append({'pl': pl, 'features': [0.5, 0.1, 0.9, 0.02, 0.002], 'kelly_size': 0.01, 'actual_position_size': 0.01})
    return history

def test_risk_manager_init(risk_manager):
    assert risk_manager.risk_model is not None
    assert risk_manager.scaler is not None

def test_train_risk_model(risk_manager, sample_trade_history):
    # Mockear el fit del modelo para evitar entrenamiento real
    with patch.object(risk_manager.risk_model, 'fit') as mock_fit:
        result = risk_manager.train_risk_model(sample_trade_history)
        assert result is True
        mock_fit.assert_called_once()

def test_calculate_position_size(risk_manager, sample_trade_history):
    # Mockear el predict del modelo de riesgo
    with patch.object(risk_manager.risk_model, 'predict', return_value=np.array([1.0])):
        # Mockear el fit del scaler para evitar errores si no se ha entrenado
        with patch.object(risk_manager.scaler, 'fit'):
            # Mockear el transform del scaler
            with patch.object(risk_manager.scaler, 'transform', return_value=np.array([[0.5, 0.0, 1.0, 0.0, 0.0]])):
                position_size, trade_features = risk_manager.calculate_position_size(
                    trade_history=sample_trade_history,
                    current_balance=10000,
                    max_balance=10000,
                    atr=10,
                    volatility=0.01,
                    losing_streak=0,
                    max_drawdown=0.10,
                    commission_rate=0.001
                )
                assert isinstance(position_size, float)
                assert position_size >= 0
                assert 'features' in trade_features
                assert 'kelly_size' in trade_features
                assert 'actual_position_size' in trade_features