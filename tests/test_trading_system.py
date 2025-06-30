import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.trading_system import LiveLearningTradingSystem
from src.indicators import DataProcessor
from src.risk_management import RiskManager
from src.execution import TradeManager
from sklearn.neural_network import MLPRegressor

@pytest.fixture
def mock_data_processor():
    mock_dp = MagicMock(spec=DataProcessor)
    mock_dp.preprocess.return_value = pd.DataFrame({
        'open': np.random.rand(101) * 100,
        'high': np.random.rand(101) * 100 + 5,
        'low': np.random.rand(101) * 100 - 5,
        'close': np.random.rand(101) * 100,
        'volume': np.random.randint(1000, 10000, 101),
        'return': np.random.rand(101) * 0.01,
        'volatility_5': np.random.rand(101) * 0.01,
        'volatility_20': np.random.rand(101) * 0.01,
        'atr': np.random.rand(101) * 1.0
    })
    return mock_dp

@pytest.fixture
def mock_risk_manager():
    mock_rm = MagicMock(spec=RiskManager)
    mock_rm.calculate_position_size.return_value = (0.01, {'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01})
    mock_rm.train_risk_model.return_value = True
    return mock_rm

@pytest.fixture
def mock_trade_manager():
    mock_tm = MagicMock(spec=TradeManager)
    return mock_tm

@pytest.fixture
def mock_predictor():
    mock_pred = MagicMock()
    mock_pred.predict.return_value = np.array([0.01]) # Simular una predicción de retorno
    return mock_pred

@pytest.fixture
def mock_mlp_regressor():
    mock_mlp = MagicMock(spec=MLPRegressor)
    return mock_mlp

@pytest.fixture
def trading_system(mock_data_processor, mock_risk_manager, mock_trade_manager, mock_predictor, mock_mlp_regressor):
    with patch('src.indicators.DataProcessor', return_value=mock_data_processor):
        with patch('src.risk_management.RiskManager', return_value=mock_risk_manager):
            with patch('src.execution.TradeManager', return_value=mock_trade_manager):
                with patch('src.trading_system.GradientBoostingRegressor', return_value=mock_predictor):
                    system = LiveLearningTradingSystem(data_processor=mock_data_processor)
                    system.risk_model = mock_mlp_regressor
                    return system

@pytest.fixture
def sample_market_data():
    # Datos de mercado de ejemplo para el backtesting
    data = {
        'open': np.random.uniform(100, 200, 200),
        'high': np.random.uniform(105, 205, 200),
        'low': np.random.uniform(95, 195, 200),
        'close': np.random.uniform(100, 200, 200),
        'volume': np.random.randint(1000, 10000, 200)
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_current_data(sample_market_data):
    # Devuelve una sola fila de datos de mercado preprocesados
    # Esto simula los datos de un solo punto en el tiempo
    processed_data = pd.DataFrame({
        'open': [100], 'high': [103], 'low': [99], 'close': [101],
        'volume': [1000], 'return': [0.01], 'volatility_5': [0.01],
        'volatility_20': [0.01], 'atr': [1.0]
    })
    return processed_data.iloc[0]

def test_trading_system_init(trading_system):
    assert trading_system.initial_balance == 10000
    assert trading_system.current_balance == 10000
    assert trading_system.predictor is not None
    assert trading_system.risk_model is not None
    assert trading_system.scaler is not None
    assert trading_system.data_buffer.empty

def test_train_predictor(trading_system, mock_data_processor, mock_predictor, sample_market_data):
    result = trading_system.train_predictor(sample_market_data)
    assert result is True
    mock_data_processor.preprocess.assert_called_once()
    mock_predictor.fit.assert_called_once()
    assert trading_system.model_version == 2 # Se incrementa después del entrenamiento inicial

def test_train_risk_model(trading_system):
    trading_system.trade_history = [
        {'pl': 10, 'features': [0.5, 0.1, 0.9, 0.02, 0.002], 'kelly_size': 0.01, 'actual_position_size': 0.01, 'type': 'main'} for _ in range(20)
    ]
    result = trading_system.train_risk_model()
    assert result is True
    trading_system.risk_model.fit.assert_called_once()

def test_predict_direction(trading_system, mock_data_processor, mock_predictor, sample_current_data):
    # Asegurarse de que el scaler esté ajustado para la predicción
    trading_system.scaler.fit(pd.DataFrame({
        'volatility_5': [0.01],
        'volatility_20': [0.01],
        'atr': [1.0],
        'return': [0.01]
    }))
    
    direction = trading_system.predict_direction(sample_current_data)
    assert direction in [0, 1, -1]
    mock_predictor.predict.assert_called_once()

def test_run_backtest(trading_system, mock_data_processor, mock_risk_manager, mock_trade_manager, mock_predictor, sample_market_data):
    # Mockear los métodos internos que run_backtest llama
    trading_system.preprocess_data = MagicMock(return_value=sample_market_data.iloc[0]) # Devuelve una Serie
    trading_system.train_predictor = MagicMock(return_value=True)
    trading_system.train_risk_model = MagicMock(return_value=True)
    trading_system.predict_direction = MagicMock(return_value=1) # Siempre predice compra
    trading_system._check_drawdown = MagicMock(return_value=False)
    trading_system._check_reversal = MagicMock(return_value=False)
    trading_system._calculate_trailing_stop = MagicMock(return_value=None)
    trading_system._open_trade = MagicMock()
    trading_system._close_trade = MagicMock()
    trading_system._open_hedge_trade = MagicMock()
    trading_system._close_hedge_trade = MagicMock()
    trading_system.save_state = MagicMock()

    trading_system._calculate_performance_metrics = MagicMock(return_value={'final_balance': 10000.0, 'total_trades': 1})

    results = trading_system.run_backtest(sample_market_data, "test_state.joblib")
    
    assert isinstance(results, dict)
    assert 'final_balance' in results
    trading_system.save_state.assert_called_once_with("test_state.joblib")
    trading_system._calculate_performance_metrics.assert_called_once()

def test_save_load_state(trading_system, sample_market_data):
    # Entrenar el sistema para tener un estado que guardar
    trading_system.train_predictor(sample_market_data.iloc[:100])
    trading_system.train_risk_model()
    trading_system.trade_history = [
        {'pl': 10, 'type': 'main', 'model_version': 1, 'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01},
    ]
    trading_system.current_balance = 10500.0
    trading_system.max_balance = 10500.0
    trading_system.equity_curve = [10000.0, 10500.0]
    
    test_path = "temp_trading_system_state.joblib"

    # Mockear joblib.dump y joblib.load
    mock_state_storage = {}
    with patch('joblib.dump', side_effect=lambda state, path: mock_state_storage.update({path: state})):
        with patch('joblib.load', side_effect=lambda path: mock_state_storage[path]):
            trading_system.save_state(test_path)
            
            new_system = LiveLearningTradingSystem()
            loaded = new_system.load_state(test_path)
            
            assert loaded is True
            assert new_system.current_balance == 10500.0
            assert new_system.max_balance == 10500.0
            assert len(new_system.trade_history) == 1
            assert new_system.trade_history[0]['pl'] == 10
            
            # No necesitamos eliminar el archivo real ya que estamos mockeando
            # os.remove(test_path)
