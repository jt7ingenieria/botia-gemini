import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.trading_system import LiveLearningTradingSystem
from src.indicators import DataProcessor
from src.risk_management import AdvancedRiskManager
from src.execution import TradeManager
from sklearn.neural_network import MLPRegressor

@pytest.fixture
def mock_data_processor():
    mock_dp = MagicMock(spec=DataProcessor)
    def mock_preprocess_side_effect(df):
        # Simula el comportamiento de preprocess: añade algunas columnas y luego dropna
        processed_df = df.copy()
        processed_df['returns'] = processed_df['close'].pct_change()
        processed_df['volatility_5'] = processed_df['returns'].rolling(5).std()
        processed_df['volatility_20'] = processed_df['returns'].rolling(20).std()
        processed_df['atr'] = 1.0 # Valor dummy
        processed_df['rsi'] = 50.0 # Valor dummy
        processed_df['macd'] = 0.0 # Valor dummy
        processed_df['signal'] = 0.0 # Valor dummy
        processed_df['ha_open'] = processed_df['open']
        processed_df['ha_high'] = processed_df['high']
        processed_df['ha_low'] = processed_df['low']
        processed_df['ha_close'] = processed_df['close']
        return processed_df.dropna()
    mock_dp.preprocess.side_effect = mock_preprocess_side_effect
    return mock_dp

@pytest.fixture
def mock_risk_manager():
    mock_rm = MagicMock(spec=AdvancedRiskManager)
    mock_rm.calculate_position_size.return_value = (0.01, {'features': [], 'kelly_size': 0.01, 'actual_position_size': 0.01})
    mock_rm.atr = 1.0 # Added for TradeExecutor mock
    mock_rm.volatility = 0.01 # Added for TradeExecutor mock
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
    system = LiveLearningTradingSystem(initial_balance=10000, commission_rate=0.001, max_drawdown=0.1, strategy_config={}, indicators_config={}, risk_manager_config={})
    system.data_processor = mock_data_processor
    system.predictor = mock_predictor
    system.predictor.trained = True # Set trained to True for the mock predictor
    system.risk_model = mock_risk_manager # Assign mock_risk_manager directly
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
def sample_current_data():
    # Devuelve una sola fila de datos de mercado preprocesados
    # Esto simula los datos de un solo punto en el tiempo
    processed_data = pd.DataFrame({
        'open': [100], 'high': [103], 'low': [99], 'close': [101],
        'volume': [1000], 'returns': [0.01], 'volatility_5': [0.01],
        'volatility_20': [0.01], 'atr': [1.0], 'rsi': [50.0], 'macd': [0.0], 'signal': [0.0],
        'ha_open': [100], 'ha_high': [103], 'ha_low': [99], 'ha_close': [101]
    })
    return processed_data.iloc[0]

def test_trading_system_init(trading_system):
    assert trading_system.initial_balance == 10000
    assert trading_system.current_balance == 10000
    assert trading_system.predictor is not None
    assert trading_system.risk_model is not None
    assert trading_system.data_buffer.empty

def test_train_predictor(trading_system, mock_data_processor, mock_predictor, sample_market_data):
    result = trading_system.train_predictor(sample_market_data)
    assert result is True
    mock_predictor.train.assert_called_once()
    assert trading_system.model_version == 2 # Se incrementa después del entrenamiento inicial

def test_predict_direction(trading_system, mock_data_processor, mock_predictor, sample_current_data):
    
    
    direction = trading_system.predict_direction(sample_current_data)
    assert direction in [0, 1, -1]
    mock_predictor.predict.assert_called_once()

def test_run_backtest(trading_system, mock_data_processor, mock_risk_manager, mock_trade_manager, mock_predictor, sample_market_data):
    # Mockear los métodos internos que run_backtest llama
    trading_system.preprocess_data = MagicMock(return_value=sample_market_data.iloc[0]) # Devuelve una Serie
    trading_system.train_predictor = MagicMock(return_value=True)
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
            # Manually set the mocked objects for the new_system
            new_system.data_processor = mock_data_processor
            new_system.predictor = mock_predictor
            new_system.risk_model = mock_risk_manager

            loaded = new_system.load_state(test_path)
            
            assert loaded is True
            assert new_system.current_balance == 10500.0
            assert new_system.max_balance == 10500.0
            assert len(new_system.trade_history) == 1
            assert new_system.trade_history[0]['pl'] == 10