import pytest
import pandas as pd
import numpy as np
import os # Importar os
from src.strategy import ARIMAModel, GaussianProcessModel, MonteCarloSimulator, FinancialPredictor
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock, patch
from src.indicators import DataProcessor # Importar DataProcessor

@pytest.fixture
def sample_dataframe():
    # Crear un DataFrame de ejemplo con datos de precios (más grande para TimeSeriesSplit)
    data = {
        'open': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(105, 205, 50),
        'low': np.random.uniform(95, 195, 50),
        'close': np.random.uniform(100, 200, 50),
        'volume': np.random.randint(1000, 10000, 50)
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_processed_dataframe(sample_dataframe):
    # Simular un DataFrame preprocesado para las pruebas
    df = sample_dataframe.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['rsi'] = np.random.rand(len(df)) * 100 # Placeholder
    df['macd'] = np.random.rand(len(df)) # Placeholder
    df['signal'] = np.random.rand(len(df)) # Placeholder
    df['ha_open'] = df['open'] # Placeholder
    df['ha_high'] = df['high'] # Placeholder
    df['ha_low'] = df['low'] # Placeholder
    df['ha_close'] = df['close'] # Placeholder
    return df.dropna()

def test_arima_model_cross_validate():
    y = np.array([i for i in range(100, 200)]) # 100 muestras
    model = ARIMAModel()
    trained_model = model.cross_validate(y, n_splits=5)
    assert trained_model is not None
    assert model.model is not None

def test_gaussian_process_model_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([10, 12, 11, 13, 14])
    model = GaussianProcessModel()
    trained_model = model.fit(X, y)
    assert trained_model.model is not None

def test_monte_carlo_simulator_calibrate():
    returns = pd.Series(np.random.randn(100))
    simulator = MonteCarloSimulator()
    params = simulator.calibrate(returns)
    assert 'mu' in params
    assert 'sigma' in params

def test_monte_carlo_simulator_simulate():
    simulator = MonteCarloSimulator()
    simulator.params = {
        'mu': 0.001,
        'sigma': 0.01,
        'lambda_jump': 0.01,
        'mu_jump': 0.1,
        'sigma_jump': 0.05
    }
    S0 = 100
    prediction = simulator.simulate(S0, steps=1, n_simulations=10)
    assert isinstance(prediction, float)
    assert prediction > 0

def test_financial_predictor_train(sample_dataframe, sample_processed_dataframe):
    with patch('src.strategy.DataProcessor.preprocess', return_value=sample_processed_dataframe) as mock_preprocess:
        with patch('src.strategy.ARIMAModel.cross_validate') as mock_arima_cv:
            with patch('src.strategy.GaussianProcessModel.fit') as mock_gp_fit:
                with patch('src.strategy.BayesianRidge.fit') as mock_bayesian_fit:
                    with patch('src.strategy.MonteCarloSimulator.calibrate') as mock_mc_calibrate:
                        with patch('src.strategy.GradientBoostingRegressor.fit') as mock_ensemble_fit:
                            with patch('src.strategy.FinancialPredictor.save_model') as mock_save_model:
                                
                                predictor = FinancialPredictor(model_path='test_model.joblib')
                                
                                # Configurar los mocks para que simulen el entrenamiento
                                # Mockear los métodos directamente en las instancias de los modelos del predictor
                                mock_arima_model_instance = MagicMock()
                                mock_arima_model_instance.model = MagicMock()
                                mock_arima_model_instance.model.forecast.return_value = np.array([100] * len(sample_processed_dataframe))
                                mock_arima_cv.return_value = mock_arima_model_instance
                                predictor.models['arima'] = mock_arima_model_instance # Asignar el mock al modelo ARIMA
                                
                                mock_gp_model_instance = MagicMock()
                                mock_gp_model_instance.model = MagicMock()
                                mock_gp_model_instance.model.predict.return_value = np.array([100] * len(sample_processed_dataframe))
                                mock_gp_fit.return_value = mock_gp_model_instance
                                predictor.models['gp'] = mock_gp_model_instance # Asignar el mock al modelo GP
                                
                                mock_bayesian_model_instance = MagicMock()
                                mock_bayesian_model_instance.predict.return_value = np.array([100] * len(sample_processed_dataframe))
                                mock_bayesian_fit.return_value = mock_bayesian_model_instance
                                predictor.models['bayesian'] = mock_bayesian_model_instance # Asignar el mock al modelo Bayesian
                                
                                mock_montecarlo_instance = MagicMock()
                                mock_montecarlo_instance.calibrate.return_value = {} # Calibrate no devuelve un modelo
                                mock_montecarlo_instance.simulate.return_value = 100
                                mock_mc_calibrate.return_value = mock_montecarlo_instance
                                predictor.models['montecarlo'] = mock_montecarlo_instance # Asignar el mock al simulador Monte Carlo
                                
                                mock_ensemble_model_instance = MagicMock()
                                mock_ensemble_model_instance.predict.return_value = np.array([100] * len(sample_processed_dataframe))
                                mock_ensemble_fit.return_value = mock_ensemble_model_instance
                                predictor.models['ensemble'] = mock_ensemble_model_instance # Asignar el mock al modelo Ensemble
                                
                                predictor.train(sample_dataframe, save_model=True)
                                
                                assert predictor.trained is True
                                mock_preprocess.assert_called_once()
                                predictor.models['arima'].cross_validate.assert_called_once()
                                predictor.models['gp'].fit.assert_called_once()
                                predictor.models['bayesian'].fit.assert_called_once()
                                predictor.models['montecarlo'].calibrate.assert_called_once()
                                predictor.models['ensemble'].fit.assert_called_once()
                                mock_save_model.assert_called_once()

def test_financial_predictor_predict(sample_dataframe, sample_processed_dataframe):
    predictor = FinancialPredictor(model_path='test_model.joblib')
    predictor.trained = True # Simular que el modelo está entrenado
    predictor.scaler = StandardScaler() # Necesario para transform
    predictor.scaler.fit(sample_processed_dataframe.drop(columns=['close']))
    predictor.data_processor = MagicMock(spec=DataProcessor)
    predictor.data_processor.preprocess.return_value = sample_processed_dataframe
    
    # Mockear los modelos internos para que predict funcione
    predictor.models['arima'].model = MagicMock()
    predictor.models['arima'].model.forecast.return_value = np.array([100])
    predictor.models['gp'].model = MagicMock()
    predictor.models['gp'].model.predict = MagicMock(return_value=np.array([100]))
    predictor.models['bayesian'].predict = MagicMock(return_value=np.array([100]))
    predictor.models['montecarlo'].simulate = MagicMock(return_value=100)
    predictor.models['ensemble'].predict = MagicMock(return_value=np.array([100]))
    
    # Prueba de predicción con ensemble
    predictions = predictor.predict(sample_dataframe, steps=1, method='ensemble')
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1
    
    # Prueba de predicción con arima
    predictions = predictor.predict(sample_dataframe, steps=1, method='arima')
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1

def test_financial_predictor_save_load_model(sample_dataframe):
    predictor = FinancialPredictor(model_path='test_model_save_load.joblib')
    predictor.train(sample_dataframe, save_model=True)
    
    # Verificar que el archivo existe
    assert os.path.exists('test_model_save_load.joblib')
    
    # Cargar el modelo
    loaded_predictor = FinancialPredictor(model_path='test_model_save_load.joblib')
    loaded_predictor.load_model()
    
    assert loaded_predictor.trained is True
    assert isinstance(loaded_predictor.scaler, StandardScaler)
    assert isinstance(loaded_predictor.data_processor, DataProcessor)
    
    # Limpiar el archivo
    os.remove('test_model_save_load.joblib')