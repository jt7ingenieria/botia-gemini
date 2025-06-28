import pytest
import pandas as pd
import numpy as np
from src.backtesting import ModelEvaluator
from src.strategy import FinancialPredictor # Necesario para mockear
from unittest.mock import MagicMock, patch

@pytest.fixture
def sample_test_dataframe():
    # Crear un DataFrame de ejemplo para pruebas de evaluación
    data = {
        'open': [100, 102, 105, 103, 106, 108, 110, 109, 112, 115, 113, 116, 118, 120, 119],
        'high': [103, 106, 107, 105, 108, 111, 112, 111, 114, 117, 115, 118, 120, 122, 121],
        'low': [99, 101, 103, 102, 104, 106, 108, 107, 110, 113, 111, 114, 116, 118, 117],
        'close': [102, 104, 106, 104, 107, 109, 111, 110, 113, 116, 114, 117, 119, 121, 120]
    }
    df = pd.DataFrame(data)
    return df

def test_model_evaluator_evaluate(sample_test_dataframe):
    # Mockear el predictor para controlar sus predicciones
    mock_predictor = MagicMock(spec=FinancialPredictor)
    
    # Configurar el mock para que devuelva una predicción constante
    # En un escenario real, esto sería más complejo y simularía predicciones realistas
    mock_predictor.predict.return_value = np.array([110]) # Predicción constante para simplificar
    
    evaluator = ModelEvaluator()
    mse = evaluator.evaluate(mock_predictor, sample_test_dataframe, method='ensemble')
    
    # Verificar que el MSE es un número flotante
    assert isinstance(mse, float)
    
    # Verificar que el método predict del mock_predictor fue llamado
    # Se espera que se llame len(test_data) - 10 veces (debido al if i < 10 continue)
    expected_calls = len(sample_test_dataframe) - 10
    assert mock_predictor.predict.call_count == expected_calls