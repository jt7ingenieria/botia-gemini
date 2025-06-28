# tests/test_indicators.py
# Pruebas unitarias para el módulo indicators.py.
import pytest
import pandas as pd
import numpy as np
from src.indicators import DataProcessor

@pytest.fixture
def sample_dataframe():
    # Crear un DataFrame de ejemplo con datos de precios
    data = {
        'open': [100, 102, 105, 103, 106, 108, 110, 109, 112, 115, 113, 116, 118, 120, 119],
        'high': [103, 106, 107, 105, 108, 111, 112, 111, 114, 117, 115, 118, 120, 122, 121],
        'low': [99, 101, 103, 102, 104, 106, 108, 107, 110, 113, 111, 114, 116, 118, 117],
        'close': [102, 104, 106, 104, 107, 109, 111, 110, 113, 116, 114, 117, 119, 121, 120]
    }
    df = pd.DataFrame(data)
    return df

def test_calculate_heikin_ashi(sample_dataframe):
    processor = DataProcessor()
    ha_df = processor.calculate_heikin_ashi(sample_dataframe)
    
    assert 'ha_open' in ha_df.columns
    assert 'ha_high' in ha_df.columns
    assert 'ha_low' in ha_df.columns
    assert 'ha_close' in ha_df.columns
    
    # Verificar algunos valores iniciales (aproximados)
    assert np.isclose(ha_df['ha_close'].iloc[0], (100+103+99+102)/4)
    assert np.isclose(ha_df['ha_open'].iloc[0], (100+102)/2)
    
    # Verificar que no hay NaNs introducidos por el cálculo de HA
    assert not ha_df[['ha_open', 'ha_high', 'ha_low', 'ha_close']].isnull().any().any()

def test_add_technical_features(sample_dataframe):
    processor = DataProcessor()
    features_df = processor.add_technical_features(sample_dataframe)
    
    assert 'returns' in features_df.columns
    assert 'volatility_5' in features_df.columns
    assert 'volatility_10' in features_df.columns
    assert 'rsi' in features_df.columns
    assert 'macd' in features_df.columns
    assert 'signal' in features_df.columns
    
    # Verificar que no hay NaNs después de dropna()
    assert not features_df.isnull().any().any()
    
    # Verificar que el RSI no tiene valores infinitos o NaN debido a división por cero
    assert not np.isinf(features_df['rsi']).any()
    assert not features_df['rsi'].isnull().any()

def test_preprocess(sample_dataframe):
    processor = DataProcessor()
    processed_df = processor.preprocess(sample_dataframe)
    
    # Verificar que todas las columnas esperadas están presentes
    expected_columns = [
        'open', 'high', 'low', 'close',
        'ha_open', 'ha_high', 'ha_low', 'ha_close',
        'returns', 'volatility_5', 'volatility_10', 'rsi', 'ema12', 'ema26', 'macd', 'signal'
    ]
    for col in expected_columns:
        assert col in processed_df.columns
        
    # Verificar que no hay NaNs en el DataFrame final
    assert not processed_df.isnull().any().any()
