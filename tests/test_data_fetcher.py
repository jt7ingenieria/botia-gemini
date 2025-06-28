# tests/test_data_fetcher.py
# Pruebas unitarias para el módulo data_fetcher.py.
import pytest
import pandas as pd
from src.data_fetcher import fetch_historical_data

def test_fetch_historical_data():
    symbol = "BTCUSD"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    
    df = fetch_historical_data(symbol, interval, start_date, end_date)
    
    # Verificar que el resultado es un DataFrame de pandas
    assert isinstance(df, pd.DataFrame)
    
    # Verificar que el DataFrame no está vacío
    assert not df.empty
    
    # Verificar las columnas esperadas
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_columns:
        assert col in df.columns
        
    # Verificar que el índice es de tipo DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Verificar el número de filas (5 días de datos)
    assert len(df) == 5
