# src/data_fetcher.py
# Módulo para la obtención de datos de mercado.
# Entradas: Símbolo del activo, intervalo de tiempo, rango de fechas.
# Salidas: DataFrame de pandas con datos históricos (OHLCV).

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_historical_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene datos históricos de un activo.
    Por ahora, devuelve un DataFrame de pandas de ejemplo.
    """
    # Generar fechas de ejemplo
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Simplificación: asume intervalo diario para generar datos
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    
    # Generar datos OHLCV de ejemplo
    np.random.seed(42) # Para reproducibilidad
    open_prices = np.random.uniform(100, 200, len(dates))
    high_prices = open_prices + np.random.uniform(1, 5, len(dates))
    low_prices = open_prices - np.random.uniform(1, 5, len(dates))
    close_prices = open_prices + np.random.uniform(-2, 2, len(dates))
    volumes = np.random.randint(1000, 10000, len(dates))

    data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    data.set_index('Date', inplace=True)
    return data
