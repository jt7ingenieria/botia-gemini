# src/data_fetcher.py
# Módulo para la obtención de datos de mercado.
# Entradas: Símbolo del activo, intervalo de tiempo, rango de fechas.
# Salidas: DataFrame de pandas con datos históricos (OHLCV).

# src/data_fetcher.py
# Módulo para la obtención de datos de mercado.
# Entradas: Símbolo del activo, intervalo de tiempo, rango de fechas.
# Salidas: DataFrame de pandas con datos históricos (OHLCV).

# src/data_fetcher.py
# Módulo para la obtención de datos de mercado.
# Entradas: Símbolo del activo, intervalo de tiempo, rango de fechas.
# Salidas: DataFrame de pandas con datos históricos (OHLCV).

import pandas as pd
import logging

logger = logging.getLogger(__name__)
import numpy as np
from datetime import datetime

from config import CRYPTO_DATA_CONFIG
from .data_manager.crypto_data_fetcher import CryptoDataFetcher

# Función para obtener datos de mercado reales
def generate_market_data() -> pd.DataFrame:
    """Obtiene datos de mercado reales usando CryptoDataFetcher."""
    fetcher = CryptoDataFetcher(CRYPTO_DATA_CONFIG)
    
    # Fetch historical data for the primary symbol
    # Assuming fetch_historical_data returns a dict with 'data' key
    # and that we only care about the primary symbol for now.
    exchange_name = CRYPTO_DATA_CONFIG['exchange']
    symbol = CRYPTO_DATA_CONFIG['symbol']
    timeframe = CRYPTO_DATA_CONFIG['timeframe']
    start_date_str = CRYPTO_DATA_CONFIG['start_date']
    end_date_str = CRYPTO_DATA_CONFIG['end_date']

    # Convert string dates to datetime objects for fetcher
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')

    # Crear una copia mutable de la configuración y actualizar las fechas
    fetcher_config = CRYPTO_DATA_CONFIG.copy()
    fetcher_config['start_date'] = start_date
    fetcher_config['end_date'] = end_date

    fetcher = CryptoDataFetcher(fetcher_config)
    
    # Initialize exchange (mocked in tests, real in production)
    exchange = fetcher.initialize_exchange(exchange_name)
    if exchange is None:
        logger.error(f"Failed to initialize exchange {exchange_name}. Cannot fetch real data.")
        return pd.DataFrame() # Return empty DataFrame on failure

    # Fetch data
    fetched_data = fetcher.fetch_historical_data(exchange, symbol, timeframe)
    
    if fetched_data and 'data' in fetched_data and not fetched_data['data'].empty:
        df = fetched_data['data']
        logger.info(f"Fetched real market data for {symbol} with shape: {df.shape}")
        return df
    else:
        logger.warning(f"No real market data fetched for {symbol}.")
        return pd.DataFrame()
