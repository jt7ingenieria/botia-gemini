import numpy as np
import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Módulo 1: Procesamiento de Datos
# =============================================================================
class DataProcessor:
    @staticmethod
    def calculate_heikin_ashi(df):
        """Calcula velas Heikin-Ashi vectorizado."""
        ha_df = df.copy()
        ha_df['ha_close'] = (df[['open', 'high', 'low', 'close']].mean(axis=1))
        
        # Calcular ha_open de forma vectorizada
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2
        
        ha_df['ha_open'] = ha_open
        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
        return ha_df

    @staticmethod
    def add_technical_features(df, window=14):
        """Añade indicadores técnicos con manejo de edge cases."""
        df = df.copy()
        # Retornos logarítmicos
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatilidad
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std() # Añadido
        
        # RSI con manejo de división por cero
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        # Evitar división por cero
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # ATR (Average True Range) - Añadido
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = np.max(np.array([high_low, high_close, low_close]).T, axis=1)
        df['atr'] = df['tr'].rolling(window).mean()
        
        return df.dropna()

    def preprocess(self, df):
        """Pipeline completo de preprocesamiento."""
        ha_df = self.calculate_heikin_ashi(df)
        processed_df = self.add_technical_features(ha_df)
        return processed_df