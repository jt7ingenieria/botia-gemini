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
    def __init__(self, indicators_config=None):
        self.indicators_config = indicators_config if indicators_config is not None else {}

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

    def add_technical_features(self, df):
        """Añade indicadores técnicos con manejo de edge cases."""
        df = df.copy()
        # Retornos logarítmicos
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatilidad
        for vol_window in self.indicators_config.get("VOLATILITY_WINDOWS", [5, 10, 20]):
            df[f'volatility_{vol_window}'] = df['returns'].rolling(vol_window).std()
        
        # RSI con manejo de división por cero
        rsi_window = self.indicators_config.get("RSI_WINDOW", 14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(rsi_window).mean()
        avg_loss = loss.rolling(rsi_window).mean()
        
        # Evitar división por cero
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_ema_fast_span = self.indicators_config.get("MACD_EMA_FAST_SPAN", 12)
        macd_ema_slow_span = self.indicators_config.get("MACD_EMA_SLOW_SPAN", 26)
        macd_signal_span = self.indicators_config.get("MACD_SIGNAL_SPAN", 9)

        df['ema_fast'] = df['close'].ewm(span=macd_ema_fast_span, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=macd_ema_slow_span, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal'] = df['macd'].ewm(span=macd_signal_span, adjust=False).mean()

        # ATR (Average True Range)
        atr_window = self.indicators_config.get("ATR_WINDOW", 14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = np.max(np.array([high_low, high_close, low_close]).T, axis=1)
        df['atr'] = df['tr'].rolling(atr_window).mean()
        
        return df

    def preprocess(self, df):
        """Pipeline completo de preprocesamiento."""
        ha_df = self.calculate_heikin_ashi(df)
        processed_df = self.add_technical_features(ha_df)
        return processed_df