"""
Configuración centralizada del Agente de Trading Inteligente.

Este módulo contiene todos los parámetros configurables del sistema de trading,
incluyendo configuración general, estrategias, indicadores técnicos y gestión de riesgo.
Toda la configuración sensible se carga desde variables de entorno usando python-dotenv.

Variables de entorno requeridas:
- API_KEY: Clave de API para conexión con exchange
- API_SECRET: Secreto de API para conexión con exchange
- DB_URL: URL de conexión a base de datos (opcional)
- LOG_LEVEL: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- MAX_TRADE_AMOUNT: Cantidad máxima por operación
- RISK_PERCENTAGE: Porcentaje de riesgo por operación

Ejemplo de uso:
    from config import BOT_CONFIG, STRATEGY_CONFIG
    print(BOT_CONFIG['initial_balance'])

Atributos:
    BOT_CONFIG: Dict[str, Any] - Configuración general del bot
        Contiene:
        - num_market_data_points: Puntos de datos históricos a cargar
        - commission_rate: Comisión por operación
        - initial_balance: Balance inicial de simulación
        - trading_system_state_file: Archivo para persistir estado
    
    STRATEGY_CONFIG: Dict[str, Any] - Parámetros de estrategias
        Contiene configuración para:
        - Modelos ARIMA
        - Procesos Gaussianos
        - Simulaciones Monte Carlo
        - Gradient Boosting
    
    INDICATORS_CONFIG: Dict[str, Any] - Configuración de indicadores técnicos
        Contiene parámetros para:
        - Medias móviles
        - RSI
        - MACD
        - Bollinger Bands
"""
from typing import Dict, Any, Final
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Configurar logger principal
def setup_logging():
    """
    Configura el sistema de logging con handlers para consola y archivo.
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger('trading_bot')
    logger.setLevel(LOG_LEVEL)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Handler para archivo (rotativo)
    file_handler = RotatingFileHandler(
        'trading_bot.log',
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# General Bot Configuration
BOT_CONFIG: Dict[str, Any] = {
    "commission_rate": 0.001,
    "initial_balance": 10000,
    "trading_system_state_file": "trading_system_state.joblib",
    "training_period_days": 365, # 1 year for training
    "validation_period_days": 90, # 3 months for validation
}

# Cryptocurrency Data Configuration
CRYPTO_DATA_CONFIG: Dict[str, Any] = {
    "exchange": "okx",
    "symbol": "SOL/USDT",
    "timeframe": "1h",
    "start_date": "2023-01-01 00:00:00", # YYYY-MM-DD HH:MM:SS
    "end_date": "2025-07-01 00:00:00", # YYYY-MM-DD HH:MM:SS
    "max_retries": 5,
    "backoff_base": 2,
    "rate_limit": 1000, # milliseconds
}

# Strategy Models Configuration (from src/strategy.py)
STRATEGY_CONFIG: Dict[str, Any] = {
    "ARIMA_ORDER": (1, 1, 1),
    "ARIMA_N_SPLITS": 5,
    "GP_KERNEL_CONSTANT": 1.0,
    "GP_KERNEL_RBF_LENGTH_SCALE": 1.0,
    "GP_N_RESTARTS_OPTIMIZER": 10,
    "MONTE_CARLO_STEPS": 1,
    "MONTE_CARLO_N_SIMULATIONS": 1000,
    "GRADIENT_BOOSTING_N_ESTIMATORS": 100,
    "GRADIENT_BOOSTING_LEARNING_RATE": 0.1,
    "GRADIENT_BOOSTING_MAX_DEPTH": 3,
    "GRADIENT_BOOSTING_RANDOM_STATE": 42,
}

# Indicators Configuration (from src/indicators.py)
INDICATORS_CONFIG: Dict[str, Any] = {
    "RSI_WINDOW": 14,
    "ATR_WINDOW": 14,
    "VOLATILITY_WINDOWS": [5, 10, 20],
    "MACD_EMA_FAST_SPAN": 12,
    "MACD_EMA_SLOW_SPAN": 26,
    "MACD_SIGNAL_SPAN": 9,
}

# Risk Management Model Configuration (from src/risk_management.py)
RISK_MANAGER_CONFIG: Dict[str, Any] = {
    "STATE_DIM": 7,  # Dimensión del vector de estado para DQN
    "ACTION_DIM": 3, # Número de acciones (conservador, balanceado, agresivo)
    "HIDDEN_DIM": 128, # Tamaño de capas ocultas para DQN
    "LEARNING_RATE": 0.001, # Tasa de aprendizaje para DQN
    "MEMORY_SIZE": 10000, # Tamaño de la memoria de experiencia replay
    "BATCH_SIZE": 64, # Tamaño del batch para entrenamiento DQN
    "GAMMA": 0.95, # Factor de descuento para DQN
    "INITIAL_EPSILON": 1.0, # Epsilon inicial para exploración
    "EPSILON_DECAY": 0.995, # Decaimiento de epsilon
    "MIN_EPSILON": 0.01, # Epsilon mínimo
    "TARGET_UPDATE_FREQ": 50, # Frecuencia de actualización de la red objetivo
    "MIN_TRAINING_SAMPLES": 100, # Mínimo de muestras para empezar a entrenar
    "RECENT_TRADES_WINDOW": 20, # Ventana para calcular métricas de trades recientes
    "KELLY_FRACTION_MULTIPLIER": 0.5, # Multiplicador para la fracción de Kelly
    "KELLY_FRACTION_MAX": 0.2, # Máximo para la fracción de Kelly
}
