"""
Módulo de configuración centralizada para el bot de trading.

Utiliza python-dotenv para cargar variables de entorno desde un archivo .env
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la API
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
API_URL = os.getenv('API_URL', 'https://api.example.com')

# Configuración de trading
MAX_TRADE_AMOUNT = float(os.getenv('MAX_TRADE_AMOUNT', 1000))
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', 1))

# Configuración de logging
import logging
from logging.handlers import RotatingFileHandler

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')
MAX_LOG_SIZE = int(os.getenv('MAX_LOG_SIZE', 10485760))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))

# Configurar logger base
def setup_logger(name: str) -> logging.Logger:
    """
    Configura un logger con handlers para archivo y consola.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Formato común
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para archivo (rotativo)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger