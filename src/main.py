# src/main.py
# Punto de entrada principal del bot de trading.
# Orquesta los diferentes módulos.

from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_sma, calculate_rsi
from src.strategy import generate_signals
from src.risk_management import calculate_position_size
from src.execution import execute_order
from src.backtesting import run_backtest
from src.utils import log_message

def main():
    log_message("Iniciando el bot de trading...")
    # Aquí se orquestarán las llamadas a los diferentes módulos.
    # Ejemplo de flujo:
    # 1. Obtener datos históricos
    # 2. Calcular indicadores
    # 3. Generar señales de trading
    # 4. Calcular tamaño de posición y gestionar riesgo
    # 5. Ejecutar órdenes (en modo real o simulación)
    # 6. Realizar backtesting
    log_message("Bot de trading finalizado.")

if __name__ == "__main__":
    main()
