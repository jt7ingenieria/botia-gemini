import pandas as pd
import logging

from src.data_fetcher import fetch_historical_data
from src.strategy import FinancialPredictor
from src.backtesting import ModelEvaluator
from src.utils import log_message

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    log_message("Iniciando el bot de trading...")

    # Ejemplo de uso del FinancialPredictor y ModelEvaluator
    # NOTA: Necesitarás un archivo 'financial_data.csv' con columnas 'date', 'open', 'high', 'low', 'close', 'volume'
    # Por ahora, usaremos un placeholder o generaremos datos si no existe el archivo.
    try:
        # Intentar cargar datos reales si existen
        data = pd.read_csv('financial_data.csv', parse_dates=['date'], index_col='date')
        log_message("Datos cargados desde financial_data.csv")
    except FileNotFoundError:
        log_message("financial_data.csv no encontrado. Generando datos de ejemplo...")
        # Generar datos de ejemplo si el archivo no existe
        # Usaremos fetch_historical_data para generar datos de ejemplo
        data = fetch_historical_data(symbol="EXAMPLE", interval="1d", start_date="2020-01-01", end_date="2023-12-31")
        # Asegurarse de que las columnas sean 'open', 'high', 'low', 'close', 'volume'
        data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        log_message("Datos de ejemplo generados.")

    predictor = FinancialPredictor()
    
    # Entrenar con una parte de los datos históricos
    train_data = data.iloc[:-100]
    predictor.train(train_data)  
    log_message("Modelo entrenado.")
    
    # Predicción
    historical_for_prediction = data.iloc[-100:]
    forecasts = predictor.predict(historical_for_prediction, steps=5)
    log_message(f"Pronóstico a 5 pasos: {forecasts}")
    
    # Evaluación
    evaluator = ModelEvaluator()
    test_data = data.iloc[-100:]
    mse = evaluator.evaluate(predictor, test_data)
    log_message(f"Error Cuadrático Medio (MSE) en el test: {mse:.4f}")
    
    log_message("Bot de trading finalizado.")

if __name__ == "__main__":
    main()