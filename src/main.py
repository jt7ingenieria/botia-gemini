import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt # Importar matplotlib

from src.data_fetcher import generate_market_data # Importar la función de generación de datos
from src.trading_system import LiveLearningTradingSystem # Importar la clase principal del sistema
from src.utils import log_message

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    log_message("Iniciando el bot de trading...")

    # Generar datos de mercado
    market_data = generate_market_data(num_points=1000)
    
    # Crear sistema de trading
    trading_system = LiveLearningTradingSystem(commission_rate=0.001)
    
    # Ejecutar backtesting
    results = trading_system.run_backtest(market_data, "trading_system_state.joblib")
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS FINALES DEL BACKTESTING CON APRENDIZAJE CONTINUO")
    print("="*60)
    for k, v in results.items():
        if k == 'model_performance':
            print("\nRendimiento por Versión del Modelo:")
            for ver, metrics in v.items():
                win_rate = metrics['wins'] / metrics['trades'] if metrics['trades'] > 0 else 0
                avg_pl = metrics['pl'] / metrics['trades'] if metrics['trades'] > 0 else 0
                print(f"  Versión {ver}: {metrics['trades']} operaciones, "
                      f"{win_rate*100:.1f}% win rate, PL promedio: ${avg_pl:.2f}")
        elif k == 'learning_log':
            print("\nRegistro de Aprendizaje:")
            for log in v:
                print(f"  Período {log['period']}: Entrenado modelo v{log['model_version']} "
                      f"con {log['window_size']} muestras")
        else:
            if isinstance(v, float):
                print(f"{k.replace('_', ' ').title()}: {v*100 if 'rate' in k or 'drawdown' in k else v:.4f}"
                      f"{'%' if 'rate' in k or 'drawdown' in k else ''}")
            else:
                print(f"{k.replace('_', ' ').title()}: {v}")
    
    # Visualizar resultados
    trading_system.plot_results()
    
    # Mostrar curva de aprendizaje
    learning_log = results['learning_log']
    if learning_log:
        plt.figure(figsize=(10, 6))
        periods = [log['period'] for log in learning_log]
        window_sizes = [log['window_size'] for log in learning_log]
        
        plt.plot(periods, window_sizes, 'o-')
        plt.title('Evolución del Tamaño de la Ventana de Entrenamiento')
        plt.xlabel('Período de Entrenamiento')
        plt.ylabel('Tamaño de Ventana')
        plt.grid(True)
        plt.show()
    
    log_message("Bot de trading finalizado.")

if __name__ == "__main__":
    main()