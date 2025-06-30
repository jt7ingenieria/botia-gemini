import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt # Importar matplotlib

from .data_fetcher import generate_market_data # Importar la función de generación de datos
from .trading_system import LiveLearningTradingSystem # Importar la clase principal del sistema
from .utils import log_message
from config import BOT_CONFIG, STRATEGY_CONFIG, INDICATORS_CONFIG, RISK_MANAGER_CONFIG # Importar todas las configuraciones

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config_override=None, run_optimization=False, optimization_phase='all'):
    # Crear copias mutables de las configuraciones
    bot_config = BOT_CONFIG.copy()
    strategy_config = STRATEGY_CONFIG.copy()
    indicators_config = INDICATORS_CONFIG.copy()
    risk_manager_config = RISK_MANAGER_CONFIG.copy()

    # Aplicar sobrescrituras si existen
    if config_override:
        bot_config.update(config_override.get("BOT_CONFIG", {}))
        strategy_config.update(config_override.get("STRATEGY_CONFIG", {}))
        indicators_config.update(config_override.get("INDICATORS_CONFIG", {}))
        risk_manager_config.update(config_override.get("RISK_MANAGER_CONFIG", {}))

    log_message("Iniciando el bot de trading...")

    # Generar datos de mercado
    market_data = generate_market_data(num_points=bot_config["num_market_data_points"])
    
    if run_optimization:
        from optimize_advanced import AdvancedOptimizer
        logger.info(f"Running advanced optimization for phase: {optimization_phase}...")
        
        # Combine all configs into a single dictionary for the optimizer
        full_initial_config = {
            "BOT_CONFIG": bot_config,
            "STRATEGY_CONFIG": strategy_config,
            "INDICATORS_CONFIG": indicators_config,
            "RISK_MANAGER_CONFIG": risk_manager_config,
        }
        
        optimizer = AdvancedOptimizer(full_initial_config)
        best_params = optimizer.optimize(market_data) # Pass market_data for regime detection
        
        if best_params:
            logger.info(f"Optimization completed. Best parameters: {best_params}")
            # Apply best parameters to the config_override for the actual backtest run
            for key, value in best_params.items():
                if key.startswith('bot__'):
                    bot_config[key.replace('bot__', '')] = value
                elif key.startswith('strategy__'):
                    # Special handling for ARIMA_ORDER tuple
                    if key == 'strategy__arima_p':
                        current_arima_order = strategy_config.get("ARIMA_ORDER", (0,0,0))
                        strategy_config["ARIMA_ORDER"] = (value, current_arima_order[1], current_arima_order[2])
                    elif key == 'strategy__arima_d':
                        current_arima_order = strategy_config.get("ARIMA_ORDER", (0,0,0))
                        strategy_config["ARIMA_ORDER"] = (current_arima_order[0], value, current_arima_order[2])
                    elif key == 'strategy__arima_q':
                        current_arima_order = strategy_config.get("ARIMA_ORDER", (0,0,0))
                        strategy_config["ARIMA_ORDER"] = (current_arima_order[0], current_arima_order[1], value)
                    else:
                        strategy_config[key.replace('strategy__', '')] = value
                elif key.startswith('indicators__'):
                    indicators_config[key.replace('indicators__', '')] = value
                elif key.startswith('risk_manager__'):
                    risk_manager_config[key.replace('risk_manager__', '')] = value
        else:
            logger.warning("Optimization did not return best parameters. Running backtest with default/initial configs.")

    # Crear sistema de trading, pasando todas las configuraciones
    trading_system = LiveLearningTradingSystem(
        commission_rate=bot_config["commission_rate"],
        initial_balance=bot_config["initial_balance"], # Ensure initial_balance is passed
        strategy_config=strategy_config,
        indicators_config=indicators_config,
        risk_manager_config=risk_manager_config
    )
    
    # Ejecutar backtesting
    results = trading_system.run_backtest(market_data, bot_config["trading_system_state_file"])
    
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
    return results # Devolver los resultados del backtesting

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run the trading bot or its optimization.')
    parser.add_argument('--optimize', action='store_true', 
                        help='Run hyperparameter optimization before backtesting.')
    parser.add_argument('--phase', type=str, default='all', 
                        choices=['bot', 'strategy', 'indicators', 'risk_manager', 'all'],
                        help='Optimization phase to run (bot, strategy, indicators, risk_manager, all). Only applicable with --optimize.')
    args = parser.parse_args()

    if args.optimize:
        main(run_optimization=True, optimization_phase=args.phase)
    else:
        main()
