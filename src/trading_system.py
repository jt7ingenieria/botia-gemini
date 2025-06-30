import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
from src.indicators import DataProcessor
from src.strategy import FinancialPredictor # Importar FinancialPredictor
from src.risk_management import RiskManager # Importar RiskManager
import joblib
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveLearningTradingSystem')

class LiveLearningTradingSystem:
    """Sistema de trading con aprendizaje continuo en tiempo real"""
    
    def __init__(self, initial_balance=10000, max_drawdown=0.10, commission_rate=0.001, 
                 strategy_config=None, indicators_config=None, risk_manager_config=None):
        
        # Configuración financiera
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.commission_rate = commission_rate # Usar el valor pasado, no el fijo
        self.max_balance = initial_balance
        self.equity_curve = [initial_balance]
        
        # Historial de operaciones
        self.trade_history = []
        self.active_trade = None
        self.hedge_trade = None
        
        # Estado del sistema
        self.losing_streak = 0
        self.winning_streak = 0
        self.atr = 0.0
        self.volatility = 0.0
        self.position_size = 0.0
        self.model_version = 1
        self.learning_log = []
        
        # Modelos
        self.indicators_config = indicators_config if indicators_config is not None else {}
        self.strategy_config = strategy_config if strategy_config is not None else {}
        self.risk_manager_config = risk_manager_config if risk_manager_config is not None else {}

        self.data_processor = DataProcessor(indicators_config=self.indicators_config)
        self.predictor = FinancialPredictor(strategy_config=self.strategy_config, indicators_config=self.indicators_config)
        self.risk_model = RiskManager(risk_manager_config=self.risk_manager_config)
        self.scaler = StandardScaler() # Scaler para las entradas del modelo predictivo
        self.data_buffer = pd.DataFrame()
        
        # Configuración de aprendizaje
        self.retrain_interval = 30  # Reentrenar cada 30 días
        self.retrain_counter = 0
        self.warmup_period = 200  # Período inicial de calentamiento
        self.lookback_window = 365  # Ventana de datos para reentrenamiento
        
    def train_predictor(self, data):
        """Entrena el modelo predictivo con datos históricos"""
        # El FinancialPredictor ya tiene su propio método train
        self.predictor.train(data, save_model=False) # No guardar el modelo en cada reentrenamiento
        
        # Actualizar versión del modelo
        self.model_version += 1
        self.learning_log.append({
            'period': len(self.data_buffer),
            'model_version': self.model_version,
            'window_size': len(data) # Usar el tamaño de los datos de entrenamiento
        })
        
        logger.info(f"Modelo predictivo reentrenado (versión {self.model_version}) con {len(data)} muestras")
        return True
    
    def train_risk_model(self):
        """Entrena el modelo de gestión de riesgo con datos de operaciones"""
        # El RiskManager ya tiene su propio método train_risk_model
        return self.risk_model.train_risk_model(self.trade_history)
    
    def predict_direction(self, current_data):
        """Predice la dirección del mercado usando el modelo actual"""
        # El FinancialPredictor ya tiene su propio método predict
        # Necesitamos pasarle el DataFrame completo para que pueda preprocesar
        # y extraer las características necesarias para la predicción.
        # current_data es una Serie, necesitamos convertirla a DataFrame para que funcione con preprocess
        current_data_df = current_data.to_frame().T # Convertir Serie a DataFrame de 1 fila
        
        # Asumiendo que el predictor predice el precio de cierre futuro
        predicted_price = self.predictor.predict(current_data_df, steps=1, method='ensemble')[0]
        
        # Determinar dirección basándose en la predicción vs el precio actual
        # Esto es una simplificación, una estrategia real usaría más lógica
        if predicted_price > current_data['close'] * 1.005:  # Umbral para entrada alcista (ej. 0.5% de subida)
            return 1
        elif predicted_price < current_data['close'] * 0.995:  # Umbral para entrada bajista (ej. 0.5% de bajada)
            return -1
        return 0
    
    def calculate_position_size(self):
        """Calcula el tamaño de posición usando Kelly fraccionado y red neuronal"""
        # El RiskManager ya tiene su propio método calculate_position_size
        return self.risk_model.calculate_position_size(
            self.trade_history,
            self.current_balance,
            self.max_balance,
            self.atr,
            self.volatility,
            self.losing_streak,
            self.max_drawdown,
            self.commission_rate
        )
    
    def run_backtest(self, data, save_path="trading_system_state.joblib"):
        """Ejecuta el backtesting con aprendizaje continuo"""
        logger.info("Iniciando backtesting con aprendizaje continuo...")
        data = data.reset_index(drop=True)
        self.data_buffer = data.iloc[:self.warmup_period].copy()
        
        # Entrenamiento inicial
        # Asegurarse de que el predictor se entrena con un DataFrame, no una Serie
        self.predictor.train(self.data_buffer, save_model=False)
        
        # Bucle principal
        for i in tqdm(range(self.warmup_period, len(data)), desc="Backtesting con aprendizaje"):
            # Agregar nuevo punto de datos
            new_data = data.iloc[i:i+1]
            self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
            
            # Preprocesar datos actuales
            # Asegurarse de que el data_processor.preprocess recibe un DataFrame con suficiente historial
            # para calcular los indicadores.
            # Usamos una ventana de datos para el preprocesamiento, y luego tomamos la última fila válida.
            processed_window_df = self.data_processor.preprocess(self.data_buffer.iloc[-self.lookback_window:].copy())
            processed_window_df = processed_window_df.dropna() # Eliminar cualquier NaN residual de los indicadores
            
            if processed_window_df.empty:
                logger.warning(f"No valid data after preprocessing at period {i}. Skipping trade decision.")
                self.equity_curve.append(self.current_balance) # Keep equity curve consistent
                continue # Skip this iteration if no valid data

            current_data = processed_window_df.iloc[-1] # Obtener la Serie del último punto válido
            
            # Actualizar ATR y Volatilidad para calculate_position_size
            self.atr = current_data['atr'] if 'atr' in current_data else 0.0
            self.volatility = current_data['volatility_20'] if 'volatility_20' in current_data else 0.0

            # Verificar drawdown máximo
            if self._check_drawdown():
                logger.warning(f"Drawdown máximo alcanzado en el período {i}. Deteniendo backtest.")
                return self._calculate_performance_metrics() # Devolver métricas actuales
                break
            
            # Reentrenamiento periódico
            self.retrain_counter += 1
            if self.retrain_counter >= self.retrain_interval:
                self.retrain_counter = 0
                
                # Reentrenar modelo predictivo
                train_data = self.data_buffer.iloc[-self.lookback_window:] if len(self.data_buffer) > self.lookback_window else self.data_buffer
                self.predictor.train(train_data, save_model=False)
                
                # Reentrenar modelo de riesgo
                self.risk_model.train_risk_model(self.trade_history)
            
            # Predecir dirección del mercado
            direction = self.predict_direction(current_data)
            
            # Gestionar operaciones existentes
            self._manage_open_trades(current_data['close'])
            
            # Abrir nueva operación principal si no hay activa
            if not self.active_trade and direction != 0:
                self._open_trade(current_data, direction, i)
            
            # Verificar si se necesita cobertura
            if self.active_trade and not self.hedge_trade:
                if self._check_reversal(current_data['close'], direction):
                    self._open_hedge_trade(current_data, -direction, i)
            
            # Actualizar curva de equity
            self.equity_curve.append(self.current_balance)
        
        # Guardar estado final del sistema
        self.save_state(save_path)
        
        logger.info("Backtesting completado")
        return self._calculate_performance_metrics()
    
    def _manage_open_trades(self, current_price):
        """Gestiona operaciones abiertas"""
        # Operación principal
        if self.active_trade:
            trade = self.active_trade
            direction = trade['direction']
            
            # Actualizar trailing stop
            trade['trailing_stop'] = self._calculate_trailing_stop(
                current_price, trade['entry_price'], direction)
            
            # Verificar condiciones de cierre
            close_trade = False
            close_reason = ""
            
            # TP
            if ((direction > 0 and current_price >= trade['tp']) or
                (direction < 0 and current_price <= trade['tp'])):
                close_trade = True
                close_reason = "take_profit"
            
            # SL
            if ((direction > 0 and current_price <= trade['sl']) or
                (direction < 0 and current_price >= trade['sl'])):
                close_trade = True
                close_reason = "stop_loss"
            
            # Trailing stop
            if trade['trailing_stop']:
                if ((direction > 0 and current_price <= trade['trailing_stop']) or
                    (direction < 0 and current_price >= trade['trailing_stop'])):
                    close_trade = True
                    close_reason = "trailing_stop"
            
            # Cerrar operación principal si es necesario (a menos que la cobertura la cierre)
            # La lógica de cierre de la operación principal por cobertura se maneja en la sección de cobertura
            if close_trade and not self.hedge_trade:
                self._close_trade(current_price, close_reason)
            
        # Operación de cobertura
        if self.hedge_trade:
            hedge = self.hedge_trade
            direction = hedge['direction']
            
            close_hedge_by_tp = False
            close_hedge_by_sl = False

            # Verificar si el TP de la cobertura se activa
            if ((direction > 0 and current_price >= hedge['tp']) or
                (direction < 0 and current_price <= hedge['tp'])):
                close_hedge_by_tp = True
            
            # Verificar si el SL de la cobertura se activa
            if ((direction > 0 and current_price <= hedge['sl']) or
                (direction < 0 and current_price >= hedge['sl'])):
                close_hedge_by_sl = True

            if close_hedge_by_tp or close_hedge_by_sl:
                # Calcular P&L de la cobertura si se cierra ahora
                if direction > 0:
                    hedge_pl = (current_price - hedge['entry_price']) * (hedge['size'] / hedge['entry_price'])
                else:
                    hedge_pl = (hedge['entry_price'] - current_price) * (hedge['size'] / hedge['entry_price'])

                # Calcular P&L no realizado de la operación principal
                main_trade_unrealized_pl = 0
                if self.active_trade: # Asegurarse de que la operación principal aún esté activa
                    main_direction = self.active_trade['direction']
                    main_entry_price = self.active_trade['entry_price']
                    main_size = self.active_trade['size']
                    if main_direction > 0:
                        main_trade_unrealized_pl = (current_price - main_entry_price) * (main_size / main_entry_price)
                    else:
                        main_trade_unrealized_pl = (main_entry_price - current_price) * (main_size / main_entry_price)
                
                combined_pl = hedge_pl + main_trade_unrealized_pl

                # Si el TP de la cobertura se activa Y el P&L combinado cumple el objetivo
                if close_hedge_by_tp and combined_pl >= hedge['target_profit_from_hedge']:
                    logger.info(f"Objetivo de ganancia combinado alcanzado. Cerrando operación principal y de cobertura.")
                    if self.active_trade: # Asegurarse de que la operación principal aún esté activa
                        self._close_trade(current_price, "hedge_profit_target") # Cerrar operación principal
                    self._close_hedge_trade(current_price) # Cerrar operación de cobertura
                elif close_hedge_by_sl: # Si el SL de la cobertura se activa, cerrar solo la cobertura
                    logger.info(f"SL de cobertura alcanzado. Cerrando operación de cobertura.")
                    self._close_hedge_trade(current_price)
                # Si el TP de la cobertura se activa pero el P&L combinado no cumple el objetivo, la cobertura permanece abierta
    
    def _open_trade(self, current_data, direction, index):
        """Abre una nueva operación principal"""
        current_price = current_data['close']
        position_size, trade_features = self.calculate_position_size()
        trade_value = max(self.current_balance * position_size, self.current_balance * 0.005) # Asegurar un tamaño mínimo de operación (0.5% del balance)
        
        # Calcular SL y TP dinámicos
        sl_multiplier = 1.5
        tp_multiplier = 2.5
        sl_distance = sl_multiplier * self.atr
        tp_distance = tp_multiplier * self.atr
        
        if direction > 0:  # Operación larga
            sl = current_price - sl_distance
            tp = current_price + tp_distance
        else:  # Operación corta
            sl = current_price + sl_distance
            tp = current_price - tp_distance
        
        # Comisión
        commission = trade_value * self.commission_rate
        
        # Crear operación
        self.active_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': trade_value,
            'position_size': position_size,
            'sl': sl,
            'tp': tp,
            'trailing_stop': None,
            'commission': commission,
            'features': trade_features['features'],
            'kelly_size': trade_features['kelly_size'],
            'actual_position_size': position_size,
            'model_version': self.model_version
        }
        
        # Actualizar balance
        self.current_balance -= commission
        
    
    def _open_hedge_trade(self, current_data, direction, index):
        """Abre una operación de cobertura"""
        current_price = current_data['close']

        # Calcular pérdida potencial de la operación principal si alcanza su SL
        main_trade = self.active_trade
        if main_trade['direction'] > 0:  # Operación principal es larga
            potential_main_loss = (main_trade['entry_price'] - main_trade['sl']) * (main_trade['size'] / main_trade['entry_price'])
        else:  # Operación principal es corta
            potential_main_loss = (main_trade['sl'] - main_trade['entry_price']) * (main_trade['size'] / main_trade['entry_price'])

        # Asegurar que la pérdida potencial sea un valor positivo
        potential_main_loss = abs(potential_main_loss)

        # Beneficio objetivo de la cobertura: cubrir pérdida principal + 1% del balance inicial
        target_profit_from_hedge = potential_main_loss + (self.initial_balance * 0.01)

        # Tamaño de la operación de cobertura: igual al tamaño de la operación principal
        # Esto simplifica el cálculo del TP, asumiendo que el valor nominal es el mismo.
        hedge_value = main_trade['size']

        # Calcular el cambio de precio necesario para que la cobertura alcance el beneficio objetivo
        # P&L = (exit_price - entry_price) * (size / entry_price) para largo
        # P&L = (entry_price - exit_price) * (size / entry_price) para corto
        # Despejando (exit_price - entry_price) = P&L * (entry_price / size)
        # Para la cobertura, el 'entry_price' es current_price y 'size' es hedge_value
        # Entonces, (exit_price - current_price) = target_profit_from_hedge * (current_price / hedge_value)
        
        # Asegurarse de que hedge_value no sea cero para evitar división por cero
        if hedge_value == 0:
            logger.warning("Hedge value is zero, cannot open hedge trade.")
            return

        required_price_change = target_profit_from_hedge * (current_price / hedge_value)

        # Calcular TP para la cobertura
        if direction > 0:  # Cobertura es larga
            tp = current_price + required_price_change
        else:  # Cobertura es corta
            tp = current_price - required_price_change

        # SL para la cobertura (más agresivo, por ejemplo, 1 ATR)
        if direction > 0:
            sl = current_price - 1 * self.atr
        else:
            sl = current_price + 1 * self.atr

        # Comisión
        commission = hedge_value * self.commission_rate

        # Crear operación de cobertura
        self.hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': hedge_value,
            'sl': sl,
            'tp': tp,
            'commission': commission,
            'main_trade_potential_loss': potential_main_loss, # Para depuración/análisis
            'target_profit_from_hedge': target_profit_from_hedge # Para depuración/análisis
        }

        # Actualizar balance
        self.current_balance -= commission
    
    def _close_trade(self, current_price, reason):
        """Cierra la operación principal"""
        trade = self.active_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        # Actualizar balance
        self.current_balance += pl
        
        # Registrar trade
        trade_record = {
            'type': 'main',
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'],
            'direction': direction,
            'pl': pl,
            'commission': trade['commission'],
            'duration': len(self.equity_curve) - trade['entry_index'],
            'close_reason': reason,
            'model_version': trade['model_version'],
            'features': trade['features'],
            'kelly_size': trade['kelly_size'],
            'actual_position_size': trade['actual_position_size']
        }
        self.trade_history.append(trade_record)
        
        # Actualizar rachas
        if pl > 0:
            self.winning_streak += 1
            self.losing_streak = 0
        else:
            self.losing_streak += 1
            self.winning_streak = 0
        
        # Actualizar balance máximo
        if self.current_balance > self.max_balance:
            self.max_balance = self.current_balance
        
        # Resetear operación
        self.active_trade = None
        
        return trade_record
    
    def _close_hedge_trade(self, current_price):
        """Cierra la operación de cobertura"""
        trade = self.hedge_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        # Actualizar balance
        self.current_balance += pl
        
        # Registrar trade
        trade_record = {
            'type': 'hedge',
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'],
            'direction': direction,
            'pl': pl,
            'commission': trade['commission']
        }
        self.trade_history.append(trade_record)
        
        # Resetear operación
        self.hedge_trade = None
        
        return trade_record
    
    def _check_drawdown(self):
        """Verifica si se ha excedido el drawdown máximo"""
        drawdown = (self.max_balance - self.current_balance) / self.max_balance
        return drawdown >= self.max_drawdown
    
    def _check_reversal(self, current_price, predicted_direction):
        """Verifica si se debe activar una operación de cobertura"""
        if self.active_trade is None or self.hedge_trade is not None:
            return False
        
        direction = self.active_trade['direction']
        entry_price = self.active_trade['entry_price']
        
        # Calcular pérdida actual (si es positiva, es una pérdida)
        if direction > 0:  # Operación larga
            current_loss = max(0, entry_price - current_price) # Pérdida si el precio baja
        else:  # Operación corta
            current_loss = max(0, current_price - entry_price) # Pérdida si el precio sube
        
        # Condiciones para cobertura:
        # 1. Pérdida > 1.2 ATR
        # 2. Señal de reversión fuerte
        # 3. Sin cobertura activa
        
        return current_loss > 1.2 * self.atr and predicted_direction != direction
    
    def _calculate_trailing_stop(self, current_price, entry_price, direction):
        """Calcula trailing stop dinámico"""
        # Precio de activación (cuando la ganancia supera 1.2 ATR)
        activation_level = 1.2 * self.atr
        new_trailing_stop = self.active_trade.get('trailing_stop', None)
        
        if direction > 0:  # Largo
            unrealized_profit = current_price - entry_price
            if unrealized_profit > activation_level:
                new_trailing_stop = max(
                    new_trailing_stop or 0, 
                    current_price - 0.8 * self.atr
                )
        else:  # Corto
            unrealized_profit = entry_price - current_price
            if unrealized_profit > activation_level:
                new_trailing_stop = min(
                    new_trailing_stop or float('inf'), 
                    current_price + 0.8 * self.atr
                )
        
        return new_trailing_stop
    
    def _calculate_performance_metrics(self):
        """Calcula métricas de rendimiento finales"""
        if not self.trade_history:
            return {}
        
        # Filtrar operaciones principales
        main_trades = [t for t in self.trade_history if t['type'] == 'main']
        hedge_trades = [t for t in self.trade_history if t['type'] == 'hedge']
        
        # Cálculos básicos
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        max_equity = max(self.equity_curve)
        min_equity = min(self.equity_curve)
        max_drawdown = (max_equity - min_equity) / max_equity
        
        # Métricas de operaciones principales
        if main_trades:
            wins = [t for t in main_trades if t['pl'] > 0]
            losses = [t for t in main_trades if t['pl'] <= 0]
            
            win_rate = len(wins) / len(main_trades)
            avg_win = np.mean([t['pl'] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t['pl']) for t in losses]) if losses else 0
            profit_factor = sum(t['pl'] for t in wins) / sum(abs(t['pl']) for t in losses) if losses else float('inf')
            
            # Agrupar por versión del modelo
            model_performance = {}
            for trade in main_trades:
                ver = trade['model_version']
                if ver not in model_performance:
                    model_performance[ver] = {'trades': 0, 'wins': 0, 'pl': 0}
                
                model_performance[ver]['trades'] += 1
                model_performance[ver]['pl'] += trade['pl']
                if trade['pl'] > 0:
                    model_performance[ver]['wins'] += 1
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            model_performance = {}
        
        # Métricas de cobertura
        hedge_profit = sum(t['pl'] for t in hedge_trades) if hedge_trades else 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(main_trades),
            'hedge_trades': len(hedge_trades),
            'hedge_profit': hedge_profit,
            'losing_streak': self.losing_streak,
            'winning_streak': self.winning_streak,
            'model_versions': self.model_version,
            'model_performance': model_performance,
            'learning_log': self.learning_log
        }
    
    def evaluate_predictor_performance(self, predictor, test_df, method='ensemble'):
        """Evalúa el modelo sin alterar datos originales."""
        test_data = test_df.copy()
        true_values = []
        predictions = []
        
        for i in range(len(test_data)):
            if i < 10:  # Mínimo histórico para predicción
                continue
                
            historical = test_data.iloc[:i]
            true = test_data.iloc[i]['close']
            
            try:
                pred = predictor.predict(historical, steps=1, method=method)[0]
                true_values.append(true)
                predictions.append(pred)
                
                # Actualizar con valor real (no predicho) para siguiente iteración
                test_data.loc[test_data.index[i], 'close'] = true
            except Exception as e:
                logger.error(f"Evaluation error at step {i}: {str(e)}")
        
        return mean_squared_error(true_values, predictions)

    def save_state(self, path):
        """Guarda el estado actual del sistema"""
        state = {
            'predictor': self.predictor,
            'risk_model': self.risk_model,
            'scaler': self.scaler,
            'current_balance': self.current_balance,
            'max_balance': self.max_balance,
            'trade_history': self.trade_history,
            'model_version': self.model_version,
            'learning_log': self.learning_log,
            'equity_curve': self.equity_curve
        }
        joblib.dump(state, path)
        logger.info(f"Estado del sistema guardado en {path}")
    
    def load_state(self, path):
        """Carga un estado guardado del sistema"""
        if not os.path.exists(path):
            logger.warning(f"Archivo {path} no encontrado")
            return False
        
        state = joblib.load(path)
        self.predictor = state['predictor']
        self.risk_model = state['risk_model']
        self.scaler = state['scaler']
        self.current_balance = state['current_balance']
        self.max_balance = state['max_balance']
        self.trade_history = state['trade_history']
        self.model_version = state['model_version']
        self.learning_log = state['learning_log']
        self.equity_curve = state['equity_curve']
        
        logger.info(f"Estado del sistema cargado desde {path}")
        return True
    
    def plot_results(self):
        """Visualiza los resultados del backtesting"""
        if not self.trade_history:
            print("No hay operaciones para visualizar")
            return
        
        plt.figure(figsize=(18, 12))
        
        # Curva de Equity
        plt.subplot(2, 3, 1)
        plt.plot(self.equity_curve)
        plt.title('Curva de Equity')
        plt.xlabel('Períodos')
        plt.ylabel('Balance')
        plt.grid(True)
        
        # Drawdown
        plt.subplot(2, 3, 2)
        rolling_max = pd.Series(self.equity_curve).cummax()
        drawdown = (rolling_max - self.equity_curve) / rolling_max
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Períodos')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        # Evolución del rendimiento por versión del modelo
        if self.learning_log:
            plt.subplot(2, 3, 3)
            versions = sorted(set([log['model_version'] for log in self.learning_log]))
            win_rates = []
            for ver in versions:
                trades = [t for t in self.trade_history if t.get('model_version') == ver]
                if trades:
                    wins = [t for t in trades if t['pl'] > 0]
                    win_rates.append(len(wins) / len(trades) if trades else 0)
                else:
                    win_rates.append(0)
            
            plt.plot(versions, win_rates, 'o-')
            plt.title('Evolución del Win Rate por Versión del Modelo')
            plt.xlabel('Versión del Modelo')
            plt.ylabel('Win Rate')
            plt.grid(True)
        
        # Distribución de retornos
        plt.subplot(2, 3, 4)
        returns = [t['pl'] / t['size'] for t in self.trade_history if t['type'] == 'main' and t.get('size', 0) != 0]
        if returns:
            plt.hist(returns, bins=30, alpha=0.7)
            plt.axvline(0, color='red', linestyle='--')
            plt.title('Distribución de Retornos por Operación')
            plt.xlabel('Retorno %')
            plt.ylabel('Frecuencia')
            plt.grid(True)
        else:
            plt.title('Distribución de Retornos por Operación (No hay datos)')
            plt.text(0.5, 0.5, 'No hay operaciones con tamaño > 0', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.grid(True)
        
        # Razones de cierre
        plt.subplot(2, 3, 5)
        main_trades = [t for t in self.trade_history if t['type'] == 'main']
        close_reasons = [t['close_reason'] for t in main_trades]
        reason_counts = pd.Series(close_reasons).value_counts()
        plt.pie(reason_counts, labels=reason_counts.index, autopct='%1.1f%%')
        plt.title('Razones de Cierre de Operaciones')
        
        # Tamaño de posición vs Kelly
        plt.subplot(2, 3, 6)
        if main_trades:
            kelly_sizes = [t.get('kelly_size', 0) for t in main_trades]
            actual_sizes = [t.get('position_size', 0) for t in main_trades]
            
            plt.plot(kelly_sizes, label='Kelly Fraction')
            plt.plot(actual_sizes, label='Tamaño Real')
            plt.title('Evolución del Tamaño de Posición')
            plt.xlabel('Operación')
            plt.ylabel('Tamaño (% balance)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Función para simular datos financieros
def generate_market_data(num_points=2000, volatility=0.02, trend=0.0001):
    """Genera datos de mercado sintéticos con tendencia y volatilidad"""
    prices = [100]
    for i in range(1, num_points):
        # Movimiento base + tendencia + volatilidad
        change = trend + volatility * np.random.randn()
        prices.append(prices[-1] * (1 + change))
    
    # Crear DataFrame con OHLC
    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, num_points)
    })
    return df
