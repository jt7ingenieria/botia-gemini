import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.indicators import DataProcessor
import joblib
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveLearningTradingSystem')

class LiveLearningTradingSystem:
    """Sistema de trading con aprendizaje continuo en tiempo real"""
    
    def __init__(self, initial_balance=10000, max_drawdown=0.10, commission_rate=0.001, data_processor=None):
        # ... (el resto del constructor se mantiene igual)
        self.data_processor = data_processor if data_processor else DataProcessor()
        # Configuración financiera
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.commission_rate = 0.0001
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
        self.predictor = self._build_initial_predictor()
        self.risk_model = self._build_risk_model()
        self.scaler = StandardScaler()
        self.data_buffer = pd.DataFrame()
        
        # Configuración de aprendizaje
        self.retrain_interval = 30  # Reentrenar cada 30 días
        self.retrain_counter = 0
        self.warmup_period = 200  # Período inicial de calentamiento
        self.lookback_window = 365  # Ventana de datos para reentrenamiento
        
    def _build_initial_predictor(self):
        """Construye el modelo predictivo inicial"""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    
    def _build_risk_model(self):
        """Construye el modelo de gestión de riesgo"""
        return MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            warm_start=True
        )
    
    
    
    def train_predictor(self, data):
        """Entrena el modelo predictivo con datos históricos"""
        # Preprocesar datos
        processed_data = self.data_processor.preprocess(data.copy())
        
        # Preparar características y objetivo
        X = processed_data[['volatility_5', 'volatility_20', 'atr', 'return']].shift(1).dropna()
        y = processed_data['return'].iloc[1:len(X)+1]
        
        # Verificar que tenemos suficientes datos
        if len(X) < 100 or len(y) < 100:
            logger.warning("Datos insuficientes para entrenar el modelo predictivo")
            return False
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo
        self.predictor.fit(X_scaled, y)
        
        # Actualizar versión del modelo
        self.model_version += 1
        self.learning_log.append({
            'period': len(self.data_buffer),
            'model_version': self.model_version,
            'window_size': len(X)
        })
        
        logger.info(f"Modelo predictivo reentrenado (versión {self.model_version}) con {len(X)} muestras")
        return True
    
    def train_risk_model(self):
        """Entrena el modelo de gestión de riesgo con datos de operaciones"""
        if len(self.trade_history) < 20:
            return False
        
        # Preparar datos de entrenamiento
        X = []
        y = []
        
        for trade in self.trade_history:
            if 'features' in trade and 'actual_position_size' in trade:
                X.append(trade['features'])
                # Objetivo: ratio entre tamaño de posición real y tamaño sugerido por Kelly
                if trade['kelly_size'] != 0:
                    y.append(trade['actual_position_size'] / trade['kelly_size'])
                else:
                    # Si kelly_size es 0, esta operación no es útil para entrenar el modelo de riesgo
                    continue
        
        if not X or not y:
            logger.warning("Datos insuficientes o inválidos para entrenar el modelo de riesgo.")
            return False

        if len(X) < 10:
            logger.warning(f"Se requieren al menos 10 muestras para entrenar el modelo de riesgo, pero se encontraron {len(X)}.")
            return False
        
        # Entrenar modelo
        self.risk_model.fit(X, y)
        logger.info(f"Modelo de riesgo reentrenado con {len(X)} muestras")
        return True
    
    def predict_direction(self, current_data):
        """Predice la dirección del mercado usando el modelo actual"""
        # Crear características para predicción
        features = pd.DataFrame({
            'volatility_5': [current_data['volatility_5']],
            'volatility_20': [current_data['volatility_20']],
            'atr': [current_data['atr']],
            'return': [current_data['return']]
        })
        
        # Escalar características
        features_scaled = self.scaler.transform(features)
        
        # Predecir return
        predicted_return = self.predictor.predict(features_scaled)[0]
        
        # Determinar dirección
        if predicted_return > 0.005:  # Umbral para entrada alcista
            return 1
        elif predicted_return < -0.005:  # Umbral para entrada bajista
            return -1
        return 0
    
    def calculate_position_size(self):
        """Calcula el tamaño de posición usando Kelly fraccionado y red neuronal"""
        # Obtener métricas recientes
        recent_trades = self.trade_history[-20:] if len(self.trade_history) > 20 else self.trade_history
        win_rate = 0.5
        avg_win = 1.0
        avg_loss = 1.0
        
        if recent_trades:
            wins = [t for t in recent_trades if t.get('pl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pl', 0) <= 0]
            
            if wins and losses:
                win_rate = len(wins) / len(recent_trades)
                avg_win = np.mean([t['pl'] for t in wins])
                avg_loss = np.abs(np.mean([t['pl'] for t in losses]))
        
        # Cálculo de Kelly
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            win_ratio = avg_win / avg_loss
            kelly_fraction = (win_rate - (1 - win_rate) / win_ratio) if win_ratio > 0 else 0.0
        
        # Aplicar fracción conservadora
        kelly_fraction = max(0.01, min(kelly_fraction * 0.5, 0.2))  # Asegurar un mínimo de 0.01
        
        # Entradas para la red neuronal
        nn_inputs = [
            win_rate,
            self.losing_streak / 10.0,
            self.current_balance / self.max_balance,
            self.volatility * 100,
            self.atr / self.current_balance * 100
        ]
        
        # Factor de ajuste de la red neuronal (0.5-1.5)
        # Si el modelo de riesgo no está entrenado, usar un factor por defecto
        if hasattr(self.risk_model, 'coefs_') and self.risk_model.coefs_ is not None:
            nn_factor = self.risk_model.predict([nn_inputs])[0]
            nn_factor = max(0.5, min(nn_factor, 1.5))
        else:
            nn_factor = 1.0 # Usar un factor por defecto si el modelo no está entrenado
        
        # Tamaño final de posición
        position_size = kelly_fraction * nn_factor
        
        # Asegurar que no se exceda el drawdown máximo y que haya un tamaño mínimo
        # Simplificar max_allowed a un porcentaje fijo del balance para evitar valores extremos
        max_allowed_percentage = 0.05 # Por ejemplo, máximo 5% del balance
        max_allowed_by_balance = self.current_balance * max_allowed_percentage / self.current_balance # Esto es solo max_allowed_percentage
        
        position_size = max(0.001, min(position_size, max_allowed_by_balance)) # Asegurar un mínimo de 0.001 y un máximo del 5%
        
        # Guardar características para entrenamiento futuro
        trade_features = {
            'features': nn_inputs,
            'kelly_size': kelly_fraction,
            'actual_position_size': position_size
        }
        
        return position_size, trade_features
    
    def run_backtest(self, data, save_path="trading_system_state.joblib"):
        """Ejecuta el backtesting con aprendizaje continuo"""
        logger.info("Iniciando backtesting con aprendizaje continuo...")
        data = data.reset_index(drop=True)
        self.data_buffer = data.iloc[:self.warmup_period].copy()
        
        # Entrenamiento inicial
        self.train_predictor(self.data_buffer)
        
        # Bucle principal
        for i in tqdm(range(self.warmup_period, len(data)), desc="Backtesting con aprendizaje"):
            # Agregar nuevo punto de datos
            new_data = data.iloc[i:i+1]
            self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
            
            # Preprocesar datos actuales
            current_data = self.data_processor.preprocess(self.data_buffer).iloc[-1]
            
            # Verificar drawdown máximo
            if self._check_drawdown():
                logger.warning(f"Drawdown máximo alcanzado en el período {i}. Deteniendo backtest.")
                break
            
            # Reentrenamiento periódico
            self.retrain_counter += 1
            if self.retrain_counter >= self.retrain_interval:
                self.retrain_counter = 0
                
                # Reentrenar modelo predictivo
                train_data = self.data_buffer.iloc[-self.lookback_window:] if len(self.data_buffer) > self.lookback_window else self.data_buffer
                self.train_predictor(train_data)
                
                # Reentrenar modelo de riesgo
                self.train_risk_model()
            
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
            
            # Cerrar operación si es necesario
            if close_trade:
                self._close_trade(current_price, close_reason)
        
        # Operación de cobertura
        if self.hedge_trade:
            hedge = self.hedge_trade
            direction = hedge['direction']
            
            # Verificar condiciones de cierre
            close_hedge = False
            
            # TP
            if ((direction > 0 and current_price >= hedge['tp']) or
                (direction < 0 and current_price <= hedge['tp'])):
                close_hedge = True
            
            # SL
            if ((direction > 0 and current_price <= hedge['sl']) or
                (direction < 0 and current_price >= hedge['sl'])):
                close_hedge = True
            
            # Cerrar cobertura
            if close_hedge:
                self._close_hedge_trade(current_price)
    
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
        
        # Tamaño de cobertura (50% de la posición principal)
        hedge_size = min(self.active_trade['size'] * 0.5, self.current_balance * 0.1)
        
        # Calcular SL y TP para cobertura (más agresivos)
        if direction > 0:
            sl = current_price - 3 * self.atr
            tp = current_price + 1.5 * self.atr
        else:
            sl = current_price + 3 * self.atr
            tp = current_price - 1.5 * self.atr
        
        # Comisión
        commission = hedge_size * self.commission_rate
        
        # Crear operación de cobertura
        self.hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': hedge_size,
            'sl': sl,
            'tp': tp,
            'commission': commission
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
