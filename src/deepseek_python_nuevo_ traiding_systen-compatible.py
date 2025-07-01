import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from src.indicators import DataProcessor
from src.strategy import FinancialPredictor
from src.risk_management import AdvancedRiskManager
import joblib
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveLearningTradingSystem')

class Trade:
    """Clase para encapsular la lógica de una operación"""
    def __init__(self, trade_type, entry_price, direction, size, commission, model_version):
        self.type = trade_type
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.commission = commission
        self.model_version = model_version
        self.entry_index = None
        self.sl = None
        self.tp = None
        self.trailing_stop = None
        self.metadata = {}
        
    def calculate_pl(self, exit_price):
        """Calcula el P&L de la operación"""
        if self.direction > 0:  # Long
            return (exit_price - self.entry_price) * (self.size / self.entry_price)
        else:  # Short
            return (self.entry_price - exit_price) * (self.size / self.entry_price)

class LiveLearningTradingSystem:
    """Sistema de trading con aprendizaje continuo en tiempo real"""
    
    def __init__(self, initial_balance=10000, max_drawdown=0.10, commission_rate=0.001, 
                 strategy_config=None, indicators_config=None, risk_manager_config=None):
        
        # Configuración financiera
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_drawdown = max_drawdown
        self.commission_rate = commission_rate
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
        self.indicators_config = indicators_config or {}
        self.strategy_config = strategy_config or {}
        self.risk_manager_config = risk_manager_config or {}

        self.data_processor = DataProcessor(indicators_config=self.indicators_config)
        self.predictor = FinancialPredictor(
            strategy_config=self.strategy_config, 
            indicators_config=self.indicators_config
        )
        self.risk_model = AdvancedRiskManager(config=self.risk_manager_config)
        self.data_buffer = pd.DataFrame()
        
        # Configuración de aprendizaje
        self.retrain_interval = 30  # Reentrenar cada 30 días
        self.retrain_counter = 0
        self.warmup_period = 200  # Período inicial de calentamiento
        self.lookback_window = 365  # Ventana de datos para reentrenamiento
        
    def train_predictor(self, data):
        """Entrena el modelo predictivo con datos históricos"""
        self.predictor.train(data, save_model=False)
        self.model_version += 1
        self.learning_log.append({
            'period': len(self.data_buffer),
            'model_version': self.model_version,
            'window_size': len(data)
        })
        logger.info(f"Modelo predictivo reentrenado (v{self.model_version}) con {len(data)} muestras")
        return True
    
    def predict_direction(self, current_data):
        """Predice la dirección del mercado usando el modelo actual"""
        current_data_df = current_data.to_frame().T
        predicted_price = self.predictor.predict(current_data_df, steps=1, method='ensemble')[0]
        
        if predicted_price > current_data['close'] * 1.005:
            return 1  # Señal alcista
        elif predicted_price < current_data['close'] * 0.995:
            return -1  # Señal bajista
        return 0  # Neutral
    
    def calculate_position_size(self):
        """Calcula el tamaño de posición usando el gestor de riesgo"""
        portfolio_corr = self._calculate_portfolio_correlation()
        market_trend = self._calculate_market_trend()
        
        return self.risk_model.calculate_position_size(
            self.trade_history,
            self.current_balance,
            self.max_balance,
            self.atr,
            self.volatility,
            self.losing_streak,
            self.max_drawdown,
            portfolio_corr,
            market_trend
        )

    def _calculate_portfolio_correlation(self):
        """Calcula la correlación entre activos (simplificado)"""
        # Implementación real usaría datos históricos de múltiples activos
        return 0.0  # Valor temporal para pruebas

    def _calculate_market_trend(self):
        """Determina la fuerza de la tendencia del mercado"""
        if len(self.equity_curve) < 10:
            return 0.5  # Neutral
        
        # Calcular pendiente de los últimos 10 puntos
        x = np.arange(10)
        y = np.array(self.equity_curve[-10:])
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalizar pendiente entre 0-1
        return min(1.0, max(0.0, (slope / (self.initial_balance * 0.001))))

    def run_backtest(self, data, save_path="trading_system_state.joblib"):
        """Ejecuta el backtesting con aprendizaje continuo"""
        logger.info("Iniciando backtesting con aprendizaje continuo...")
        data = data.reset_index(drop=True)
        self.data_buffer = data.iloc[:self.warmup_period].copy()
        
        # Entrenamiento inicial
        self.predictor.train(self.data_buffer, save_model=False)
        
        # Bucle principal
        for i in tqdm(range(self.warmup_period, len(data)), desc="Backtesting"):
            # Agregar nuevo punto de datos
            new_data = data.iloc[i:i+1]
            self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
            
            # Optimizar: mantener solo datos necesarios
            if len(self.data_buffer) > self.lookback_window * 2:
                self.data_buffer = self.data_buffer.iloc[-self.lookback_window:]
            
            # Preprocesamiento incremental
            processed_df = self.data_processor.preprocess(self.data_buffer.copy())
            processed_df = processed_df.dropna()
            
            if processed_df.empty:
                logger.warning(f"No hay datos válidos en el período {i}")
                self.equity_curve.append(self.current_balance)
                continue

            current_data = processed_df.iloc[-1]
            
            # Actualizar métricas de mercado
            self.atr = current_data.get('atr', 0.0)
            self.volatility = current_data.get('volatility_20', 0.0)

            # Verificar drawdown máximo
            if self._check_drawdown():
                logger.warning(f"Drawdown máximo alcanzado en período {i}")
                return self._calculate_performance_metrics()
            
            # Reentrenamiento periódico
            self.retrain_counter += 1
            if self.retrain_counter >= self.retrain_interval:
                self.retrain_counter = 0
                train_data = self.data_buffer.iloc[-self.lookback_window:] 
                self.predictor.train(train_data, save_model=False)
            
            # Predicción y gestión de trades
            direction = self.predict_direction(current_data)
            self._manage_open_trades(current_data['close'])
            
            # Abrir nueva operación principal
            if not self.active_trade and direction != 0:
                self._open_trade(current_data, direction, i)
            
            # Verificar cobertura
            if self.active_trade and not self.hedge_trade:
                if self._check_reversal(current_data['close'], direction):
                    self._open_hedge_trade(current_data, -direction, i)
            
            # Actualizar curva de equity
            self.equity_curve.append(self.current_balance)
        
        # Guardar estado final
        self.save_state(save_path)
        logger.info("Backtesting completado")
        return self._calculate_performance_metrics()
    
    def _manage_open_trades(self, current_price):
        """Gestiona operaciones abiertas"""
        # Operación principal
        if self.active_trade:
            self._update_trailing_stop(current_price)
            
            # Verificar condiciones de cierre
            close_main = self._check_close_conditions(self.active_trade, current_price)
            if close_main and not self.hedge_trade:
                self._close_trade(self.active_trade, current_price, "main_close")
        
        # Operación de cobertura
        if self.hedge_trade:
            close_hedge = self._check_close_conditions(self.hedge_trade, current_price)
            if close_hedge:
                self._close_hedge_trade(current_price)

    def _update_trailing_stop(self, current_price):
        """Actualiza el trailing stop para la operación activa"""
        trade = self.active_trade
        direction = trade.direction
        
        # Precio de activación (1.2 ATR de ganancia)
        activation_level = 1.2 * self.atr
        
        if direction > 0:  # Largo
            unrealized = current_price - trade.entry_price
            if unrealized > activation_level:
                new_stop = current_price - 0.8 * self.atr
                trade.trailing_stop = max(trade.trailing_stop or 0, new_stop)
        else:  # Corto
            unrealized = trade.entry_price - current_price
            if unrealized > activation_level:
                new_stop = current_price + 0.8 * self.atr
                trade.trailing_stop = min(trade.trailing_stop or float('inf'), new_stop)

    def _check_close_conditions(self, trade, current_price):
        """Verifica condiciones de cierre para una operación"""
        direction = trade.direction
        
        # Take Profit
        if (direction > 0 and current_price >= trade.tp) or (direction < 0 and current_price <= trade.tp):
            return True
        
        # Stop Loss
        if (direction > 0 and current_price <= trade.sl) or (direction < 0 and current_price >= trade.sl):
            return True
        
        # Trailing Stop
        if trade.trailing_stop:
            if (direction > 0 and current_price <= trade.trailing_stop) or \
               (direction < 0 and current_price >= trade.trailing_stop):
                return True
        
        return False

    def _open_trade(self, current_data, direction, index):
        """Abre una nueva operación principal"""
        current_price = current_data['close']
        
        # Calcular tamaño de posición
        position_size, metadata = self.calculate_position_size()
        trade_value = max(
            self.current_balance * position_size, 
            self.current_balance * 0.005  # Mínimo 0.5%
        )
        
        # Calcular SL y TP
        sl = self.risk_model.dynamic_stop_loss(
            current_price, current_price, self.atr, self.volatility, 
            "long" if direction > 0 else "short"
        )
        tp = self.risk_model.dynamic_take_profit(
            current_price, current_price, self.atr, self.volatility, 
            "long" if direction > 0 else "short"
        )
        
        # Comisión
        commission = trade_value * self.commission_rate
        
        # Crear operación
        self.active_trade = Trade(
            trade_type='main',
            entry_price=current_price,
            direction=direction,
            size=trade_value,
            commission=commission,
            model_version=self.model_version
        )
        self.active_trade.sl = sl
        self.active_trade.tp = tp
        self.active_trade.entry_index = index
        self.active_trade.metadata = metadata
        
        # Actualizar balance
        self.current_balance -= commission
        logger.info(f"Apertura operación: ${trade_value:.2f} ({'LONG' if direction > 0 else 'SHORT'})")

    def _open_hedge_trade(self, current_data, direction, index):
        """Abre una operación de cobertura"""
        current_price = current_data['close']
        main_trade = self.active_trade
        
        # Calcular pérdida potencial
        if main_trade.direction > 0:
            potential_loss = (main_trade.entry_price - main_trade.sl) * (main_trade.size / main_trade.entry_price)
        else:
            potential_loss = (main_trade.sl - main_trade.entry_price) * (main_trade.size / main_trade.entry_price)
        
        # Beneficio objetivo
        target_profit = abs(potential_loss) + (self.initial_balance * 0.01)
        hedge_value = main_trade.size
        
        # Calcular TP
        required_change = target_profit * (current_price / hedge_value)
        if direction > 0:
            tp = current_price + required_change
            sl = current_price - self.atr
        else:
            tp = current_price - required_change
            sl = current_price + self.atr
        
        # Comisión
        commission = hedge_value * self.commission_rate
        
        # Crear operación
        self.hedge_trade = Trade(
            trade_type='hedge',
            entry_price=current_price,
            direction=direction,
            size=hedge_value,
            commission=commission,
            model_version=self.model_version
        )
        self.hedge_trade.sl = sl
        self.hedge_trade.tp = tp
        self.hedge_trade.entry_index = index
        self.hedge_trade.target_profit = target_profit
        
        # Actualizar balance
        self.current_balance -= commission
        logger.info(f"Apertura cobertura: ${hedge_value:.2f}")

    def _close_trade(self, trade, current_price, reason):
        """Cierra una operación principal"""
        pl = trade.calculate_pl(current_price)
        self.current_balance += trade.size + pl
        
        # Registrar trade
        trade_record = {
            'type': trade.type,
            'entry_price': trade.entry_price,
            'exit_price': current_price,
            'size': trade.size,
            'direction': trade.direction,
            'pl': pl,
            'commission': trade.commission,
            'duration': len(self.equity_curve) - trade.entry_index,
            'close_reason': reason,
            'model_version': trade.model_version,
            'metadata': trade.metadata
        }
        self.trade_history.append(trade_record)
        
        # Actualizar modelo de riesgo
        next_state = self._create_next_state()
        self.risk_model.update_with_trade_result(trade_record, next_state)
        
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
        logger.info(f"Cierre operación: PL ${pl:.2f} ({reason})")
        return trade_record

    def _close_hedge_trade(self, current_price):
        """Cierra una operación de cobertura"""
        trade = self.hedge_trade
        pl = trade.calculate_pl(current_price)
        self.current_balance += trade.size + pl
        
        # Registrar trade
        trade_record = {
            'type': trade.type,
            'entry_price': trade.entry_price,
            'exit_price': current_price,
            'size': trade.size,
            'direction': trade.direction,
            'pl': pl,
            'commission': trade.commission
        }
        self.trade_history.append(trade_record)
        
        # Resetear operación
        self.hedge_trade = None
        logger.info(f"Cierre cobertura: PL ${pl:.2f}")
        return trade_record

    def _create_next_state(self):
        """Crea el estado para el modelo de riesgo"""
        return {
            'trade_history': self.trade_history,
            'current_balance': self.current_balance,
            'max_balance': self.max_balance,
            'atr': self.atr,
            'volatility': self.volatility,
            'losing_streak': self.losing_streak,
            'max_drawdown': self.max_drawdown,
            'portfolio_correlation': self._calculate_portfolio_correlation(),
            'market_trend': self._calculate_market_trend()
        }

    def _check_drawdown(self):
        """Verifica si se ha excedido el drawdown máximo"""
        drawdown = (self.max_balance - self.current_balance) / self.max_balance
        return drawdown >= self.max_drawdown
    
    def _check_reversal(self, current_price, predicted_direction):
        """Verifica si se debe activar una cobertura"""
        if not self.active_trade or self.hedge_trade:
            return False
        
        trade = self.active_trade
        entry_price = trade.entry_price
        
        # Calcular pérdida actual
        if trade.direction > 0:
            loss = max(0, entry_price - current_price)
        else:
            loss = max(0, current_price - entry_price)
        
        return loss > 1.2 * self.atr and predicted_direction != trade.direction

    # ... (Métodos restantes: _calculate_performance_metrics, save_state, load_state, plot_results)
    # Se mantienen iguales pero optimizados para usar la clase Trade

# Función para simular datos financieros
def generate_market_data(num_points=2000, volatility=0.02, trend=0.0001):
    """Genera datos de mercado sintéticos con tendencia y volatilidad"""
    prices = [100]
    for i in range(1, num_points):
        change = trend + volatility * np.random.randn()
        prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, num_points)
    })