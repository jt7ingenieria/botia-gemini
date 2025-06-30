import numpy as np
import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, risk_manager_config=None):
        self.risk_manager_config = risk_manager_config if risk_manager_config is not None else {}
        self.risk_model = self._build_risk_model()
        self.scaler = StandardScaler() # Scaler para las entradas del modelo de riesgo

    def _build_risk_model(self):
        """Construye el modelo de gestión de riesgo"""
        return MLPRegressor(
            hidden_layer_sizes=self.risk_manager_config.get("MLP_HIDDEN_LAYER_SIZES", (32, 16)),
            activation=self.risk_manager_config.get("MLP_ACTIVATION", 'relu'),
            solver=self.risk_manager_config.get("MLP_SOLVER", 'adam'),
            max_iter=self.risk_manager_config.get("MLP_MAX_ITER", 1000),
            random_state=self.risk_manager_config.get("MLP_RANDOM_STATE", 42),
            warm_start=self.risk_manager_config.get("MLP_WARM_START", True)
        )

    def calculate_position_size(self, trade_history, current_balance, max_balance, atr, volatility, losing_streak, max_drawdown, commission_rate):
        """Calcula el tamaño de posición usando Kelly fraccionado y red neuronal"""
        # Obtener métricas recientes
        recent_trades_window = self.risk_manager_config.get("RECENT_TRADES_WINDOW", 20)
        recent_trades = trade_history[-recent_trades_window:] if len(trade_history) > recent_trades_window else trade_history
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
        kelly_fraction_multiplier = self.risk_manager_config.get("KELLY_FRACTION_MULTIPLIER", 0.5)
        kelly_fraction_max = self.risk_manager_config.get("KELLY_FRACTION_MAX", 0.2)
        kelly_fraction = max(0.0, min(kelly_fraction * kelly_fraction_multiplier, kelly_fraction_max))  # Fracción de Kelly
        
        # Entradas para la red neuronal
        nn_inputs = [
            win_rate,
            losing_streak / 10.0,
            current_balance / max_balance,
            volatility * 100,
            atr / current_balance * 100
        ]
        
        # Factor de ajuste de la red neuronal (0.5-1.5)
        # Si el modelo de riesgo no está entrenado, usar un factor por defecto
        if hasattr(self.risk_model, 'coefs_') and self.risk_model.coefs_ is not None:
            nn_inputs_scaled = self.scaler.transform([nn_inputs])
            nn_factor = self.risk_model.predict(nn_inputs_scaled)[0]
            nn_factor = max(0.5, min(nn_factor, 1.5))
        else:
            nn_factor = 1.0 # Usar un factor por defecto si el modelo no está entrenado
        
        # Tamaño final de posición
        position_size = kelly_fraction * nn_factor
        
        # Asegurar que no se exceda el drawdown máximo
        max_allowed = max_drawdown / (atr / current_balance) if atr > 0 else 0.1
        position_size = min(position_size, max_allowed)
        
        # Guardar características para entrenamiento futuro
        trade_features = {
            'features': nn_inputs,
            'kelly_size': kelly_fraction,
            'actual_position_size': position_size
        }
        
        return position_size, trade_features

    def train_risk_model(self, trade_history):
        """Entrena el modelo de gestión de riesgo con datos de operaciones"""
        min_training_samples = self.risk_manager_config.get("MIN_TRAINING_SAMPLES", 10)
        if len(trade_history) < self.risk_manager_config.get("RECENT_TRADES_WINDOW", 20):
            return False
        
        # Preparar datos de entrenamiento
        X = []
        y = []
        
        for trade in trade_history:
            if 'features' in trade and 'actual_position_size' in trade and trade['kelly_size'] != 0:
                X.append(trade['features'])
                # Objetivo: ratio entre tamaño de posición real y tamaño sugerido por Kelly
                y.append(trade['actual_position_size'] / trade['kelly_size'])
        
        if len(X) < min_training_samples:
            return False
        
        # Ajustar el scaler con los datos de entrenamiento antes de transformar
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Entrenar modelo
        self.risk_model.fit(X_scaled, y)
        logger.info(f"Modelo de riesgo reentrenado con {len(X)} muestras")
        return True