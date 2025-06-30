import numpy as np
import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from config import RISK_MANAGER_CONFIG # Importar la configuración

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.risk_model = self._build_risk_model()
        self.scaler = StandardScaler() # Scaler para las entradas del modelo de riesgo

    def _build_risk_model(self):
        """Construye el modelo de gestión de riesgo"""
        return MLPRegressor(
            hidden_layer_sizes=RISK_MANAGER_CONFIG["MLP_HIDDEN_LAYER_SIZES"],
            activation=RISK_MANAGER_CONFIG["MLP_ACTIVATION"],
            solver=RISK_MANAGER_CONFIG["MLP_SOLVER"],
            max_iter=RISK_MANAGER_CONFIG["MLP_MAX_ITER"],
            random_state=RISK_MANAGER_CONFIG["MLP_RANDOM_STATE"],
            warm_start=RISK_MANAGER_CONFIG["MLP_WARM_START"]
        )

    def calculate_position_size(self, trade_history, current_balance, max_balance, atr, volatility, losing_streak, max_drawdown, commission_rate):
        """Calcula el tamaño de posición usando Kelly fraccionado y red neuronal"""
        # Obtener métricas recientes
        recent_trades = trade_history[-RISK_MANAGER_CONFIG["RECENT_TRADES_WINDOW"]:] if len(trade_history) > RISK_MANAGER_CONFIG["RECENT_TRADES_WINDOW"] else trade_history
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
        kelly_fraction = max(0.0, min(kelly_fraction * RISK_MANAGER_CONFIG["KELLY_FRACTION_MULTIPLIER"], RISK_MANAGER_CONFIG["KELLY_FRACTION_MAX"]))  # Fracción de Kelly
        
        # Entradas para la red neuronal
        nn_inputs = [
            win_rate,
            losing_streak / 10.0,
            current_balance / max_balance,
            volatility * 100,
            atr / current_balance * 100
        ]
        
        # Factor de ajuste de la red neuronal (0.5-1.5)
        # Asegurarse de que el scaler esté ajustado antes de transformar
        try:
            nn_inputs_scaled = self.scaler.transform([nn_inputs])
        except Exception:
            # Si el scaler no está ajustado (primera vez), ajustarlo con datos de ejemplo o inicializarlo
            # Para la primera ejecución, podemos usar un valor por defecto o ajustar con los primeros datos
            self.scaler.fit(np.array([[0.5, 0.0, 1.0, 0.0, 0.0]]))
            nn_inputs_scaled = self.scaler.transform([nn_inputs])

        nn_factor = self.risk_model.predict(nn_inputs_scaled)[0]
        nn_factor = max(0.5, min(nn_factor, 1.5))
        
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
        if len(trade_history) < RISK_MANAGER_CONFIG["RECENT_TRADES_WINDOW"]:
            return False
        
        # Preparar datos de entrenamiento
        X = []
        y = []
        
        for trade in trade_history:
            if 'features' in trade and 'actual_position_size' in trade and trade['kelly_size'] != 0:
                X.append(trade['features'])
                # Objetivo: ratio entre tamaño de posición real y tamaño sugerido por Kelly
                y.append(trade['actual_position_size'] / trade['kelly_size'])
        
        if len(X) < RISK_MANAGER_CONFIG["MIN_TRAINING_SAMPLES"]:
            return False
        
        # Ajustar el scaler con los datos de entrenamiento antes de transformar
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Entrenar modelo
        self.risk_model.fit(X_scaled, y)
        logger.info(f"Modelo de riesgo reentrenado con {len(X)} muestras")
        return True