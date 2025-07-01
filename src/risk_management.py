import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    def __init__(self, state_dim=8, action_dim=3, hidden_dim=128, config=None):
        """
        Sistema avanzado de gestión de riesgo con Deep Reinforcement Learning
        
        :param state_dim: Dimensión del vector de estado (default: 8)
        :param action_dim: Número de acciones (0: conservador, 1: balanceado, 2: agresivo)
        :param hidden_dim: Tamaño de capas ocultas
        :param config: Configuración adicional
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inicializando Risk Manager en dispositivo: {self.device}")
        
        # Modelo DQN para gestión de riesgo
        self.q_network = self._build_dqn(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = self._build_dqn(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.get("LEARNING_RATE", 0.001))
        
        # Experiencia replay
        self.memory = deque(maxlen=self.config.get("MEMORY_SIZE", 10000))
        self.batch_size = self.config.get("BATCH_SIZE", 64)
        self.gamma = self.config.get("GAMMA", 0.95)
        self.epsilon = self.config.get("INITIAL_EPSILON", 1.0)
        self.epsilon_decay = self.config.get("EPSILON_DECAY", 0.995)
        self.min_epsilon = self.config.get("MIN_EPSILON", 0.01)
        self.target_update_freq = self.config.get("TARGET_UPDATE_FREQ", 50)
        self.step_count = 0
        
        # Scaler para normalización
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.min_training_samples = self.config.get("MIN_TRAINING_SAMPLES", 100)
        
        # Estado actual
        self.current_state = None
        self.last_action = None

    def _build_dqn(self, state_dim, action_dim, hidden_dim):
        """Construye la red neuronal DQN"""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def get_risk_action(self, state):
        """Obtener acción de gestión de riesgo usando política epsilon-greedy"""
        if not self.scaler_fitted:
            return random.randrange(3)
        
        try:
            state_scaled = self.scaler.transform([state])
            state_tensor = torch.FloatTensor(state_scaled).to(self.device)
            
            if np.random.rand() <= self.epsilon:
                return random.randrange(3)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
        except Exception as e:
            logger.error(f"Error en get_risk_action: {e}")
            return random.randrange(3)

    def update_risk_model(self, state, action, reward, next_state, done):
        """Actualizar modelo con nueva experiencia"""
        try:
            # Si es la primera experiencia, inicializar el scaler
            if not self.scaler_fitted:
                self.scaler.partial_fit([state])
                self.scaler_fitted = True
            
            # Escalar estados
            state_scaled = self.scaler.transform([state])[0]
            next_state_scaled = self.scaler.transform([next_state])[0] if next_state is not None else None
            
            # Guardar experiencia
            self.memory.append((state_scaled, action, reward, next_state_scaled, done))
            
            # Entrenamiento periódico
            if len(self.memory) > self.min_training_samples:
                self._replay()
                
                # Actualizar target network periódicamente
                self.step_count += 1
                if self.step_count % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Actualizar epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                
            return True
        except Exception as e:
            logger.error(f"Error en update_risk_model: {e}")
            return False

    def _replay(self):
        """Entrenamiento con experiencia replay"""
        if len(self.memory) < self.batch_size:
            return
        
        try:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor([ns for ns in next_states if ns is not None]).to(self.device)
            dones = torch.BoolTensor([d for d, ns in zip(dones, next_states) if ns is not None]).to(self.device)
            
            # Calcular Q-values actuales
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Calcular Q-values objetivo (solo para estados con siguiente estado válido)
            with torch.no_grad():
                next_q = torch.zeros(len(batch), device=self.device)
                if len(next_states) > 0:
                    next_q_values = self.target_network(next_states).max(1)[0]
                    next_q[torch.tensor([i for i, ns in enumerate(next_states) if ns is not None])] = next_q_values
                target_q = rewards + self.gamma * next_q * (~dones)
            
            # Calcular pérdida
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            # Optimizar
            self.optimizer.zero_grad()
            loss.backward()
            # Clip de gradientes para estabilidad
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            logger.debug(f"Entrenamiento DQN - Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")
            return loss.item()
        except Exception as e:
            logger.error(f"Error en _replay: {e}")
            return None

    def calculate_position_size(self, trade_history, current_balance, max_balance, 
                               atr, volatility, losing_streak, max_drawdown, 
                               portfolio_correlation, market_trend=None):
        """
        Calcular tamaño de posición con gestión de riesgo avanzada
        
        :param trade_history: Historial de operaciones recientes
        :param current_balance: Balance actual de la cuenta
        :param max_balance: Balance máximo histórico
        :param atr: Average True Range actual
        :param volatility: Volatilidad del mercado actual
        :param losing_streak: Racha actual de operaciones perdedoras
        :param max_drawdown: Drawdown máximo permitido
        :param portfolio_correlation: Correlaciones entre activos en cartera
        :param market_trend: Fuerza de la tendencia del mercado (0-1)
        :return: Tamaño de posición y metadatos
        """
        try:
            # Crear vector de estado
            state = self._create_state_vector(
                trade_history, current_balance, max_balance, 
                atr, volatility, losing_streak, max_drawdown, 
                portfolio_correlation, market_trend
            )
            
            # Guardar estado actual para actualización posterior
            self.current_state = state
            
            # Obtener acción de gestión de riesgo
            action = self.get_risk_action(state)
            self.last_action = action
            
            logger.debug(f"Estado: {state}, Acción: {action}")

            # Calcular tamaño de posición basado en acción
            if action == 0:  # Modo conservador
                position_size = self._conservative_position(trade_history, current_balance, volatility)
            elif action == 1:  # Modo balanceado
                position_size = self._balanced_position(trade_history, current_balance, volatility)
            else:  # Modo agresivo (solo en condiciones óptimas)
                position_size = self._aggressive_position(trade_history, current_balance, volatility)
            
            logger.debug(f"Tamaño de posición inicial: {position_size}")
            
            # Aplicar restricciones de riesgo
            max_position = self._calculate_max_position(
                current_balance, volatility, max_drawdown, atr
            )
            position_size = min(position_size, max_position)
            
            # Guardar metadatos
            metadata = {
                "state": state,
                "action": action,
                "position_size": position_size,
                "max_position": max_position
            }
            
            return position_size, metadata
        except Exception as e:
            logger.error(f"Error en calculate_position_size: {e}")
            # Fallback: posición conservadora
            return min(0.01 * current_balance, 0.05 * current_balance), {"error": str(e)}

    def _create_state_vector(self, trade_history, current_balance, max_balance, 
                            atr, volatility, losing_streak, max_drawdown, 
                            portfolio_correlation, market_trend=None):
        """
        Crear vector de estado para DQN
        
        Componentes del estado:
        0: Tasa de aciertos recientes
        1: Racha de pérdidas normalizada
        2: Ratio de balance actual/máximo
        3: Volatilidad del mercado
        4: Ratio ATR/precio
        5: Fuerza de tendencia del mercado
        6: Riesgo de correlación de cartera
        """
        # Calcular métricas de rendimiento
        recent_trades = trade_history[-20:] if trade_history and len(trade_history) > 20 else trade_history or []
        win_rate = self._calculate_win_rate(recent_trades)
        
        # Calcular riesgo de correlación
        correlation_risk = 0.0
        if portfolio_correlation:
            correlations = [c for c in portfolio_correlation.values() if not np.isnan(c)]
            correlation_risk = np.std(correlations) if correlations else 0.0
        
        # Calcular fuerza de tendencia si no se proporciona
        if market_trend is None:
            market_trend = self._market_trend_strength(recent_trades)
        
        return np.array([
            win_rate,  # Tasa de aciertos
            losing_streak / 10.0,  # Racha de pérdidas normalizada
            current_balance / max_balance,  # Ratio de balance
            volatility * 100,  # Volatilidad porcentual
            atr / current_balance * 100 if current_balance > 0 else 0.1,  # Ratio ATR/balance
            market_trend,  # Fuerza de tendencia
            correlation_risk  # Riesgo de correlación
        ], dtype=np.float32)

    def _calculate_win_rate(self, trades):
        """Calcular tasa de aciertos en operaciones recientes"""
        if not trades:
            return 0.5
        
        wins = [t for t in trades if t.get('pl', 0) > 0]
        return len(wins) / len(trades)

    def _market_trend_strength(self, trades):
        """Estimar fuerza de tendencia basada en operaciones recientes"""
        if len(trades) < 5:
            return 0.5
        
        # Calcular relación de operaciones ganadoras en tendencia
        trend_trades = [t for t in trades[-5:] if t.get('in_trend', False)]
        if not trend_trades:
            return 0.5
        
        wins = [t for t in trend_trades if t.get('pl', 0) > 0]
        return len(wins) / len(trend_trades)

    def _calculate_kelly(self, trades):
        """Calcular fracción de Kelly basada en rendimiento histórico"""
        if not trades or len(trades) < 10:
            return 0.05  # Valor por defecto
        
        wins = [t for t in trades if t.get('pl', 0) > 0]
        losses = [t for t in trades if t.get('pl', 0) <= 0]
        
        if not wins or not losses:
            return 0.05
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['pl'] for t in wins])
        avg_loss = np.abs(np.mean([t['pl'] for t in losses]))
        
        if avg_loss == 0:
            return 0.05
        
        kelly = (win_rate - (1 - win_rate) / avg_loss) if avg_loss > 0 else 0.0
        return max(0.01, min(kelly, 0.2))  # Limitar entre 1% y 20%

    def _calculate_max_position(self, balance, volatility, max_drawdown, atr):
        """
        Calcular posición máxima permitida por restricciones de riesgo
        """
        # Límite basado en drawdown máximo
        drawdown_limit = max_drawdown * balance
        
        # Límite basado en volatilidad
        volatility_limit = balance * 0.1 / max(volatility, 0.01)
        
        # Límite basado en ATR
        atr_limit = balance * 0.05 / (atr if atr > 0 else 0.01)
        
        # Tomar el límite más conservador
        return min(drawdown_limit, volatility_limit, atr_limit)

    def _conservative_position(self, trades, balance, volatility):
        """Tamaño de posición conservador"""
        kelly = self._calculate_kelly(trades)
        volatility_factor = max(0.3, 1.0 - volatility * 15)  # Más conservador en alta volatilidad
        return min(0.01 * balance, kelly * balance * volatility_factor)

    def _balanced_position(self, trades, balance, volatility):
        """Tamaño de posición balanceado"""
        kelly = self._calculate_kelly(trades)
        volatility_factor = max(0.5, 1.0 - volatility * 10)
        return min(0.05 * balance, kelly * balance * volatility_factor)

    def _aggressive_position(self, trades, balance, volatility):
        """Tamaño de posición agresivo (solo en condiciones óptimas)"""
        if volatility > 0.05:  # No usar modo agresivo en alta volatilidad
            return self._balanced_position(trades, balance, volatility)
        
        kelly = self._calculate_kelly(trades)
        win_streak = self._calculate_win_streak(trades)
        streak_bonus = min(1.5, 1.0 + win_streak * 0.1)
        return min(0.1 * balance, kelly * balance * streak_bonus)

    def _calculate_win_streak(self, trades):
        """Calcular racha actual de operaciones ganadoras"""
        if not trades:
            return 0
        
        streak = 0
        for trade in reversed(trades):
            if trade.get('pl', 0) > 0:
                streak += 1
            else:
                break
        return streak

    def dynamic_stop_loss(self, entry_price, current_price, atr, volatility, position_type="long"):
        """
        Calcular stop-loss dinámico
        
        :param entry_price: Precio de entrada
        :param current_price: Precio actual
        :param atr: Average True Range actual
        :param volatility: Volatilidad del mercado
        :param position_type: 'long' o 'short'
        :return: Precio de stop-loss
        """
        try:
            # Factor de riesgo basado en volatilidad
            risk_factor = 1.5 + (volatility * 10)
            
            # Para posiciones largas
            if position_type == "long":
                # ATR-based stop
                atr_stop = current_price - (atr * risk_factor)
                
                # Porcentaje de drawdown
                drawdown_stop = entry_price * (1 - (0.01 + volatility * 0.5))
                
                # Tomar el stop más cercano al precio actual
                return max(atr_stop, drawdown_stop)
            
            # Para posiciones cortas
            else:
                atr_stop = current_price + (atr * risk_factor)
                drawdown_stop = entry_price * (1 + (0.01 + volatility * 0.5))
                return min(atr_stop, drawdown_stop)
        except Exception as e:
            logger.error(f"Error en dynamic_stop_loss: {e}")
            # Fallback: stop-loss del 2%
            return entry_price * 0.98 if position_type == "long" else entry_price * 1.02

    def dynamic_take_profit(self, entry_price, current_price, atr, volatility, position_type="long"):
        """
        Calcular take-profit dinámico
        
        :param entry_price: Precio de entrada
        :param current_price: Precio actual
        :param atr: Average True Range actual
        :param volatility: Volatilidad del mercado
        :param position_type: 'long' o 'short'
        :return: Precio de take-profit
        """
        try:
            # Factor de ganancia basado en volatilidad
            profit_factor = 2.0 + (1.0 - volatility) * 2.0
            
            # Para posiciones largas
            if position_type == "long":
                # ATR-based take profit
                atr_tp = entry_price + (atr * profit_factor)
                
                # Objetivo porcentual
                percent_tp = entry_price * (1 + (0.03 + (1 - volatility) * 0.07))
                
                return min(atr_tp, percent_tp)
            
            # Para posiciones cortas
            else:
                atr_tp = entry_price - (atr * profit_factor)
                percent_tp = entry_price * (1 - (0.03 + (1 - volatility) * 0.07))
                return max(atr_tp, percent_tp)
        except Exception as e:
            logger.error(f"Error en dynamic_take_profit: {e}")
            # Fallback: take-profit del 5%
            return entry_price * 1.05 if position_type == "long" else entry_price * 0.95

    def hedge_recommendation(self, portfolio, correlation_matrix, market_volatility):
        """
        Generar recomendación de cobertura basada en correlaciones
        
        :param portfolio: Dict de activos y exposiciones
        :param correlation_matrix: Matriz de correlaciones
        :param market_volatility: Volatilidad actual del mercado
        :return: Dict con activos y ratios de cobertura
        """
        recommendations = {}
        if not portfolio or not correlation_matrix:
            return recommendations
        
        try:
            assets = list(portfolio.keys())
            n_assets = len(assets)
            
            # Umbral de volatilidad para activar cobertura
            hedge_threshold = 0.04 + market_volatility * 0.5
            
            for i, asset in enumerate(assets):
                exposure = portfolio[asset]
                
                # Solo considerar activos con exposición significativa
                if abs(exposure) < 0.05 * sum(abs(v) for v in portfolio.values()):
                    continue
                
                # Encontrar mejor activo para cobertura (mayor correlación negativa)
                best_hedge_idx = None
                best_correlation = 1.0
                
                for j in range(n_assets):
                    if i == j:
                        continue
                    
                    corr = correlation_matrix[i][j]
                    if corr < best_correlation:
                        best_correlation = corr
                        best_hedge_idx = j
                
                # Aplicar cobertura si la correlación negativa es fuerte
                if best_hedge_idx is not None and best_correlation < -0.3:
                    hedge_asset = assets[best_hedge_idx]
                    
                    # Calcular ratio de cobertura
                    hedge_ratio = min(
                        0.7, 
                        abs(exposure) * (1 - abs(best_correlation)) * (1 + market_volatility * 2)
                    )
                    
                    # Para posiciones largas, cobertura corta y viceversa
                    position_type = 'short' if exposure > 0 else 'long'
                    
                    recommendations[hedge_asset] = {
                        'ratio': hedge_ratio,
                        'position': position_type,
                        'correlation': best_correlation
                    }
            
            return recommendations
        except Exception as e:
            logger.error(f"Error en hedge_recommendation: {e}")
            return {}

    def update_with_trade_result(self, trade_result, next_state_data):
        """
        Actualizar modelo con resultado de operación
        
        :param trade_result: Dict con resultados de la operación
        :param next_state_data: Dict con los datos del sistema para el siguiente estado
        """
        if self.current_state is None or self.last_action is None:
            return False
        
        try:
            # Calcular recompensa basada en resultado
            reward = self._calculate_reward(trade_result)
            
            # Crear nuevo estado basado en resultado
            next_state = self._create_next_state(next_state_data)
            
            # Actualizar modelo
            done = trade_result.get('closed', False)
            self.update_risk_model(
                self.current_state, 
                self.last_action, 
                reward, 
                next_state, 
                done
            )
            
            # Resetear estado actual
            self.current_state = None
            self.last_action = None
            
            return True
        except Exception as e:
            logger.error(f"Error en update_with_trade_result: {e}")
            return False

    def _calculate_reward(self, trade_result):
        """
        Calcular recompensa basada en resultado de operación
        
        :param trade_result: Dict con resultados de la operación
        """
        pl = trade_result.get('pl', 0)
        duration = trade_result.get('duration', 1)
        drawdown = trade_result.get('max_drawdown', 0)
        
        # Recompensa base basada en profit (escalada para ser menos extrema)
        reward = pl * 1.0 # Multiplicador reducido
        
        # Penalizar por drawdown (escalada para ser menos extrema)
        reward -= drawdown * 10.0 # Multiplicador reducido
        
        # Recompensa/penalización por duración
        if pl > 0:
            reward += 0.1 / duration  # Premiar ganancias rápidas, pero menos agresivo
        else:
            reward -= duration * 0.01  # Penalizar pérdidas prolongadas, pero menos agresivo
            
        # Añadir una pequeña penalización por cada paso para fomentar la eficiencia
        reward -= 0.01
            
        return reward

    def _create_next_state(self, next_state_data):
        """
        Crear próximo estado basado en los datos del sistema después de la operación.
        
        :param next_state_data: Dict con los datos del sistema para el siguiente estado.
        """
        trade_history = next_state_data.get('trade_history', [])
        current_balance = next_state_data.get('current_balance', 0)
        max_balance = next_state_data.get('max_balance', 0)
        atr = next_state_data.get('atr', 0)
        volatility = next_state_data.get('volatility', 0)
        losing_streak = next_state_data.get('losing_streak', 0)
        max_drawdown = next_state_data.get('max_drawdown', 0) # Aunque no se usa en el estado, se pasa para consistencia
        portfolio_correlation = next_state_data.get('portfolio_correlation', None)
        market_trend = next_state_data.get('market_trend', None)

        return self._create_state_vector(
            trade_history, current_balance, max_balance,
            atr, volatility, losing_streak, max_drawdown,
            portfolio_correlation, market_trend
        )

    def save_model(self, filepath):
        """
        Guardar modelo en archivo
        """
        try:
            torch.save({
                'q_network_state': self.q_network.state_dict(),
                'target_network_state': self.target_network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scaler_params': {
                    'mean': self.scaler.mean_,
                    'scale': self.scaler.scale_,
                    'var': self.scaler.var_,
                    'n_samples_seen': self.scaler.n_samples_seen_
                },
                'epsilon': self.epsilon,
                'memory': list(self.memory),
                'config': self.config
            }, filepath)
            logger.info(f"Modelo guardado en {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False

    def load_model(self, filepath):
        """
        Cargar modelo desde archivo
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Cargar estados de redes
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Cargar scaler
            scaler_params = checkpoint.get('scaler_params')
            if scaler_params:
                self.scaler.mean_ = scaler_params['mean']
                self.scaler.scale_ = scaler_params['scale']
                self.scaler.var_ = scaler_params['var']
                self.scaler.n_samples_seen_ = scaler_params['n_samples_seen']
                self.scaler_fitted = True
            
            # Cargar otros parámetros
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.memory = deque(checkpoint.get('memory', []), maxlen=self.config.get("MEMORY_SIZE", 10000))
            self.config = checkpoint.get('config', self.config)
            
            logger.info(f"Modelo cargado desde {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False