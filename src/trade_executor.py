import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, initial_balance, commission_rate, max_drawdown, risk_model):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_drawdown = max_drawdown
        self.max_balance = initial_balance
        self.trade_history = []
        self.active_trade = None
        self.hedge_trade = None
        self.losing_streak = 0
        self.winning_streak = 0
        self.risk_model = risk_model # AdvancedRiskManager instance

    def _open_trade(self, current_data, direction, index, position_size, metadata, atr, volatility):
        current_price = current_data['close']
        
        # amount_to_invest es el valor monetario a invertir
        amount_to_invest = max(self.current_balance * position_size, self.current_balance * 0.005) # Asegurar un tamaño mínimo de operación (0.5% del balance)
        
        # Calcular el número de unidades a operar
        num_units = amount_to_invest / current_price if current_price > 0 else 0

        # Calcular SL y TP dinámicos usando el risk_model
        sl = self.risk_model.dynamic_stop_loss(current_price, current_price, atr, volatility, "long" if direction > 0 else "short")
        tp = self.risk_model.dynamic_take_profit(current_price, current_price, atr, volatility, "long" if direction > 0 else "short")
        
        # Comisión
        commission = amount_to_invest * self.commission_rate
        
        # Crear operación
        self.active_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': num_units, # 'size' ahora representa el número de unidades
            'position_size': position_size, # Fracción del balance
            'sl': sl,
            'tp': tp,
            'trailing_stop': None,
            'commission': commission,
            'metadata': metadata, # Guardar el metadata del RiskManager
            'model_version': None # Se actualizará desde LiveLearningTradingSystem
        }

        # Actualizar balance
        self.current_balance -= commission
        logger.debug(f"OPEN TRADE: Balance actual: {self.current_balance:.2f}, Tamaño de posición (fracción): {position_size:.4f}, Unidades: {num_units:.4f}, Valor monetario: {amount_to_invest:.2f}, Comisión: {commission:.2f}")
        
    def _open_hedge_trade(self, current_data, direction, index, atr):
        current_price = current_data['close']

        # Calcular pérdida potencial de la operación principal si alcanza su SL
        main_trade = self.active_trade
        if main_trade['direction'] > 0:  # Operación principal es larga
            potential_main_loss = (main_trade['entry_price'] - main_trade['sl']) * main_trade['size'] # size es num_units
        else:  # Operación principal es corta
            potential_main_loss = (main_trade['sl'] - main_trade['entry_price']) * main_trade['size'] # size es num_units

        # Asegurar que la pérdida potencial sea un valor positivo
        potential_main_loss = abs(potential_main_loss)

        # Beneficio objetivo de la cobertura: cubrir pérdida principal + 1% del balance inicial
        target_profit_from_hedge = potential_main_loss + (self.initial_balance * 0.01)

        # Tamaño de la operación de cobertura: número de unidades igual al de la operación principal
        hedge_units = main_trade['size']

        # Calcular el cambio de precio necesario para que la cobertura alcance el beneficio objetivo
        # P&L = (exit_price - entry_price) * num_units para largo
        # P&L = (entry_price - exit_price) * num_units para corto
        # Despejando (exit_price - entry_price) = P&L / num_units
        # Para la cobertura, el 'entry_price' es current_price y 'num_units' es hedge_units
        
        # Asegurarse de que hedge_units no sea cero para evitar división por cero
        if hedge_units == 0:
            logger.warning("Hedge units is zero, cannot open hedge trade.")
            return

        required_price_change = target_profit_from_hedge / hedge_units

        # Calcular TP para la cobertura
        if direction > 0:  # Cobertura es larga
            tp = current_price + required_price_change
        else:  # Cobertura es corta
            tp = current_price - required_price_change

        # SL para la cobertura (más agresivo, por ejemplo, 1 ATR)
        if direction > 0:
            sl = current_price - 1 * atr
        else:
            sl = current_price + 1 * atr

        # Comisión
        commission = hedge_units * current_price * self.commission_rate # Comisión basada en valor monetario de las unidades

        # Crear operación de cobertura
        self.hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': hedge_units, # 'size' ahora representa el número de unidades
            'sl': sl,
            'tp': tp,
            'commission': commission,
            'main_trade_potential_loss': potential_main_loss, # Para depuración/análisis
            'target_profit_from_hedge': target_profit_from_hedge # Para depuración/análisis
        }

        # Actualizar balance
        self.current_balance -= commission
        logger.debug(f"OPEN HEDGE TRADE: Balance actual: {self.current_balance:.2f}, Unidades: {hedge_units:.4f}, Comisión: {commission:.2f}")
    
    def _close_trade(self, current_price, reason, equity_curve_len):
        trade = self.active_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * trade['size'] # size es num_units
        else:
            pl = (trade['entry_price'] - current_price) * trade['size'] # size es num_units
        
        # Actualizar balance
        self.current_balance += pl
        logger.debug(f"CLOSE TRADE (Main): Balance actual: {self.current_balance:.2f}, P&L: {pl:.2f}, Razón: {reason}")
        
        # Registrar trade
        trade_record = {
            'type': 'main',
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'], # 'size' es num_units
            'direction': direction,
            'pl': pl,
            'commission': trade['commission'],
            'duration': equity_curve_len - trade['entry_index'],
            'close_reason': reason,
            'model_version': trade['model_version'],
            'metadata': trade['metadata'] # Guardar el metadata del RiskManager
        }
        self.trade_history.append(trade_record)
        
        # Actualizar el modelo de riesgo con el resultado de la operación
        next_state_data = {
            'trade_history': self.trade_history,
            'current_balance': self.current_balance,
            'max_balance': self.max_balance,
            'atr': self.risk_model.atr, # Usar el atr del risk_model
            'volatility': self.risk_model.volatility, # Usar la volatilidad del risk_model
            'losing_streak': self.losing_streak,
            'max_drawdown': self.max_drawdown,
            'portfolio_correlation': None, # No disponible en este contexto
            'market_trend': None # No disponible en este contexto
        }
        self.risk_model.update_with_trade_result(trade_record, next_state_data)
        
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
        trade = self.hedge_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * trade['size'] # size es num_units
        else:
            pl = (trade['entry_price'] - current_price) * trade['size'] # size es num_units
        
        # Actualizar balance
        self.current_balance += pl
        logger.debug(f"CLOSE TRADE (Hedge): Balance actual: {self.current_balance:.2f}, P&L: {pl:.2f}")
        
        # Registrar trade
        trade_record = {
            'type': 'hedge',
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'], # 'size' es num_units
            'direction': direction,
            'pl': pl,
            'commission': trade['commission']
        }
        self.trade_history.append(trade_record)
        
        # Resetear operación
        self.hedge_trade = None
        
        return trade_record
    
    def _check_drawdown(self):
        drawdown = (self.max_balance - self.current_balance) / self.max_balance
        return drawdown >= self.max_drawdown
    
    def _check_reversal(self, current_price, predicted_direction, atr):
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
        
        return current_loss > 1.2 * atr and predicted_direction != direction

    def _calculate_trailing_stop(self, current_price, entry_price, direction, atr):
        # Precio de activación (cuando la ganancia supera 1.2 ATR)
        activation_level = 1.2 * atr
        new_trailing_stop = self.active_trade.get('trailing_stop', None)
        
        if direction > 0:  # Largo
            unrealized_profit = current_price - entry_price
            if unrealized_profit > activation_level:
                new_trailing_stop = max(
                    new_trailing_stop or 0, 
                    current_price - 0.8 * atr
                )
        else:  # Corto
            unrealized_profit = entry_price - current_price
            if unrealized_profit > activation_level:
                new_trailing_stop = min(
                    new_trailing_stop or float('inf'), 
                    current_price + 0.8 * atr
                )
        
        return new_trailing_stop
