import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self, trading_system_instance):
        self.system = trading_system_instance # Reference to the LiveLearningTradingSystem instance

    def manage_open_trades(self, current_price):
        """Gestiona operaciones abiertas"""
        # Operación principal
        if self.system.active_trade:
            trade = self.system.active_trade
            direction = trade['direction']
            
            # Actualizar trailing stop
            trade['trailing_stop'] = self.system._calculate_trailing_stop(
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
        if self.system.hedge_trade:
            hedge = self.system.hedge_trade
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

    def open_trade(self, current_data, direction, index):
        """Abre una nueva operación principal"""
        current_price = current_data['close']
        position_size, trade_features = self.system.risk_manager.calculate_position_size(
            self.system.trade_history, self.system.current_balance, self.system.max_balance, 
            self.system.atr, self.system.volatility, self.system.losing_streak, 
            self.system.max_drawdown, self.system.commission_rate
        )
        trade_value = self.system.current_balance * position_size
        
        # Calcular SL y TP dinámicos
        sl_multiplier = 1.5
        tp_multiplier = 2.5
        sl_distance = sl_multiplier * self.system.atr
        tp_distance = tp_multiplier * self.system.atr
        
        if direction > 0:  # Operación larga
            sl = current_price - sl_distance
            tp = current_price + tp_distance
        else:  # Operación corta
            sl = current_price + sl_distance
            tp = current_price - tp_distance
        
        # Comisión
        commission = trade_value * self.system.commission_rate
        
        # Crear operación
        self.system.active_trade = {
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
            'model_version': self.system.model_version
        }
        
        # Actualizar balance
        self.system.current_balance -= commission
    
    def open_hedge_trade(self, current_data, direction, index):
        """Abre una operación de cobertura"""
        current_price = current_data['close']
        
        # Tamaño de cobertura (50% de la posición principal)
        hedge_size = min(self.system.active_trade['size'] * 0.5, self.system.current_balance * 0.1)
        
        # Calcular SL y TP para cobertura (más agresivos)
        if direction > 0:
            sl = current_price - 3 * self.system.atr
            tp = current_price + 1.5 * self.system.atr
        else:
            sl = current_price + 3 * self.system.atr
            tp = current_price - 1.5 * self.system.atr
        
        # Comisión
        commission = hedge_size * self.system.commission_rate
        
        # Crear operación de cobertura
        self.system.hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': hedge_size,
            'sl': sl,
            'tp': tp,
            'commission': commission
        }
        
        # Actualizar balance
        self.system.current_balance -= commission
    
    def _close_trade(self, current_price, reason):
        """Cierra la operación principal"""
        trade = self.system.active_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        # Actualizar balance
        self.system.current_balance += trade['size'] + pl
        
        # Registrar trade
        trade_record = {
            'type': 'main',
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'],
            'direction': direction,
            'pl': pl,
            'commission': trade['commission'],
            'duration': len(self.system.equity_curve) - trade['entry_index'],
            'close_reason': reason,
            'model_version': trade['model_version'],
            'features': trade['features'],
            'kelly_size': trade['kelly_size'],
            'actual_position_size': trade['actual_position_size']
        }
        self.system.trade_history.append(trade_record)
        
        # Actualizar rachas
        if pl > 0:
            self.system.winning_streak += 1
            self.system.losing_streak = 0
        else:
            self.system.losing_streak += 1
            self.system.winning_streak = 0
        
        # Actualizar balance máximo
        if self.system.current_balance > self.system.max_balance:
            self.system.max_balance = self.system.current_balance
        
        # Resetear operación
        self.system.active_trade = None
        
        return trade_record
    
    def _close_hedge_trade(self, current_price):
        """Cierra la operación de cobertura"""
        trade = self.system.hedge_trade
        direction = trade['direction']
        
        # Calcular P&L
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        # Actualizar balance
        self.system.current_balance += trade['size'] + pl
        
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
        self.system.trade_history.append(trade_record)
        
        # Resetear operación
        self.system.hedge_trade = None
        
        return trade_record