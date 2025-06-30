import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TradeManager:
    def __init__(self, trading_system_instance):
        self.system = trading_system_instance

    def manage_open_trades(self, current_data, index):
        current_price = current_data['close']
        
        if self.system.active_trade:
            trade = self.system.active_trade
            direction = trade['direction']
            
            trade['trailing_stop'] = self.system._calculate_trailing_stop(current_price, trade['entry_price'], direction)
            
            close_trade = False
            close_reason = ""
            
            if ((direction > 0 and current_price >= trade['tp']) or (direction < 0 and current_price <= trade['tp'])):
                close_trade = True
                close_reason = "take_profit"
            
            if ((direction > 0 and current_price <= trade['sl']) or (direction < 0 and current_price >= trade['sl'])):
                close_trade = True
                close_reason = "stop_loss"
            
            if trade['trailing_stop'] and ((direction > 0 and current_price <= trade['trailing_stop']) or (direction < 0 and current_price >= trade['trailing_stop'])):
                close_trade = True
                close_reason = "trailing_stop"
            
            if close_trade:
                self._close_trade(current_price, close_reason)
            elif self.system._check_reversal(current_data) and not self.system.hedge_trade:
                self.open_hedge_trade(current_data, -direction, index)
        
        if self.system.hedge_trade:
            hedge = self.system.hedge_trade
            direction = hedge['direction']
            
            close_hedge = False
            if ((direction > 0 and current_price >= hedge['tp']) or (direction < 0 and current_price <= hedge['tp'])):
                close_hedge = True
            
            if ((direction > 0 and current_price <= hedge['sl']) or (direction < 0 and current_price >= hedge['sl'])):
                close_hedge = True
            
            if close_hedge:
                self._close_hedge_trade(current_price)

    def open_trade(self, current_data, direction, index):
        current_price = current_data['close']
        position_size, trade_features = self.system.risk_manager.calculate_position_size(
            self.system.trade_history, self.system.current_balance, self.system.max_balance, 
            self.system.atr, self.system.volatility, self.system.losing_streak, 
            self.system.max_drawdown, self.system.commission_rate
        )
        trade_value = self.system.current_balance * position_size
        
        sl_multiplier = 1.5
        tp_multiplier = 2.5
        sl_distance = sl_multiplier * self.system.atr
        tp_distance = tp_multiplier * self.system.atr
        
        if direction > 0:
            sl = current_price - sl_distance
            tp = current_price + tp_distance
        else:
            sl = current_price + sl_distance
            tp = current_price - tp_distance
        
        commission = trade_value * self.system.commission_rate
        
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
        
        self.system.current_balance -= commission
    
    def open_hedge_trade(self, current_data, direction, index):
        current_price = current_data['close']
        hedge_size = min(self.system.active_trade['size'] * 0.5, self.system.current_balance * 0.1)
        
        if direction > 0:
            sl = current_price - 3 * self.system.atr
            tp = current_price + 1.5 * self.system.atr
        else:
            sl = current_price + 3 * self.system.atr
            tp = current_price - 1.5 * self.system.atr
        
        commission = hedge_size * self.system.commission_rate
        
        self.system.hedge_trade = {
            'entry_index': index,
            'entry_price': current_price,
            'direction': direction,
            'size': hedge_size,
            'sl': sl,
            'tp': tp,
            'commission': commission
        }
        
        self.system.current_balance -= commission
    
    def _close_trade(self, current_price, reason):
        trade = self.system.active_trade
        direction = trade['direction']
        
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        self.system.current_balance += trade['size'] + pl
        
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
        
        if pl > 0:
            self.system.winning_streak += 1
            self.system.losing_streak = 0
        else:
            self.system.losing_streak += 1
            self.system.winning_streak = 0
        
        if self.system.current_balance > self.system.max_balance:
            self.system.max_balance = self.system.current_balance
        
        self.system.active_trade = None
        
        return trade_record
    
    def _close_hedge_trade(self, current_price):
        trade = self.system.hedge_trade
        direction = trade['direction']
        
        if direction > 0:
            pl = (current_price - trade['entry_price']) * (trade['size'] / trade['entry_price'])
        else:
            pl = (trade['entry_price'] - current_price) * (trade['size'] / trade['entry_price'])
        
        self.system.current_balance += trade['size'] + pl
        
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
        
        self.system.hedge_trade = None
        
        return trade_record
