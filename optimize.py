# optimize.py
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import TimeSeriesSplit
from skopt.callbacks import CheckpointSaver
import logging
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

# Project-specific imports
from src.data_fetcher import generate_market_data
from src.trading_system import LiveLearningTradingSystem
from src.indicators import DataProcessor
from config import BOT_CONFIG, STRATEGY_CONFIG, INDICATORS_CONFIG, RISK_MANAGER_CONFIG

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Scikit-learn Estimator Wrapper for the Trading System ---
class TradingSystemEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        all_configs = {}
        all_configs.update(BOT_CONFIG)
        all_configs.update(STRATEGY_CONFIG)
        all_configs.update(INDICATORS_CONFIG)
        all_configs.update(RISK_MANAGER_CONFIG)

        for key, value in all_configs.items():
            setattr(self, key, value)

        self.set_params(**kwargs)

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'MLP_HIDDEN_LAYER_SIZES' and isinstance(value, str):
                # Convert string back to tuple for internal use
                setattr(self, key, tuple(map(int, value.split('_'))))
            elif key == 'VOLATILITY_WINDOWS' and isinstance(value, str):
                # Convert string back to tuple for internal use
                setattr(self, key, tuple(map(int, value.split('_'))))
            elif key == 'ARIMA_ORDER' and isinstance(value, str):
                # ARIMA_ORDER is handled by individual p, d, q params in search space
                # Convert string back to tuple for internal use
                setattr(self, key, tuple(map(int, value.split('_'))))
            else:
                # For other parameters, or if already a tuple, set directly
                setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        # Get all attributes that are not private and not methods
        params = {key: getattr(self, key) for key in self.__dict__ if not key.startswith('_') and not callable(getattr(self, key))}

        # Convert tuples to string representations for skopt Categorical space
        if 'MLP_HIDDEN_LAYER_SIZES' in params and isinstance(params['MLP_HIDDEN_LAYER_SIZES'], tuple):
            params['MLP_HIDDEN_LAYER_SIZES'] = '_'.join(map(str, params['MLP_HIDDEN_LAYER_SIZES']))
        if 'VOLATILITY_WINDOWS' in params and isinstance(params['VOLATILITY_WINDOWS'], tuple):
            params['VOLATILITY_WINDOWS'] = '_'.join(map(str, params['VOLATILITY_WINDOWS']))
        if 'ARIMA_ORDER' in params and isinstance(params['ARIMA_ORDER'], tuple):
            params['ARIMA_ORDER'] = '_'.join(map(str, params['ARIMA_ORDER']))

        return params

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def run_backtest(self, market_data):
        bot_conf = {k: getattr(self, k) for k in BOT_CONFIG}
        strat_conf = {k: getattr(self, k) for k in STRATEGY_CONFIG}
        ind_conf = {k: getattr(self, k) for k in INDICATORS_CONFIG}
        risk_conf = {k: getattr(self, k) for k in RISK_MANAGER_CONFIG}

        if isinstance(strat_conf.get('ARIMA_ORDER'), str):
            p, d, q = map(int, strat_conf['ARIMA_ORDER'].split('_'))
            strat_conf['ARIMA_ORDER'] = (p, d, q)
        
        if isinstance(ind_conf.get('VOLATILITY_WINDOWS'), str):
            ind_conf['VOLATILITY_WINDOWS'] = tuple(map(int, ind_conf['VOLATILITY_WINDOWS'].split('_')))
        
        if isinstance(risk_conf.get('MLP_HIDDEN_LAYER_SIZES'), str):
            risk_conf['MLP_HIDDEN_LAYER_SIZES'] = tuple(map(int, risk_conf['MLP_HIDDEN_LAYER_SIZES'].split('_')))

        system = LiveLearningTradingSystem(
            initial_balance=bot_conf['initial_balance'],
            commission_rate=bot_conf['commission_rate'],
            strategy_config=strat_conf,
            indicators_config=ind_conf,
            risk_manager_config=risk_conf
        )
        results = system.run_backtest(market_data)
        return results


# --- Regime Detector ---
class MarketRegimeDetector:
    def __init__(self):
        self.data_processor = DataProcessor(INDICATORS_CONFIG)

    def detect(self, data):
        processed_data = self.data_processor.preprocess(data.copy()).dropna()
        if processed_data.empty:
            return 'mean_reverting'
        latest_data = processed_data.iloc[-1]
        volatility = latest_data.get('volatility_20', 0.02)
        rsi = latest_data.get('rsi', 50)
        if volatility > 0.04:
            return 'high_volatility'
        elif rsi > 60 or rsi < 40:
            return 'trending'
        else:
            return 'mean_reverting'


# --- Custom Scorer ---
class RiskAdjustedScorer:
    def __call__(self, estimator, X, y=None):
        results = estimator.run_backtest(X)
        total_return = results.get('total_return', -1.0)
        max_drawdown = results.get('max_drawdown', 1.0)
        win_rate = results.get('win_rate', 0.0)
        profit_factor = results.get('profit_factor', 0.0)
        total_trades = results.get('total_trades', 0)
        if np.isinf(profit_factor): profit_factor = 10.0
        if total_trades < 10: return -1000
        score = (1 + total_return) * (1 - max_drawdown) * (1 + win_rate) * (1 + profit_factor)
        return -score

# --- Callback for tqdm progress bar ---
def pbar_update_callback(res, pbar):
    pbar.update(1)
    pbar.set_description(f"Best score: {-res.fun:.4f}")

# --- Main Optimizer Class ---
class AdvancedOptimizer:
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.scorer = RiskAdjustedScorer()
        self.param_spaces = self._define_adaptive_spaces()

    def _define_adaptive_spaces(self):
        return {
            'trending': {
                'RSI_WINDOW': Integer(14, 40, name='RSI_WINDOW'),
                'MACD_EMA_FAST_SPAN': Integer(12, 20, name='MACD_EMA_FAST_SPAN'),
                'MACD_EMA_SLOW_SPAN': Integer(26, 50, name='MACD_EMA_SLOW_SPAN'),
                'KELLY_FRACTION_MULTIPLIER': Real(0.5, 1.0, name='KELLY_FRACTION_MULTIPLIER'),
                'KELLY_FRACTION_MAX': Real(0.15, 0.3, name='KELLY_FRACTION_MAX'),
            },
            'mean_reverting': {
                'RSI_WINDOW': Integer(7, 14, name='RSI_WINDOW'),
                'VOLATILITY_WINDOWS': Categorical(['5_10_20', '7_14_28'], name='VOLATILITY_WINDOWS'),
                'KELLY_FRACTION_MULTIPLIER': Real(0.2, 0.6, name='KELLY_FRACTION_MULTIPLIER'),
                'KELLY_FRACTION_MAX': Real(0.05, 0.15, name='KELLY_FRACTION_MAX'),
            },
            'high_volatility': {
                'ATR_WINDOW': Integer(14, 30, name='ATR_WINDOW'),
                'KELLY_FRACTION_MULTIPLIER': Real(0.1, 0.4, name='KELLY_FRACTION_MULTIPLIER'),
                'KELLY_FRACTION_MAX': Real(0.01, 0.1, name='KELLY_FRACTION_MAX'),
                'MLP_HIDDEN_LAYER_SIZES': Categorical(['16_8', '32_16'], name='MLP_HIDDEN_LAYER_SIZES'),
            }
        }

    def optimize(self, data, n_iter=50):
        regime = self.regime_detector.detect(data)
        logger.info(f"Market regime detected: {regime}")
        search_space = self.param_spaces[regime]
        logger.info(f"Using search space for '{regime}' regime.")
        
        tscv = TimeSeriesSplit(n_splits=3)
        opt = BayesSearchCV(
            estimator=TradingSystemEstimator(),
            search_spaces=search_space, cv=tscv, n_iter=n_iter, scoring=self.scorer,
            optimizer_kwargs={'base_estimator': 'GP'}, random_state=42, n_jobs=-1, verbose=0
        )
        
        checkpoint_path = "./checkpoints/optimization_checkpoint.pkl"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_saver = CheckpointSaver(checkpoint_path, store_objective=False)
        
        logger.info(f"Starting Bayesian optimization for {n_iter} iterations...")
        with tqdm(total=n_iter) as pbar:
            # Use functools.partial to pass pbar to the callback
            from functools import partial
            callback_with_pbar = partial(pbar_update_callback, pbar=pbar)
            opt.fit(data, callback=[checkpoint_saver, callback_with_pbar])
        
        logger.info("Optimización completada!")
        self._save_optimization_results(opt, regime)
        return opt.best_params_

    def _save_optimization_results(self, optimizer, regime):
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        joblib.dump(optimizer, os.path.join(results_dir, f"optimization_results_{regime}.pkl"))
        pd.DataFrame(optimizer.cv_results_).to_csv(os.path.join(results_dir, f"optimization_history_{regime}.csv"))
        logger.info(f"Best score found: {-optimizer.best_score_:.4f}")
        logger.info(f"Best parameters: {optimizer.best_params_}")
        self._plot_convergence(optimizer, regime)

    def _plot_convergence(self, optimizer, regime):
        plt.figure(figsize=(10, 6))
        scores = [-s for s in optimizer.cv_results_['mean_test_score']]
        iterations = range(1, len(scores) + 1)
        plt.plot(iterations, scores, 'o-', label='Score per iteration')
        best_scores = np.maximum.accumulate(scores)
        plt.plot(iterations, best_scores, 'r--', label='Best score so far')
        plt.title(f'Optimization Convergence ({regime.capitalize()} Regime)')
        plt.xlabel('Iteración'); plt.ylabel('Puntuación Media')
        plt.legend(); plt.grid(True)
        plot_path = os.path.join("./results", f"optimization_convergence_{regime}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Convergence plot saved to {plot_path}")

# --- Main execution block ---
if __name__ == "__main__":
    logger.info("Generating market data for optimization...")
    market_data = generate_market_data(num_points=1500)
    optimizer = AdvancedOptimizer()
    best_params = optimizer.optimize(market_data, n_iter=30)
    
    print("\n" + "="*50)
    print("OPTIMIZACIÓN COMPLETADA - PARÁMETROS FINALES")
    print("="*50)
    for param, value in best_params.items():
        if param in ['MLP_HIDDEN_LAYER_SIZES', 'VOLATILITY_WINDOWS'] and isinstance(value, str):
            final_value = tuple(map(int, value.split('_')))
        else:
            final_value = value
        print(f"- {param}: {final_value}")
    print("="*50)
    print("\nTo run a backtest with these parameters, update your config.py file.")
