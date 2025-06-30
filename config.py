# config.py

# General Bot Configuration
BOT_CONFIG = {
    "num_market_data_points": 1000,
    "commission_rate": 0.001,
    "initial_balance": 10000,
    "trading_system_state_file": "trading_system_state.joblib",
}

# Strategy Models Configuration (from src/strategy.py)
STRATEGY_CONFIG = {
    "ARIMA_ORDER": (1, 1, 1),
    "ARIMA_N_SPLITS": 5,
    "GP_KERNEL_CONSTANT": 1.0,
    "GP_KERNEL_RBF_LENGTH_SCALE": 1.0,
    "GP_N_RESTARTS_OPTIMIZER": 10,
    "MONTE_CARLO_STEPS": 1,
    "MONTE_CARLO_N_SIMULATIONS": 1000,
    "GRADIENT_BOOSTING_N_ESTIMATORS": 100,
    "GRADIENT_BOOSTING_LEARNING_RATE": 0.1,
    "GRADIENT_BOOSTING_MAX_DEPTH": 3,
    "GRADIENT_BOOSTING_RANDOM_STATE": 42,
}

# Indicators Configuration (from src/indicators.py)
INDICATORS_CONFIG = {
    "RSI_WINDOW": 14,
    "ATR_WINDOW": 14,
    "VOLATILITY_WINDOWS": [5, 10, 20],
    "MACD_EMA_FAST_SPAN": 12,
    "MACD_EMA_SLOW_SPAN": 26,
    "MACD_SIGNAL_SPAN": 9,
}

# Risk Management Model Configuration (from src/risk_management.py)
RISK_MANAGER_CONFIG = {
    "MLP_HIDDEN_LAYER_SIZES": (32, 16),
    "MLP_ACTIVATION": 'relu',
    "MLP_SOLVER": 'adam',
    "MLP_MAX_ITER": 1000,
    "MLP_RANDOM_STATE": 42,
    "MLP_WARM_START": True,
    "KELLY_FRACTION_MULTIPLIER": 0.5,
    "KELLY_FRACTION_MAX": 0.2,
    "RECENT_TRADES_WINDOW": 20,
    "MIN_TRAINING_SAMPLES": 10,
}
