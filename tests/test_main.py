import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from src.main import main

@pytest.fixture
def mock_dependencies(request):
    mock_generate_market_data = patch('src.main.generate_market_data').start()
    mock_trading_system = patch('src.main.LiveLearningTradingSystem').start()
    mock_log_message = patch('src.main.logger.info').start()
    mock_bot_config = patch('src.main.BOT_CONFIG', {'commission_rate': 0.001, 'initial_balance': 10000, 'trading_system_state_file': 'test_state.joblib'}).start()
    request.addfinalizer(patch.stopall)

    # Configure mock_generate_market_data to return a dummy DataFrame
    mock_generate_market_data.return_value = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200]
    })

    # Configure mock_trading_system methods
    mock_trading_system.return_value.run_backtest.return_value = {
        'final_balance': 10500.0,
        'initial_balance': 10000.0,
        'total_return': 0.05,
        'max_drawdown': 0.01,
        'total_trades': 10,
        'win_rate': 0.6,
        'avg_win': -50.0,
        'avg_loss': -20.0,
        'profit_factor': 2.5,
        'model_performance': {},
        'learning_log': []
    }
    mock_trading_system.return_value.plot_results = MagicMock()

    return {
        'generate_market_data': mock_generate_market_data,
        'trading_system': mock_trading_system,
        'log_message': mock_log_message
    }

def test_main_runs_without_error(mock_dependencies):
    # This test now relies on the mocked dependencies
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an unexpected exception: {e}")

    mock_dependencies['generate_market_data'].assert_called_once() # Check generate_market_data was called
    mock_dependencies['trading_system'].assert_called_once() # Check LiveLearningTradingSystem was instantiated
    mock_dependencies['trading_system'].return_value.run_backtest.assert_called_once() # Check run_backtest was called
    mock_dependencies['trading_system'].return_value.plot_results.assert_called_once() # Check plot_results was called
    mock_dependencies['log_message'].assert_has_calls([
        call("Iniciando el bot de trading..."),
        call("Bot de trading finalizado.")
    ])