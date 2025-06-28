import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, poisson
from joblib import dump, load
import os

from src.indicators import DataProcessor # Importar DataProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Módulo 2: Modelos de Pronóstico
# =============================================================================
class ARIMAModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
    
    def cross_validate(self, y, n_splits=5):
        """Validación cruzada temporal correcta."""
        best_model = None
        best_mse = float('inf')
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for train_index, test_index in tscv.split(y):
            y_train, y_test = y[train_index], y[test_index]
            try:
                model = ARIMA(y_train, order=self.order).fit()
                preds = model.forecast(steps=len(y_test))
                mse = mean_squared_error(y_test, preds)
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
            except Exception as e:
                logger.warning(f"ARIMA CV error: {str(e)}")
        
        # Reentrenar con todos los datos
        if best_model:
            self.model = ARIMA(y, order=self.order).fit()
        return self.model

class GaussianProcessModel:
    def __init__(self, kernel=None):
        self.kernel = kernel or ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

class MonteCarloSimulator:
    def __init__(self):
        self.params = {}
    
    def calibrate(self, returns):
        """Calcula parámetros de saltos usando momentos."""
        n = len(returns)
        lambda_ = np.sum(np.abs(returns) > 3 * returns.std()) / n
        jump_idx = np.where(np.abs(returns) > 3 * returns.std())[0]
        jump_returns = returns.iloc[jump_idx]
        
        self.params = {
            'mu': returns.mean(),
            'sigma': returns.std(),
            'lambda_jump': lambda_,
            'mu_jump': jump_returns.mean(),
            'sigma_jump': jump_returns.std()
        }
        return self.params

    def simulate(self, S0, steps=1, n_simulations=1000):
        """Simulación de saltos con parámetros calibrados."""
        results = []
        for _ in range(n_simulations):
            St = S0
            for _ in range(steps):
                # Componente de difusión
                Z = norm.rvs()
                diffusion = (self.params['mu'] - 0.5*self.params['sigma']**2) + self.params['sigma']*Z
                
                # Componente de salto
                if np.random.rand() < self.params['lambda_jump']:
                    J = norm.rvs(loc=self.params['mu_jump'], scale=self.params['sigma_jump'])
                    diffusion += J
                
                St *= np.exp(diffusion)
            results.append(St)
        return np.mean(results)

# =============================================================================
# Módulo 3: Predictor Principal
# =============================================================================
class FinancialPredictor:
    def __init__(self, model_path='financial_predictor.joblib'):
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.trained = False
        self.models = {
            'arima': ARIMAModel(order=(1,1,1)),
            'gp': GaussianProcessModel(),
            'bayesian': BayesianRidge(),
            'montecarlo': MonteCarloSimulator(),
            'ensemble': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
        }

    def train(self, df, save_model=True):
        """Entrenamiento con pipeline modularizado."""
        # Preprocesamiento
        processed_df = self.data_processor.preprocess(df)
        X = processed_df.drop(columns=['close'])
        y = processed_df['close'].values
        
        # Escalado
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.feature_names = X.columns.tolist()
        
        # Entrenar modelos base
        self.models['arima'].cross_validate(y)
        self.models['gp'].fit(X_scaled, y)
        self.models['bayesian'].fit(X_scaled, y)
        
        # Calibrar Monte Carlo
        self.models['montecarlo'].calibrate(processed_df['returns'])
        
        # Dataset aumentado para ensemble
        X_aug = self._create_augmented_dataset(processed_df, X_scaled)
        self.models['ensemble'].fit(X_aug, y)
        
        self.trained = True
        if save_model:
            self.save_model()

    def _create_augmented_dataset(self, df, X_scaled):
        """Crea dataset con predicciones de modelos base."""
        base_preds = pd.DataFrame({
            'arima': [self.models['arima'].model.forecast(steps=1)[0] for _ in range(len(df))]
        })
        base_preds['gp'] = self.models['gp'].model.predict(X_scaled)
        base_preds['bayesian'] = self.models['bayesian'].predict(X_scaled)
        
        # Features adicionales
        ha_features = df[['ha_open', 'ha_high', 'ha_low', 'ha_close']]
        
        # Asegurarse de que X_scaled, base_preds, ha_features y df['volume'] tengan el mismo número de filas
        # Esto es crucial para np.hstack
        min_len = min(len(X_scaled), len(base_preds), len(ha_features), len(df['volume']))
        
        return np.hstack([
            X_scaled[:min_len],
            base_preds.iloc[:min_len],
            ha_features.iloc[:min_len],
            df[['volume']].iloc[:min_len]
        ])

    def predict(self, historical_data, steps=5, method='ensemble'):
        """Predicción eficiente con actualización incremental."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
            
        current_data = historical_data.copy()
        predictions = []
        
        for _ in range(steps):
            # Preprocesamiento incremental
            processed = self.data_processor.preprocess(current_data)
            latest = processed.iloc[[-1]].drop(columns=['close'])
            X_scaled = self.scaler.transform(latest)
            
            # Selección de método
            if method == 'arima':
                pred = self.models['arima'].model.forecast(steps=1)[0]
            elif method == 'gp':
                pred = self.models['gp'].model.predict(X_scaled)[0]
            elif method == 'bayesian':
                pred = self.models['bayesian'].predict(X_scaled)[0]
            elif method == 'montecarlo':
                last_price = current_data['close'].iloc[-1]
                pred = self.models['montecarlo'].simulate(last_price)
            elif method == 'ensemble':
                X_aug = self._create_augmented_dataset(latest, X_scaled)
                pred = self.models['ensemble'].predict(X_aug)[0]
            else:
                raise ValueError(f"Invalid method: {method}")
                
            predictions.append(pred)
            
            # Actualizar datos históricos con nueva observación
            new_row = current_data.iloc[-1].copy()
            # Asegurarse de que 'open', 'high', 'low', 'close' existan en new_row
            if 'open' not in new_row.index:
                new_row['open'] = pred # O un valor por defecto razonable
            if 'high' not in new_row.index:
                new_row['high'] = pred
            if 'low' not in new_row.index:
                new_row['low'] = pred
            new_row['close'] = pred
            
            # Asegurarse de que el índice sea un DatetimeIndex si historical_data lo era
            if isinstance(current_data.index, pd.DatetimeIndex):
                # Generar una nueva fecha para la predicción
                last_date = current_data.index[-1]
                # Asumiendo que el intervalo es diario para este ejemplo
                new_date = last_date + pd.Timedelta(days=1)
                new_row.name = new_date # Asignar la nueva fecha como nombre de la fila
                current_data = pd.concat([current_data, pd.DataFrame([new_row])])
            else:
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return np.array(predictions)

    def save_model(self):
        """Guarda el estado completo del predictor."""
        dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'processor': self.data_processor
        }, self.model_path)

    def load_model(self):
        """Carga un modelo previamente guardado."""
        data = load(self.model_path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.data_processor = data['processor']
        self.trained = True