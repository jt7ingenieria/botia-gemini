import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Módulo 4: Evaluación
# =============================================================================
class ModelEvaluator:
    @staticmethod
    def evaluate(predictor, test_df, method='ensemble'):
        """Evalúa el modelo sin alterar datos originales."""
        test_data = test_df.copy()
        true_values = []
        predictions = []
        
        for i in range(len(test_data)):
            if i < 10:  # Mínimo histórico para predicción
                continue
                
            historical = test_data.iloc[:i]
            true = test_data.iloc[i]['close']
            
            try:
                pred = predictor.predict(historical, steps=1, method=method)[0]
                true_values.append(true)
                predictions.append(pred)
                
                # Actualizar con valor real (no predicho) para siguiente iteración
                test_data.loc[test_data.index[i], 'close'] = true
            except Exception as e:
                logger.error(f"Evaluation error at step {i}: {str(e)}")
        
        return mean_squared_error(true_values, predictions)