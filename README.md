# Bot de Trading con IA (Versión 1.0)

Este proyecto implementa un bot de trading basado en inteligencia artificial con un enfoque de aprendizaje continuo y gestión de riesgo. La versión 1.0 incluye un sistema de backtesting robusto y la capacidad de configurar sus parámetros y hiperparámetros para optimizar su rendimiento.

## Características Principales

*   **Backtesting con Aprendizaje Continuo:** El bot simula operaciones de trading sobre datos históricos, reentrenando sus modelos predictivos y de riesgo periódicamente para adaptarse a las condiciones cambiantes del mercado.
*   **Modelos de Pronóstico Ensemble:** Utiliza una combinación de modelos avanzados para predecir movimientos de precios:
    *   **ARIMA:** Para análisis de series temporales.
    *   **Gaussian Process Regressor:** Para modelado no lineal y estimación de incertidumbre.
    *   **Bayesian Ridge Regression:** Un modelo de regresión lineal robusto.
    *   **Monte Carlo Simulator:** Para simular escenarios de precios y capturar "saltos" en el mercado.
    *   **Gradient Boosting Regressor:** Un modelo de ensemble que combina las predicciones de los modelos anteriores para una mayor precisión.
*   **Indicadores Técnicos Integrados:** Incorpora una variedad de indicadores técnicos para enriquecer los datos de entrada de los modelos:
    *   **Velas Heikin-Ashi:** Para suavizar el ruido y facilitar la identificación de tendencias.
    *   **Retornos Logarítmicos:** Para análisis de cambios de precios.
    *   **Volatilidad:** Medida en diferentes ventanas de tiempo.
    *   **RSI (Relative Strength Index):** Para identificar condiciones de sobrecompra/sobreventa.
    *   **MACD (Moving Average Convergence Divergence):** Para analizar la fuerza y dirección de la tendencia.
    *   **ATR (Average True Range):** Para medir la volatilidad del mercado.
*   **Gestión de Riesgo con Red Neuronal:** Implementa un sistema de gestión de riesgo que utiliza una red neuronal (`MLPRegressor`) para ajustar el tamaño de la posición basándose en métricas de rendimiento recientes y la fracción de Kelly.
*   **Configuración Centralizada:** Todos los parámetros y hiperparámetros clave del bot son configurables a través del archivo `config.py`, lo que facilita la experimentación y optimización.

## Estructura del Proyecto

```
.
├── config.py               # Archivo de configuración para todos los parámetros y hiperparámetros.
├── src/
│   ├── data_fetcher.py     # Módulo para la generación de datos de mercado.
│   ├── indicators.py       # Módulo para el cálculo de indicadores técnicos y preprocesamiento de datos.
│   ├── main.py             # Punto de entrada principal del bot y orquestación del backtesting.
│   ├── risk_management.py  # Módulo para la gestión de riesgo y el modelo de red neuronal.
│   ├── strategy.py         # Módulo que contiene los modelos de pronóstico (ARIMA, GP, Monte Carlo, Ensemble).
│   ├── trading_system.py   # Lógica principal del sistema de trading y backtesting.
│   └── utils.py            # Funciones de utilidad (ej. logging).
├── tests/                  # Pruebas unitarias para los módulos.
├── venv/                   # Entorno virtual de Python.
└── .git/                   # Repositorio Git.
```

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd bot_ia
    ```
2.  **Crear y activar el entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Si `requirements.txt` no existe, puedes generarlo con `pip freeze > requirements.txt` después de instalar las librerías necesarias como `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `joblib`.)*

## Uso

1.  **Configurar el Bot:**
    Abre `config.py` y ajusta los parámetros y hiperparámetros según tus necesidades. Puedes modificar:
    *   `BOT_CONFIG`: Configuración general del backtesting.
    *   `STRATEGY_CONFIG`: Hiperparámetros de los modelos de pronóstico (ARIMA, Gaussian Process, Monte Carlo, Gradient Boosting).
    *   `INDICATORS_CONFIG`: Parámetros de los indicadores técnicos (ventanas de RSI, ATR, volatilidad, spans de MACD).
    *   `RISK_MANAGER_CONFIG`: Hiperparámetros de la red neuronal de gestión de riesgo y otros parámetros relacionados.

2.  **Ejecutar el Backtesting:**
    Desde el directorio raíz del proyecto (`bot_ia`), ejecuta el script principal:
    ```bash
    python -m src.main
    ```
    El bot ejecutará el backtesting, imprimirá los resultados en la consola y generará gráficos de rendimiento.

## Optimización y Experimentación

La capacidad de configurar los parámetros a través de `config.py` es crucial para la optimización. Se recomienda:

*   **Experimentar con diferentes valores:** Modifica los hiperparámetros de los modelos y los parámetros de los indicadores para ver cómo afectan la rentabilidad y la estabilidad del bot.
*   **Realizar un Grid Search o Random Search:** Para encontrar combinaciones óptimas de hiperparámetros, aunque esto requeriría implementar una lógica de optimización externa o usar librerías como `scikit-learn` para ello.
*   **Analizar los resultados:** Presta atención al balance final, retorno total, `max_drawdown`, `win_rate` y `profit_factor` para evaluar el rendimiento.

## Mejoras y Funciones Futuras (Roadmap)

### Versión 1.0 (Actual)
*   Backtesting con aprendizaje continuo.
*   Modelos de pronóstico ensemble (ARIMA, GP, Bayesian Ridge, Monte Carlo, Gradient Boosting).
*   Indicadores técnicos (Heikin-Ashi, Retornos, Volatilidad, RSI, MACD, ATR).
*   Gestión de riesgo con red neuronal y Kelly fraccionado.
*   Configuración centralizada de parámetros y hiperparámetros.

### Próximas Mejoras y Funciones (Roadmap)

*   **Integración con Datos Reales:** Conectar el bot a APIs de exchanges (ej. Binance, Kraken) para obtener datos de mercado en tiempo real y ejecutar operaciones en vivo (con precaución y solo después de pruebas exhaustivas).
*   **Optimización de Hiperparámetros Automatizada:** Implementar algoritmos de optimización (ej. Grid Search, Random Search, Bayesian Optimization) para encontrar automáticamente los mejores conjuntos de hiperparámetros.
*   **Estrategias de Trading Adicionales:** Permitir la definición y selección de múltiples estrategias de trading, no solo la actual basada en predicción de precios.
*   **Visualización Interactiva:** Mejorar los gráficos de rendimiento y añadir dashboards interactivos para un análisis más profundo.
*   **Alertas y Notificaciones:** Implementar un sistema de alertas para notificar sobre eventos importantes (ej. operaciones ejecutadas, cambios de tendencia, drawdown significativo).
*   **Persistencia del Modelo:** Mejorar la forma en que los modelos entrenados se guardan y cargan para evitar reentrenamientos innecesarios.
*   **Manejo de Errores y Robustez:** Fortalecer el manejo de errores y la robustez del sistema ante fallos de conexión o datos.
*   **Logging Avanzado:** Implementar un sistema de logging más granular y configurable para depuración y análisis.
*   **Soporte para Múltiples Activos:** Extender la capacidad del bot para operar con múltiples pares de trading simultáneamente.
*   **Integración con Bases de Datos:** Almacenar datos históricos y de operaciones en una base de datos para un acceso más eficiente y escalable.
*   **Interfaz de Usuario (GUI/Web):** Desarrollar una interfaz gráfica de usuario o una interfaz web para controlar y monitorear el bot de manera más intuitiva.
