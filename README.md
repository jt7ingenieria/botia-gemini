# Bot de Trading con IA (Versión 1.2)

Este proyecto implementa un bot de trading avanzado con capacidades de aprendizaje continuo y optimización de hiperparámetros.

## Características Principales

*   **Backtesting con Aprendizaje Continuo:** El bot aprende y adapta su comportamiento a lo largo del tiempo utilizando datos históricos.
*   **Modelos de Pronóstico Ensemble:** Combina múltiples modelos para mejorar la precisión de las predicciones.
*   **Indicadores Técnicos:** Utiliza una variedad de indicadores para analizar el mercado.
*   **Gestión de Riesgo con Red Neuronal:** Un módulo de gestión de riesgo basado en redes neuronales para optimizar la asignación de capital.
*   **Configuración Centralizada:** Todos los parámetros del bot se gestionan a través del archivo `config.py`.
*   **Optimizador Avanzado de Hiperparámetros:** Incorpora un optimizador bayesiano que adapta la búsqueda de parámetros al régimen de mercado actual.

## Estructura del Proyecto

```
bot_ia/
├── src/
│   ├── data_fetcher.py         # Generación y procesamiento de datos de mercado
│   ├── indicators.py           # Cálculo de indicadores técnicos
│   ├── main.py                 # Punto de entrada principal del bot y orquestación
│   ├── risk_management.py      # Módulo de gestión de riesgo
│   ├── strategy.py             # Lógica de la estrategia de trading
│   ├── trading_system.py       # Simulación de backtesting y ejecución de operaciones
│   └── utils.py                # Funciones de utilidad
├── config.py                   # Archivo de configuración centralizada
├── optimize.py                 # Script para la optimización avanzada de hiperparámetros
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Archivos y directorios a ignorar por Git
└── README.md                   # Este archivo
```

## Configuración

Todos los parámetros del bot se encuentran en `config.py`. Puedes ajustar los parámetros de la estrategia, indicadores, gestión de riesgo y el comportamiento general del bot desde este archivo.

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd bot_ia
    ```
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```
3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Ejecutar el Bot (Backtesting)

Para ejecutar el bot en modo de backtesting con los parámetros definidos en `config.py`:

```bash
python -m src.main
```

El bot generará datos de mercado simulados, ejecutará el backtesting con aprendizaje continuo y mostrará los resultados finales, incluyendo métricas de rendimiento y gráficos.

### Optimización de Hiperparámetros

El bot incluye un optimizador avanzado que utiliza optimización bayesiana para encontrar los mejores hiperparámetros. La optimización se adapta al régimen de mercado detectado (tendencia, reversión a la media, alta volatilidad).

Para ejecutar la optimización, usa el argumento `--optimize`:

```bash
python -m src.main --optimize
```

**Optimización por Fases:**

Puedes especificar qué parte del bot quieres optimizar utilizando el argumento `--phase`. Esto es útil para realizar optimizaciones más pequeñas y enfocadas.

*   **Optimizar solo los parámetros del bot (generales):**
    ```bash
    python -m src.main --optimize --phase bot
    ```
*   **Optimizar solo los parámetros de la estrategia de trading:**
    ```bash
    python -m src.main --optimize --phase strategy
    ```
*   **Optimizar solo los parámetros de los indicadores técnicos:**
    ```bash
    python -m src.main --optimize --phase indicators
    ```
*   **Optimizar solo los parámetros de la gestión de riesgo:**
    ```bash
    python -m src.main --optimize --phase risk_manager
    ```
*   **Optimizar todos los parámetros (comportamiento por defecto si solo usas `--optimize`):**
    ```bash
    python -m src.main --optimize --phase all
    ```

Los resultados de la optimización (mejores parámetros, historial de búsqueda, gráficos de convergencia) se guardarán en los directorios `checkpoints/` y `results/`.

## Contribución

Si deseas contribuir a este proyecto, por favor, sigue las siguientes pautas:

1.  Haz un fork del repositorio.
2.  Crea una nueva rama para tu característica (`git checkout -b feature/nueva-caracteristica`).
3.  Realiza tus cambios y asegúrate de que las pruebas pasen.
4.  Haz commit de tus cambios (`git commit -m 'feat: añade nueva característica'`).
5.  Haz push a tu rama (`git push origin feature/nueva-caracteristica`).
6.  Abre un Pull Request.