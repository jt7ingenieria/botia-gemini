# tests/test_data_fetcher.py
# Pruebas unitarias para el módulo data_fetcher.py.
import pytest
import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_fetcher import fetch_historical_data, init_db, HistoricalData, Base, DATABASE_URL, REDIS_HOST, REDIS_PORT, REDIS_DB, redis_client, data_fetch_requests_total, data_fetch_duration_seconds
from unittest.mock import MagicMock, patch
import pickle

# Usar una base de datos en memoria para las pruebas
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="function")
def db_session():
    # Crear una base de datos en memoria para cada prueba
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    
    # Sobrescribir la sesión de la base de datos en el módulo data_fetcher para las pruebas
    # Esto es un hack, en un proyecto real se usaría inyección de dependencias
    from src import data_fetcher
    original_session_local = data_fetcher.SessionLocal
    original_engine = data_fetcher.engine
    original_database_url = data_fetcher.DATABASE_URL

    data_fetcher.SessionLocal = TestingSessionLocal
    data_fetcher.engine = engine
    data_fetcher.DATABASE_URL = TEST_DATABASE_URL

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        # Restaurar la configuración original
        data_fetcher.SessionLocal = original_session_local
        data_fetcher.engine = original_engine
        data_fetcher.DATABASE_URL = original_database_url

@pytest.fixture(scope="function")
def redis_mock():
    # Mockear el cliente de Redis para las pruebas
    from unittest.mock import MagicMock
    from src import data_fetcher
    original_redis_client = data_fetcher.redis_client
    
    mock_redis = MagicMock()
    # Configurar el mock para que el método .get() devuelva None por defecto
    mock_redis.get.return_value = None
    
    data_fetcher.redis_client = mock_redis
    
    try:
        yield mock_redis
    finally:
        data_fetcher.redis_client = original_redis_client

@pytest.fixture(scope="function")
def prometheus_metrics_mock():
    with patch('src.data_fetcher.data_fetch_requests_total') as mock_counter:
        with patch('src.data_fetcher.data_fetch_duration_seconds') as mock_histogram:
            # Configurar los mocks para que sus métodos encadenados funcionen
            mock_counter.labels.return_value.inc = MagicMock()
            mock_histogram.labels.return_value.time.return_value.__enter__ = MagicMock()
            mock_histogram.labels.return_value.time.return_value.__exit__ = MagicMock(return_value=False)
            yield mock_counter, mock_histogram

def test_fetch_historical_data_from_generation(db_session, redis_mock, prometheus_metrics_mock):
    mock_counter, mock_histogram = prometheus_metrics_mock
    symbol = "TESTSYM"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    
    # La primera llamada debería generar datos y guardarlos en la DB y caché
    df = fetch_historical_data(symbol, interval, start_date, end_date)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 5
    
    # Verificar que los datos se guardaron en la DB
    db_data = db_session.query(HistoricalData).filter_by(symbol=symbol, interval=interval).all()
    assert len(db_data) == 5
    
    # Verificar que los datos se guardaron en la caché de Redis
    redis_mock.set.assert_called_once() # Verifica que se llamó a set una vez
    
    # Verificar que las métricas de Prometheus fueron actualizadas
    mock_counter.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_counter.labels.return_value.inc.assert_called_once()
    mock_histogram.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_histogram.labels.return_value.time.assert_called_once()

def test_fetch_historical_data_from_db(db_session, redis_mock, prometheus_metrics_mock):
    mock_counter, mock_histogram = prometheus_metrics_mock
    symbol = "TESTSYM2"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    
    # Generar y guardar datos primero (simulando una primera llamada)
    _ = fetch_historical_data(symbol, interval, start_date, end_date)
    
    # Limpiar los mocks para la siguiente aserción
    redis_mock.reset_mock()
    mock_counter.reset_mock()
    mock_histogram.reset_mock()
    
    # Simular que los datos están en la DB pero no en la caché
    redis_mock.get.return_value = None # Asegurarse de que no se cargue de la caché
    
    # La segunda llamada debería cargar desde la DB y luego guardar en caché
    df_from_db = fetch_historical_data(symbol, interval, start_date, end_date)
    
    assert isinstance(df_from_db, pd.DataFrame)
    assert not df_from_db.empty
    assert len(df_from_db) == 3
    
    # Verificar que los datos cargados son los mismos que los generados
    assert df_from_db.index.equals(pd.to_datetime(pd.date_range(start=start_date, end=end_date)))
    
    # Verificar que se intentó cargar de la caché y luego se guardó en ella
    redis_mock.get.assert_called_once() # Se llamó a get una vez
    redis_mock.set.assert_called_once() # Se llamó a set una vez
    
    # Verificar que las métricas de Prometheus fueron actualizadas
    mock_counter.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_counter.labels.return_value.inc.assert_called_once()
    mock_histogram.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_histogram.labels.return_value.time.assert_called_once()

def test_fetch_historical_data_from_cache(db_session, redis_mock, prometheus_metrics_mock):
    mock_counter, mock_histogram = prometheus_metrics_mock
    symbol = "TESTSYM3"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-02"
    
    # Generar un DataFrame de ejemplo para la caché
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
    data_for_cache = pd.DataFrame({
        'Open': [100, 101],
        'High': [102, 103],
        'Low': [99, 100],
        'Close': [101, 102],
        'Volume': [1000, 1100]
    }, index=dates)
    
    # Configurar el mock de Redis para que devuelva los datos de la caché
    redis_mock.get.return_value = pickle.dumps(data_for_cache)
    
    # La llamada debería cargar directamente de la caché
    df_from_cache = fetch_historical_data(symbol, interval, start_date, end_date)
    
    assert isinstance(df_from_cache, pd.DataFrame)
    assert not df_from_cache.empty
    assert len(df_from_cache) == 2
    
    # Verificar que los datos cargados de la caché son correctos
    pd.testing.assert_frame_equal(df_from_cache, data_for_cache)
    
    # Verificar que se intentó cargar de la caché y no se llamó a la DB ni se guardó en caché
    redis_mock.get.assert_called_once() # Se llamó a get una vez
    redis_mock.set.assert_not_called() # No se llamó a set
    
    # Verificar que las métricas de Prometheus fueron actualizadas
    mock_counter.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_counter.labels.return_value.inc.assert_called_once()
    mock_histogram.labels.assert_called_once_with(symbol=symbol, interval=interval)
    mock_histogram.labels.return_value.time.assert_called_once()

def test_fetch_historical_data_error_notification(db_session, redis_mock, prometheus_metrics_mock):
    mock_counter, mock_histogram = prometheus_metrics_mock
    symbol = "ERROR_SYM"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-05"

    redis_mock.get.return_value = None

    with patch('src.data_fetcher.load_historical_data', side_effect=Exception("Simulated DB error")):
        with patch('src.data_fetcher.send_notification') as mock_send_notification:
            df = fetch_historical_data(symbol, interval, start_date, end_date)
            
            assert df.empty
            
            mock_send_notification.assert_called_once_with(
                "Error en Adquisición de Datos",
                f"Error al obtener datos para {symbol} ({interval}) de {start_date} a {end_date}: Simulated DB error"
            )
            
            mock_counter.labels.assert_called_once_with(symbol=symbol, interval=interval)
            mock_counter.labels.return_value.inc.assert_called_once()
            mock_histogram.labels.assert_called_once_with(symbol=symbol, interval=interval)
            mock_histogram.labels.return_value.time.assert_called_once()

