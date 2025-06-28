# tests/test_data_fetcher.py
# Pruebas unitarias para el módulo data_fetcher.py.
import pytest
import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_fetcher import fetch_historical_data, init_db, HistoricalData, Base, DATABASE_URL

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
        data_fetcher.DATABASE_url = original_database_url

def test_fetch_historical_data_from_generation(db_session):
    symbol = "TESTSYM"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    
    # La primera llamada debería generar datos y guardarlos en la DB
    df = fetch_historical_data(symbol, interval, start_date, end_date)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 5
    
    # Verificar que los datos se guardaron en la DB
    db_data = db_session.query(HistoricalData).filter_by(symbol=symbol, interval=interval).all()
    assert len(db_data) == 5

def test_fetch_historical_data_from_db(db_session):
    symbol = "TESTSYM2"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    
    # Generar y guardar datos primero
    _ = fetch_historical_data(symbol, interval, start_date, end_date)
    
    # La segunda llamada debería cargar desde la DB
    df_from_db = fetch_historical_data(symbol, interval, start_date, end_date)
    
    assert isinstance(df_from_db, pd.DataFrame)
    assert not df_from_db.empty
    assert len(df_from_db) == 3
    
    # Verificar que los datos cargados son los mismos que los generados
    # (comparación simplificada, en un caso real se compararían los valores)
    assert df_from_db.index.equals(pd.to_datetime(pd.date_range(start=start_date, end=end_date)))
