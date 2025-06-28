# src/data_fetcher.py
# Módulo para la obtención de datos de mercado.
# Entradas: Símbolo del activo, intervalo de tiempo, rango de fechas.
# Salidas: DataFrame de pandas con datos históricos (OHLCV).

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./trading_data.db"
Base = declarative_base()

class HistoricalData(Base):
    __tablename__ = "historical_data"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    interval = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_historical_data(df: pd.DataFrame, symbol: str, interval: str):
    db = SessionLocal()
    try:
        for index, row in df.iterrows():
            data_entry = HistoricalData(
                symbol=symbol,
                interval=interval,
                timestamp=index.to_pydatetime(),
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume']
            )
            db.add(data_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def load_historical_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    db = SessionLocal()
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        data = db.query(HistoricalData).filter(
            HistoricalData.symbol == symbol,
            HistoricalData.interval == interval,
            HistoricalData.timestamp >= start,
            HistoricalData.timestamp <= end
        ).order_by(HistoricalData.timestamp).all()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame([
            {
                'Date': entry.timestamp,
                'Open': entry.open_price,
                'High': entry.high_price,
                'Low': entry.low_price,
                'Close': entry.close_price,
                'Volume': entry.volume
            }
            for entry in data
        ])
        df.set_index('Date', inplace=True)
        return df
    finally:
        db.close()

def fetch_historical_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene datos históricos de un activo.
    Primero intenta cargar desde la base de datos. Si no están, los genera y guarda.
    """
    init_db()
    
    # Intentar cargar desde la base de datos
    df = load_historical_data(symbol, interval, start_date, end_date)
    if not df.empty:
        print(f"Datos cargados desde DB para {symbol} ({interval}) de {start_date} a {end_date}")
        return df

    print(f"Generando datos de ejemplo para {symbol} ({interval}) de {start_date} a {end_date}")
    # Generar fechas de ejemplo
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Simplificación: asume intervalo diario para generar datos
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    
    # Generar datos OHLCV de ejemplo
    np.random.seed(42) # Para reproducibilidad
    open_prices = np.random.uniform(100, 200, len(dates))
    high_prices = open_prices + np.random.uniform(1, 5, len(dates))
    low_prices = open_prices - np.random.uniform(1, 5, len(dates))
    close_prices = open_prices + np.random.uniform(-2, 2, len(dates))
    volumes = np.random.randint(1000, 10000, len(dates))

    data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    data.set_index('Date', inplace=True)
    
    # Guardar en la base de datos
    save_historical_data(data, symbol, interval)
    return data
