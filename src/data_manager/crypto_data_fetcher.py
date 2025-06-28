import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

class CryptoDataFetcher:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("data_fetcher.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_exchange(self, exchange_name):
        """Inicializa exchange con manejo de errores"""
        self.logger.info(f"Intentando inicializar exchange: {exchange_name}")
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 3000
            })
            exchange.load_markets()
            self.logger.info(f"Exchange {exchange_name} inicializado y mercados cargados.")
            return exchange
        except Exception as e:
            self.logger.error(f"Error inicializando {exchange_name}: {str(e)}")
            return None

    def fetch_ohlcv(self, exchange, symbol, timeframe, since=None, limit=None):
        """Obtiene datos OHLCV con reintentos"""
        attempt = 0
        while attempt < self.config['max_retries']:
            try:
                self.logger.info(f"Intentando fetch_ohlcv para {exchange.id} {symbol} {timeframe} (intento {attempt + 1})")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if not ohlcv:
                    self.logger.warning(f"No se recibieron datos OHLCV para {exchange.id} {symbol} {timeframe} desde {since}.")
                    return pd.DataFrame() # Return empty DataFrame if no data
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.logger.info(f"Datos OHLCV obtenidos para {exchange.id} {symbol} {timeframe}. Filas: {len(df)}")
                return df
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                wait_time = self.config['backoff_base'] * (2 ** attempt)
                self.logger.warning(f"Error ({e}) para {exchange.id} {symbol} {timeframe}, reintentando en {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            except Exception as e:
                self.logger.error(f"Error crítico en fetch_ohlcv para {exchange.id} {symbol} {timeframe}: {str(e)}")
                return pd.DataFrame()
        self.logger.error(f"Fallo después de {self.config['max_retries']} reintentos para {exchange.id} {symbol} {timeframe}")
        return pd.DataFrame()

    def fetch_yfinance_data(self, symbol, period, interval):
        """Obtiene datos de yfinance"""
        self.logger.info(f"Intentando fetch_yfinance_data para {symbol} {period} {interval}")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                self.logger.warning(f"No se recibieron datos de yfinance para {symbol} {period} {interval}.")
            else:
                df.index = df.index.tz_localize(pytz.utc)
                self.logger.info(f"Datos yfinance obtenidos para {symbol} {period} {interval}. Filas: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Error con yfinance ({symbol} {period} {interval}): {str(e)}")
            return pd.DataFrame()

    def save_to_csv(self, df, path):
        """Guarda datos con manejo de errores"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if df.empty:
                self.logger.warning(f"DataFrame vacío, no se guardará en: {path}")
                return
            df.to_csv(path)
            self.logger.info(f"Datos guardados: {path}")
        except Exception as e:
            self.logger.error(f"Error guardando CSV en {path}: {str(e)}")

    def get_time_ranges(self, start_date, end_date, timeframe):
        """Genera rangos temporales para descarga segmentada"""
        delta = self.calculate_timeframe_delta(timeframe)
        current = start_date
        ranges = []
        
        while current < end_date:
            ranges.append(current)
            current += delta
        return ranges

    def calculate_timeframe_delta(self, timeframe):
        """Calcula delta de tiempo según timeframe"""
        # Estos deltas son aproximados y deben ser ajustados a los límites de la API del exchange
        # ccxt.fetch_ohlcv por defecto suele traer 1000 velas.
        # Un delta de 1000 minutos para '1m' significa que cada llamada intentará obtener 1000 velas de 1 minuto.
        # Si el exchange solo da 1000 velas por llamada, esto es eficiente.
        # Si el exchange da más, podríamos ajustar el delta para cubrir más tiempo por llamada.
        timeframe_units = {
            '1m': timedelta(minutes=1000), # ~16.6 horas
            '5m': timedelta(minutes=5000), # ~3.4 días
            '15m': timedelta(hours=250),   # ~10.4 días
            '1h': timedelta(days=40),      # 40 días
            '1d': timedelta(days=365),     # 1 año
            '1w': timedelta(weeks=520)     # 10 años
        }
        return timeframe_units.get(timeframe, timedelta(days=30))

    def fetch_multiple_symbols(self):
        """Descarga simultánea para múltiples símbolos y timeframes"""
        self.logger.info("Iniciando descarga de múltiples símbolos y timeframes.")
        results = {}
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            
            # Fuente: Exchanges CCXT
            for exchange_name in self.config['exchanges']:
                exchange = self.initialize_exchange(exchange_name)
                if not exchange:
                    self.logger.error(f"Saltando exchange {exchange_name} debido a error de inicialización.")
                    continue
                    
                for symbol in self.config['symbols']:
                    # ccxt symbols are often uppercase, e.g., 'BTC/USDT'
                    # Check if the symbol exists in the exchange's markets
                    if symbol not in exchange.symbols:
                        self.logger.warning(f"Símbolo {symbol} no encontrado en {exchange_name}. Saltando.")
                        continue
                        
                    for timeframe in self.config['timeframes']:
                        self.logger.info(f"Añadiendo tarea para {exchange_name} {symbol} {timeframe}")
                        futures.append(executor.submit(
                            self.fetch_historical_data,
                            exchange, symbol, timeframe
                        ))

            # Fuente: yfinance
            for symbol in self.config['yfinance_symbols']:
                for interval in self.config['yfinance_intervals']:
                    self.logger.info(f"Añadiendo tarea para yfinance {symbol} {interval}")
                    # Modificar fetch_yfinance_data para que devuelva un dict con metadata
                    futures.append(executor.submit(
                        self._fetch_yfinance_data_with_metadata,
                        symbol, self.config['period'], interval
                    ))
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, dict) and 'source' in result: # Resultado con metadata
                        if result['source'] == 'yfinance':
                            if not result['data'].empty:
                                key = f"yfinance_{result['symbol'].replace('-', '_')}_{result['interval']}"
                                self.save_to_csv(result['data'], os.path.join(self.config['backup_dir'], f"{key}.csv"))
                            else:
                                self.logger.warning(f"Resultado de yfinance vacío para {result.get('symbol')} {result.get('interval')}.")
                        elif result['source'] == 'ccxt':
                            if not result['data'].empty:
                                key = f"{result['exchange']}_{result['symbol'].replace('/', '_')}_{result['timeframe']}"
                                self.save_to_csv(result['data'], os.path.join(self.config['backup_dir'], f"{key}.csv"))
                            else:
                                self.logger.warning(f"Resultado de CCXT vacío para {result.get('exchange')} {result.get('symbol')} {result.get('timeframe')}.")
                    else:
                        self.logger.error(f"Tipo de resultado inesperado o sin metadata de future: {type(result)}")
                except Exception as e:
                    self.logger.error(f"Error al procesar el resultado de una tarea: {e}")
                    
        self.logger.info("Descarga de múltiples símbolos y timeframes completada.")
        return results

    def _fetch_yfinance_data_with_metadata(self, symbol, period, interval):
        """Wrapper para fetch_yfinance_data que devuelve metadata."""
        data = self.fetch_yfinance_data(symbol, period, interval)
        return {
            'source': 'yfinance',
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'data': data
        }

    def fetch_historical_data(self, exchange, symbol, timeframe):
        """Descarga datos históricos segmentados"""
        all_data = pd.DataFrame()
        
        # Convertir start_date y end_date a UTC si no lo están
        start_date_utc = self.config['start_date'].astimezone(pytz.utc) if self.config['start_date'].tzinfo else self.config['start_date'].replace(tzinfo=pytz.utc)
        end_date_utc = self.config['end_date'].astimezone(pytz.utc) if self.config['end_date'].tzinfo else self.config['end_date'].replace(tzinfo=pytz.utc)

        time_ranges = self.get_time_ranges(
            start_date_utc,
            end_date_utc,
            timeframe
        )
        
        if not time_ranges:
            self.logger.warning(f"No se generaron rangos de tiempo para {symbol} {timeframe}. Verifique las fechas de inicio/fin y el delta del timeframe.")
            
        for start_time in time_ranges:
            data = self.fetch_ohlcv(
                exchange,
                symbol,
                timeframe,
                since=int(start_time.timestamp() * 1000)
            )
            
            if not data.empty:
                all_data = pd.concat([all_data, data])
            else:
                self.logger.warning(f"No se devolvieron datos de fetch_ohlcv para {symbol} {timeframe} comenzando en {start_time}.")
                
        return {
            'source': 'ccxt',
            'exchange': exchange.id,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': all_data
        }

    def real_time_update(self):
        """Actualización continua en tiempo real (ejecutar en segundo plano)"""
        self.logger.info("Iniciando actualización en tiempo real.")
        while True:
            try:
                self.fetch_multiple_symbols()
                self.logger.info(f"Esperando {self.config['refresh_interval']} segundos para la próxima actualización.")
                time.sleep(self.config['refresh_interval'])
            except KeyboardInterrupt:
                self.logger.info("Detenido por el usuario")
                break
            except Exception as e:
                self.logger.error(f"Error en actualización: {str(e)}")
                time.sleep(60)

# Configuración de ejemplo
config = {
    'exchanges': ['binance', 'kucoin', 'coinbasepro'],
    'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'timeframes': ['1m', '15m', '1h', '1d'],
    'yfinance_symbols': ['BTC-USD', 'ETH-USD'],
    'yfinance_intervals': ['1m', '1h'],
    'period': '2y',
    'start_date': datetime(2020, 1, 1, tzinfo=pytz.utc), # Convert to timezone-aware UTC
    'end_date': datetime.now(pytz.utc), # Use timezone-aware UTC
    'max_retries': 5,
    'backoff_base': 2,
    'max_workers': 8,
    'refresh_interval': 300,  # 5 minutos
    'backup_dir': 'data' # Cambiado a 'data' para que coincida con la estructura propuesta
}

# Uso del módulo
if __name__ == '__main__':
    # Asegurarse de que el directorio de backup exista
    os.makedirs(config['backup_dir'], exist_ok=True)
    
    fetcher = CryptoDataFetcher(config)

    # Para descarga única
    data = fetcher.fetch_multiple_symbols()

    # Para modo continuo
    # fetcher.real_time_update()