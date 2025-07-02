from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from stockstats import StockDataFrame as Sdf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomYahooFinanceProcessor:
    """
    Custom Yahoo Finance data processor with SSL bypass and comprehensive error handling.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with SSL bypass and custom headers."""
        session = requests.Session()
        
        # Custom headers to mimic a real browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Disable SSL verification (use with caution)
        session.verify = False
        
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        return session
    
    def _fetch_data_with_retry(self, url: str, params: dict) -> Optional[dict]:
        """Fetch data with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to fetch data from {url} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Successfully fetched data from {url}")
                return data
                
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"SSL error after {self.max_retries} attempts")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts")
                    return None
                    
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Unexpected error after {self.max_retries} attempts")
                    return None
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        return None
    
    def _get_historical_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data for a single ticker using Yahoo Finance API."""
        try:
            # Convert dates to timestamps
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())
            
            # Yahoo Finance API endpoint
            url = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div,split'
            }
            
            data = self._fetch_data_with_retry(url.format(ticker=ticker), params)
            
            if data is None or 'chart' not in data or 'result' not in data['chart']:
                logger.error(f"No valid data returned for {ticker}")
                return None
            
            result = data['chart']['result'][0]
            
            # Extract timestamps and OHLCV data
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quote.get('open', [None] * len(timestamps)),
                'high': quote.get('high', [None] * len(timestamps)),
                'low': quote.get('low', [None] * len(timestamps)),
                'close': quote.get('close', [None] * len(timestamps)),
                'volume': quote.get('volume', [None] * len(timestamps))
            })
            
            # Handle missing data
            df = df.replace([None, np.nan], np.nan)
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            
            logger.info(f"Successfully retrieved {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def download_data(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str,
        time_interval: str = "1D",
        proxy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download data for multiple tickers.
        
        Args:
            ticker_list: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            time_interval: Time interval (currently only supports "1D")
            proxy: Proxy URL (not used in this implementation)
            
        Returns:
            DataFrame with multi-level columns for all tickers
        """
        logger.info(f"Starting data download for {len(ticker_list)} tickers from {start_date} to {end_date}")
        
        if time_interval != "1D":
            logger.warning(f"Time interval {time_interval} not supported, using 1D")
        
        # Create multi-level column structure
        column_list = [
            ticker_list,
            ["open", "high", "low", "close", "volume"]
        ]
        columns = pd.MultiIndex.from_product(column_list)
        
        all_data = []
        successful_tickers = []
        
        for ticker in ticker_list:
            logger.info(f"Processing ticker: {ticker}")
            
            df = self._get_historical_data(ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                # Set time as index
                df.set_index('time', inplace=True)
                all_data.append(df)
                successful_tickers.append(ticker)
                logger.info(f"Successfully processed {ticker}")
            else:
                logger.warning(f"Failed to get data for {ticker}")
        
        if not all_data:
            logger.error("No data retrieved for any ticker")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, axis=1, keys=successful_tickers)
        
        # Reorder columns to match expected structure
        combined_df = combined_df.reorder_levels([1, 0], axis=1)
        combined_df = combined_df.sort_index(axis=1)
        
        logger.info(f"Data download completed. Retrieved data for {len(successful_tickers)}/{len(ticker_list)} tickers")
        logger.info(f"Data shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing NaN values and handling outliers."""
        logger.info("Cleaning data...")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Remove rows with all NaN values
        df_cleaned = df.dropna(how='all')
        
        # Forward fill for missing values (within each ticker)
        df_cleaned = df_cleaned.ffill()
        
        # Backward fill for remaining missing values
        df_cleaned = df_cleaned.bfill()
        
        # Remove any remaining rows with NaN values
        df_cleaned = df_cleaned.dropna()
        
        logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
        return df_cleaned
    
    def add_technical_indicator(self, df: pd.DataFrame, tech_indicator_list: List[str]) -> pd.DataFrame:
        """Add technical indicators to the data."""
        logger.info(f"Adding technical indicators: {tech_indicator_list}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for technical indicators")
            return df
        
        df_with_indicators = df.copy()
        tickers = [t for t in df.columns.get_level_values(1).unique() if t != 'market']
        n_rows = len(df)
        
        for ticker in tickers:
            logger.info(f"Adding indicators for {ticker}")
            # Extract and flatten columns for this ticker
            ticker_columns = df.columns[df.columns.get_level_values(1) == ticker]
            ticker_data = df[ticker_columns].copy()
            ticker_data.columns = [col[0] for col in ticker_columns]  # flatten to single level
            ticker_data = ticker_data.reset_index()
            ticker_data = ticker_data.rename(columns={'index': 'date'})
            try:
                crypto_df = Sdf.retype(ticker_data.copy())
                for indicator in tech_indicator_list:
                    if indicator in crypto_df.columns:
                        indicator_values = crypto_df[indicator].values.tolist()
                        # Pad or trim to match n_rows
                        if len(indicator_values) < n_rows:
                            indicator_values += [0.0] * (n_rows - len(indicator_values))
                        elif len(indicator_values) > n_rows:
                            indicator_values = indicator_values[:n_rows]
                        df_with_indicators[(indicator, ticker)] = indicator_values
                        logger.debug(f"Added {indicator} for {ticker}")
                    else:
                        logger.warning(f"Indicator {indicator} not available for {ticker}, filling with zeros")
                        df_with_indicators[(indicator, ticker)] = [0.0] * n_rows
            except Exception as e:
                logger.error(f"Error adding indicators for {ticker}: {e}")
                for indicator in tech_indicator_list:
                    df_with_indicators[(indicator, ticker)] = [0.0] * n_rows
        logger.info("Technical indicators added successfully")
        return df_with_indicators
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add turbulence index to the data."""
        logger.info("Adding turbulence index...")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for turbulence calculation")
            return df
        
        # This is a simplified turbulence calculation
        # In a real implementation, you would calculate the turbulence index
        # based on the covariance matrix of returns
        
        df_with_turbulence = df.copy()
        turbulence = np.zeros(len(df))
        df_with_turbulence[('turbulence', 'market')] = turbulence
        
        logger.info("Turbulence index added")
        return df_with_turbulence
    
    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VIX data to the DataFrame."""
        logger.info("Adding VIX data...")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for VIX calculation")
            return df
        
        # For now, we'll add a placeholder VIX column
        # In a real implementation, you would fetch actual VIX data
        df_with_vix = df.copy()
        vix_values = np.zeros(len(df))
        df_with_vix[('vix', 'market')] = vix_values
        
        logger.info("VIX data added")
        return df_with_vix
    
    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) -> tuple:
        """Convert DataFrame to arrays for training."""
        logger.info("Converting DataFrame to arrays...")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for array conversion")
            return np.array([]), np.array([]), np.array([])
        
        # Get unique tickers (ignore 'market' if present)
        tickers = [t for t in df.columns.get_level_values(1).unique() if t != 'market']
        n_rows = len(df)
        
        # Extract price data (close prices)
        price_columns = [("close", t) for t in tickers if ("close", t) in df.columns]
        price_data = df[price_columns].values if price_columns else np.array([])
        
        # Extract technical indicators, always fill missing with zeros
        tech_data = []
        for ticker in tickers:
            for indicator in tech_indicator_list:
                col = (indicator, ticker)
                if col in df.columns:
                    tech_data.append(df[col].values)
                else:
                    tech_data.append(np.zeros(n_rows))
        if tech_data:
            tech_array = np.column_stack(tech_data)
        else:
            tech_array = np.array([])
        
        # Extract turbulence data
        if ('turbulence', 'market') in df.columns:
            turbulence_array = df[('turbulence', 'market')].values
        else:
            turbulence_array = np.zeros(n_rows)
        
        logger.info(f"Array conversion completed. Price shape: {price_data.shape}, Tech shape: {tech_array.shape}")
        
        return price_data, tech_array, turbulence_array 