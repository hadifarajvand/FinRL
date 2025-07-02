#!/usr/bin/env python3
"""
Test script for CustomYahooFinanceProcessor
"""

import logging
import sys
from datetime import datetime, timedelta

# Add the finrl directory to the path
sys.path.append('.')

from finrl.meta.data_processors.processor_custom_yahoo import CustomYahooFinanceProcessor
from finrl.config_tickers import DOW_30_TICKER

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_ticker():
    """Test downloading data for a single ticker."""
    logger.info("=== Testing Single Ticker Download ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=3, retry_delay=1.0)
    
    # Test with a well-known stock
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    logger.info(f"Testing download for {ticker} from {start_date} to {end_date}")
    
    try:
        df = processor.download_data([ticker], start_date, end_date)
        
        if not df.empty:
            logger.info(f"✅ Success! Downloaded {len(df)} records for {ticker}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"First few rows:\n{df.head()}")
            return True
        else:
            logger.error(f"❌ Failed! No data retrieved for {ticker}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception occurred: {e}")
        return False


def test_multiple_tickers():
    """Test downloading data for multiple tickers."""
    logger.info("=== Testing Multiple Tickers Download ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=3, retry_delay=1.0)
    
    # Test with a subset of DOW 30
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2024-01-01"
    end_date = "2024-01-15"
    
    logger.info(f"Testing download for {len(test_tickers)} tickers from {start_date} to {end_date}")
    
    try:
        df = processor.download_data(test_tickers, start_date, end_date)
        
        if not df.empty:
            logger.info(f"✅ Success! Downloaded data for {len(test_tickers)} tickers")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Available tickers: {df.columns.get_level_values(1).unique().tolist()}")
            return True
        else:
            logger.error(f"❌ Failed! No data retrieved for any ticker")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception occurred: {e}")
        return False


def test_data_cleaning():
    """Test data cleaning functionality."""
    logger.info("=== Testing Data Cleaning ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=3, retry_delay=1.0)
    
    # Get some data first
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    df = processor.download_data([ticker], start_date, end_date)
    
    if df.empty:
        logger.error("❌ No data to clean")
        return False
    
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"NaN count before cleaning: {df.isna().sum().sum()}")
    
    # Clean the data
    df_cleaned = processor.clean_data(df)
    
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    logger.info(f"NaN count after cleaning: {df_cleaned.isna().sum().sum()}")
    
    if df_cleaned.isna().sum().sum() == 0:
        logger.info("✅ Data cleaning successful!")
        return True
    else:
        logger.warning("⚠️ Some NaN values remain after cleaning")
        return False


def test_technical_indicators():
    """Test adding technical indicators."""
    logger.info("=== Testing Technical Indicators ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=3, retry_delay=1.0)
    
    # Get some data first
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    df = processor.download_data([ticker], start_date, end_date)
    
    if df.empty:
        logger.error("❌ No data for technical indicators")
        return False
    
    # Clean the data first
    df_cleaned = processor.clean_data(df)
    
    # Test technical indicators
    tech_indicators = ["macd", "rsi_30", "close_30_sma"]
    
    logger.info(f"Adding technical indicators: {tech_indicators}")
    
    try:
        df_with_indicators = processor.add_technical_indicator(df_cleaned, tech_indicators)
        
        # Check if indicators were added
        added_indicators = []
        for indicator in tech_indicators:
            if (indicator, ticker) in df_with_indicators.columns:
                added_indicators.append(indicator)
        
        logger.info(f"Successfully added indicators: {added_indicators}")
        logger.info(f"Data shape with indicators: {df_with_indicators.shape}")
        
        if len(added_indicators) > 0:
            logger.info("✅ Technical indicators added successfully!")
            return True
        else:
            logger.error("❌ No technical indicators were added")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error adding technical indicators: {e}")
        return False


def test_full_pipeline():
    """Test the full data processing pipeline."""
    logger.info("=== Testing Full Pipeline ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=3, retry_delay=1.0)
    
    # Test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2024-01-01"
    end_date = "2024-01-15"
    tech_indicators = ["macd", "rsi_30", "close_30_sma"]
    
    logger.info("Running full pipeline test...")
    
    try:
        # Step 1: Download data
        logger.info("Step 1: Downloading data...")
        df = processor.download_data(test_tickers, start_date, end_date)
        
        if df.empty:
            logger.error("❌ Pipeline failed at data download")
            return False
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data...")
        df_cleaned = processor.clean_data(df)
        
        if df_cleaned.empty:
            logger.error("❌ Pipeline failed at data cleaning")
            return False
        
        # Step 3: Add technical indicators
        logger.info("Step 3: Adding technical indicators...")
        df_with_indicators = processor.add_technical_indicator(df_cleaned, tech_indicators)
        
        # Step 4: Add turbulence
        logger.info("Step 4: Adding turbulence...")
        df_with_turbulence = processor.add_turbulence(df_with_indicators)
        
        # Step 5: Add VIX
        logger.info("Step 5: Adding VIX...")
        df_with_vix = processor.add_vix(df_with_turbulence)
        
        # Step 6: Convert to arrays
        logger.info("Step 6: Converting to arrays...")
        price_array, tech_array, turbulence_array = processor.df_to_array(
            df_with_vix, tech_indicators, if_vix=True
        )
        
        logger.info(f"✅ Full pipeline completed successfully!")
        logger.info(f"Final data shape: {df_with_vix.shape}")
        logger.info(f"Price array shape: {price_array.shape}")
        logger.info(f"Tech array shape: {tech_array.shape}")
        logger.info(f"Turbulence array shape: {turbulence_array.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid inputs."""
    logger.info("=== Testing Error Handling ===")
    
    processor = CustomYahooFinanceProcessor(max_retries=1, retry_delay=0.1)
    
    # Test with invalid ticker
    logger.info("Testing with invalid ticker...")
    df_invalid = processor.download_data(["INVALID_TICKER"], "2024-01-01", "2024-01-31")
    
    if df_invalid.empty:
        logger.info("✅ Correctly handled invalid ticker")
    else:
        logger.warning("⚠️ Unexpected data returned for invalid ticker")
    
    # Test with invalid dates
    logger.info("Testing with invalid dates...")
    try:
        df_invalid_dates = processor.download_data(["AAPL"], "invalid-date", "invalid-date")
        logger.warning("⚠️ No exception raised for invalid dates")
    except Exception as e:
        logger.info(f"✅ Correctly handled invalid dates: {e}")
    
    # Test with empty ticker list
    logger.info("Testing with empty ticker list...")
    df_empty = processor.download_data([], "2024-01-01", "2024-01-31")
    
    if df_empty.empty:
        logger.info("✅ Correctly handled empty ticker list")
    else:
        logger.warning("⚠️ Unexpected data returned for empty ticker list")
    
    return True


def main():
    """Run all tests."""
    logger.info("Starting Custom Yahoo Finance Processor Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Single Ticker", test_single_ticker),
        ("Multiple Tickers", test_multiple_tickers),
        ("Data Cleaning", test_data_cleaning),
        ("Technical Indicators", test_technical_indicators),
        ("Full Pipeline", test_full_pipeline),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 