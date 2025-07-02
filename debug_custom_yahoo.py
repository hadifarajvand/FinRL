#!/usr/bin/env python3
"""
Debug script for Custom Yahoo Finance Processor
"""

import sys
import logging

# Add the finrl directory to the path
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_custom_yahoo():
    """Test the custom Yahoo Finance processor."""
    try:
        from finrl.meta.data_processors.processor_custom_yahoo import CustomYahooFinanceProcessor
        
        logger.info("Creating CustomYahooFinanceProcessor...")
        processor = CustomYahooFinanceProcessor(max_retries=2, retry_delay=0.5)
        
        # Test with a single ticker
        ticker = "AAPL"
        start_date = "2024-01-01"
        end_date = "2024-01-10"
        
        logger.info(f"Testing download for {ticker}...")
        df = processor.download_data([ticker], start_date, end_date)
        
        if not df.empty:
            logger.info(f"✅ Success! Downloaded {len(df)} records")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return True
        else:
            logger.error("❌ No data retrieved")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_custom_yahoo()
    sys.exit(0 if success else 1) 