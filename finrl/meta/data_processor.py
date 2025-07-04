from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_ccxt import CCXTEngineer
from finrl.meta.data_processors.processor_custom_yahoo import CustomYahooFinanceProcessor
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")
                self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()

        elif data_source == "custom_yahoofinance":
            self.processor = CustomYahooFinanceProcessor()

        elif data_source == "ccxt":
            self.processor = CCXTEngineer()

        else:
            raise ValueError("Data source input is NOT supported yet.")

        # Initialize variable in case it is using cache and does not use download_data() method
        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        return self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            proxy="http://127.0.0.1:2080"
        )

    def clean_data(self, df) -> pd.DataFrame:
        if hasattr(self.processor, 'clean_data'):
            df = self.processor.clean_data(df)
        else:
            # For CCXT, data is already clean
            pass
        return df

    def add_technical_indicator(
        self, df, tech_indicator_list
    ) -> pd.DataFrame:
        if hasattr(self.processor, 'add_technical_indicator'):
            df = self.processor.add_technical_indicator(df, tech_indicator_list)
        elif hasattr(self.processor, 'add_technical_indicators'):
            # For CCXT, use add_technical_indicators method
            df = self.processor.add_technical_indicators(
                df, 
                pair_list=df.columns.get_level_values(0).unique().tolist(),
                tech_indicator_list=tech_indicator_list
            )
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        if hasattr(self.processor, 'add_turbulence'):
            df = self.processor.add_turbulence(df)
        return df

    def add_vix(self, df) -> pd.DataFrame:
        if hasattr(self.processor, 'add_vix'):
            df = self.processor.add_vix(df)
        return df

    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)
        return df

    def df_to_array(
        self, df, tech_indicator_list, if_vix
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if hasattr(self.processor, 'df_to_array'):
            price_array, tech_array, turbulence_array = self.processor.df_to_array(
                df, tech_indicator_list, if_vix
            )
        elif hasattr(self.processor, 'df_to_ary'):
            # For CCXT, use df_to_ary method
            price_array, tech_array, date_ary = self.processor.df_to_ary(
                df,
                pair_list=df.columns.get_level_values(0).unique().tolist(),
                tech_indicator_list=tech_indicator_list
            )
            # CCXT doesn't have turbulence, so create empty array
            turbulence_array = np.zeros(len(price_array))
        return price_array, tech_array, turbulence_array
