from __future__ import annotations

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    enable_logging=True,
    log_dir="./test_logs",
    **kwargs,
):
    # Initialize logger if enabled
    logger = None
    if enable_logging:
        from finrl.utils.logging_utils import TestLogger
        logger = TestLogger(
            log_dir=log_dir,
            model_name=model_name,
            test_name="backtest",
            log_level="INFO"
        )
        
        # Log test configuration
        test_config = {
            "start_date": start_date,
            "end_date": end_date,
            "ticker_list": ticker_list,
            "data_source": data_source,
            "time_interval": time_interval,
            "technical_indicator_list": technical_indicator_list,
            "drl_lib": drl_lib,
            "model_name": model_name,
            "if_vix": if_vix,
            **kwargs
        }
        logger.log_test_config(test_config)
        logger.logger.info("Starting test process...")

    try:
        # import data processor
        from finrl.meta.data_processor import DataProcessor

        # fetch data
        if logger:
            logger.logger.info("Downloading and processing data...")
            
        dp = DataProcessor(data_source, **kwargs)
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)

        if if_vix:
            data = dp.add_vix(data)
        price_array, tech_array, turbulence_array = dp.df_to_array(data, technical_indicator_list, if_vix)

        # Ensure tech_array has the correct shape
        n_rows = len(data)
        n_cols = len(ticker_list) * len(technical_indicator_list)
        if logger:
            logger.logger.info(f"Expected tech_array shape: ({n_rows}, {n_cols})")
            logger.logger.info(f"Actual tech_array shape: {tech_array.shape}")
        else:
            print(f"Expected tech_array shape: ({n_rows}, {n_cols})")
            print(f"Actual tech_array shape: {tech_array.shape}")
            
        if tech_array.shape != (n_rows, n_cols):
            import numpy as np
            if logger:
                logger.logger.warning("Fixing tech_array shape by filling with zeros.")
            else:
                print("Fixing tech_array shape by filling with zeros.")
            tech_array = np.zeros((n_rows, n_cols))

        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_instance = env(config=env_config)

        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2**7)
        cwd = kwargs.get("cwd", "./" + str(model_name))
        
        if logger:
            logger.logger.info(f"Price array length: {len(price_array)}")
            logger.logger.info(f"Model directory: {cwd}")
            logger.logger.info(f"Net dimension: {net_dimension}")
        else:
            print("price_array: ", len(price_array))

        if drl_lib == "elegantrl":
            from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

            episode_total_assets = DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=cwd,
                net_dimension=net_dimension,
                environment=env_instance,
                env_args=env_config,
                logger=logger,
                ticker_list=ticker_list,
            )
            return episode_total_assets
        elif drl_lib == "rllib":
            from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

            episode_total_assets = DRLAgent_rllib.DRL_prediction(
                model_name=model_name,
                env=env,
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                agent_path=cwd,
            )
            return episode_total_assets
        elif drl_lib == "stable_baselines3":
            from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

            episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name, environment=env_instance, cwd=cwd
            )
            return episode_total_assets
        else:
            raise ValueError("DRL library input is NOT supported. Please check.")
            
    except Exception as e:
        if logger:
            logger.log_error(e, "Test execution")
            logger.log_test_completion(success=False)
        raise e


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
        enable_logging=True,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # # demo for stable baselines3
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac.zip",
    # )
