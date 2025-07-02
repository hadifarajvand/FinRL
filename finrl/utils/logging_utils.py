"""
Logging utilities for FinRL test operations
"""

import logging
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class TestLogger:
    """Comprehensive logger for test operations"""
    
    def __init__(self, log_dir: str = "./test_logs", model_name: str = "unknown", 
                 test_name: str = "test", log_level: str = "INFO"):
        """
        Initialize the test logger
        
        Args:
            log_dir: Directory to store log files
            model_name: Name of the model being tested
            test_name: Name of the test
            log_level: Logging level
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.test_name = test_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"finrl.test.{model_name}.{test_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        log_file = os.path.join(log_dir, f"{model_name}_{test_name}_{self.timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Store test results
        self.test_results = {
            "model_name": model_name,
            "test_name": test_name,
            "timestamp": self.timestamp,
            "test_config": {},
            "performance_metrics": {},
            "trading_actions": [],
            "portfolio_values": [],
            "errors": []
        }
        
        self.logger.info(f"Test logger initialized for {model_name} - {test_name}")
    
    def log_test_config(self, config: Dict[str, Any]):
        """Log test configuration"""
        self.test_results["test_config"] = config
        self.logger.info("=== Test Configuration ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 30)
    
    def log_data_info(self, price_array: np.ndarray, tech_array: np.ndarray, 
                     turbulence_array: np.ndarray, ticker_list: List[str]):
        """Log information about the test data"""
        self.logger.info("=== Data Information ===")
        self.logger.info(f"Price array shape: {price_array.shape}")
        self.logger.info(f"Technical indicators array shape: {tech_array.shape}")
        self.logger.info(f"Turbulence array shape: {turbulence_array.shape}")
        self.logger.info(f"Number of tickers: {len(ticker_list)}")
        self.logger.info(f"Tickers: {ticker_list}")
        self.logger.info(f"Total trading days: {price_array.shape[0]}")
        self.logger.info("=" * 30)
    
    def log_model_loading(self, model_path: str, model_name: str):
        """Log model loading information"""
        self.logger.info("=== Model Loading ===")
        self.logger.info(f"Loading {model_name} model from: {model_path}")
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            self.logger.info(f"Model file exists, size: {file_size:.2f} MB")
        else:
            self.logger.warning(f"Model file not found: {model_path}")
        
        self.logger.info("=" * 30)
    
    def log_step_info(self, step: int, action: np.ndarray, reward: float, 
                     total_asset: float, portfolio_value: float, done: bool):
        """Log information for each trading step"""
        if step % 50 == 0 or done:  # Log every 50 steps or at the end
            self.logger.info(f"Step {step}: Reward={reward:.4f}, "
                           f"Total Asset=${total_asset:.2f}, "
                           f"Portfolio Value=${portfolio_value:.2f}")
        
        # Store for detailed analysis
        self.test_results["trading_actions"].append({
            "step": step,
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "reward": reward,
            "total_asset": total_asset,
            "portfolio_value": portfolio_value,
            "done": done
        })
        
        self.test_results["portfolio_values"].append(portfolio_value)
    
    def log_performance_metrics(self, final_return: float, total_steps: int, 
                              initial_asset: float, final_asset: float):
        """Log final performance metrics"""
        self.logger.info("=== Performance Metrics ===")
        self.logger.info(f"Total trading steps: {total_steps}")
        self.logger.info(f"Initial portfolio value: ${initial_asset:.2f}")
        self.logger.info(f"Final portfolio value: ${final_asset:.2f}")
        self.logger.info(f"Total return: {final_return:.4f} ({final_return*100:.2f}%)")
        
        # Calculate additional metrics
        portfolio_values = np.array(self.test_results["portfolio_values"])
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            self.logger.info(f"Annualized volatility: {volatility:.4f}")
            self.logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
            self.logger.info(f"Maximum drawdown: {max_drawdown:.4f}")
            
            self.test_results["performance_metrics"] = {
                "total_steps": total_steps,
                "initial_asset": initial_asset,
                "final_asset": final_asset,
                "total_return": final_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }
        
        self.logger.info("=" * 30)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors during testing"""
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg)
        self.test_results["errors"].append({
            "context": context,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def save_results(self):
        """Save test results to files"""
        # Save JSON results
        results_file = os.path.join(self.log_dir, 
                                   f"{self.model_name}_{self.test_name}_{self.timestamp}_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save portfolio values as CSV
        if self.test_results["portfolio_values"]:
            portfolio_df = pd.DataFrame({
                'step': range(len(self.test_results["portfolio_values"])),
                'portfolio_value': self.test_results["portfolio_values"]
            })
            portfolio_file = os.path.join(self.log_dir, 
                                         f"{self.model_name}_{self.test_name}_{self.timestamp}_portfolio.csv")
            portfolio_df.to_csv(portfolio_file, index=False)
        
        # Create performance plot
        if len(self.test_results["portfolio_values"]) > 1:
            self._create_performance_plot()
        
        self.logger.info(f"Test results saved to: {self.log_dir}")
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    def _create_performance_plot(self):
        """Create performance visualization"""
        try:
            portfolio_values = np.array(self.test_results["portfolio_values"])
            steps = np.arange(len(portfolio_values))
            
            plt.figure(figsize=(12, 8))
            
            # Portfolio value over time
            plt.subplot(2, 1, 1)
            plt.plot(steps, portfolio_values, 'b-', linewidth=2)
            plt.title(f'{self.model_name} - Portfolio Value Over Time')
            plt.xlabel('Trading Step')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            
            # Returns distribution
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                plt.subplot(2, 1, 2)
                plt.hist(returns, bins=50, alpha=0.7, color='green')
                plt.title('Returns Distribution')
                plt.xlabel('Return')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.log_dir, 
                                    f"{self.model_name}_{self.test_name}_{self.timestamp}_performance.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance plot saved to: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create performance plot: {e}")
    
    def log_test_completion(self, success: bool = True):
        """Log test completion"""
        if success:
            self.logger.info("=== Test Completed Successfully ===")
        else:
            self.logger.error("=== Test Failed ===")
        
        self.logger.info(f"Test duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)


def setup_test_logging(log_dir: str = "./test_logs", level: str = "INFO"):
    """Setup basic logging configuration for tests"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("finrl.test") 