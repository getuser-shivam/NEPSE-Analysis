"""
Data Manager Module for NEPSE Analysis Tool
Handles all data operations, validation, and management
"""

import pandas as pd
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging


class DataManager:
    """Handles all data operations for the NEPSE Analysis application"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stock_data = {}
        self.portfolio = {}
        self.watchlist = []
        self.portfolio_file = 'portfolio.pkl'
        self.watchlist_file = 'watchlist.pkl'
        self.data_file = 'nepse_data.pkl'
        
    def validate_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not symbol.strip():
            return False
        
        # Basic validation: alphanumeric, 1-10 characters
        symbol = symbol.strip().upper()
        import re
        return bool(re.match(r'^[A-Z0-9]{1,10}$', symbol))
        
    def validate_date(self, date_str: str) -> bool:
        """Validate date format and range"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            # Check if date is not too far in the past or future
            min_date = datetime.now() - timedelta(days=365 * 10)  # 10 years ago
            max_date = datetime.now() + timedelta(days=30)  # 30 days in future
            return min_date <= date <= max_date
        except ValueError:
            return False
            
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate that start_date is before end_date and range is reasonable"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Start should be before end
            if start >= end:
                return False
                
            # Range should not be too large (max 10 years)
            if (end - start).days > 365 * 10:
                return False
                
            return True
        except ValueError:
            return False
            
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate the quality and integrity of fetched data"""
        try:
            if data is None or data.empty:
                self.logger.warning(f"No data received for {symbol}")
                return False
                
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return False
                
            # Check for null values in critical columns
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Null values found in {symbol} data: {null_counts[null_counts > 0].to_dict()}")
                
            # Check for reasonable price values
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    self.logger.warning(f"Non-positive prices found in {symbol} {col} data")
                    
            # Check for reasonable volume
            if (data['Volume'] < 0).any():
                self.logger.warning(f"Negative volume found in {symbol} data")
                
            # Check data freshness
            latest_date = data.index[-1] if len(data) > 0 else None
            if latest_date:
                days_old = (datetime.now() - latest_date).days
                if days_old > 7:  # Data older than 7 days
                    self.logger.warning(f"Data for {symbol} is {days_old} days old")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error for {symbol}: {e}")
            return False
            
    def add_to_portfolio(self, symbol: str, shares: int, buy_price: float, current_price: float) -> None:
        """Add stock to portfolio"""
        try:
            self.portfolio[symbol] = {
                'shares': shares,
                'buy_price': buy_price,
                'current_price': current_price,
                'last_updated': datetime.now()
            }
            self.logger.info(f"Added {symbol} to portfolio: {shares} shares at {buy_price}")
        except Exception as e:
            self.logger.error(f"Failed to add to portfolio: {e}")
            raise
            
    def remove_from_portfolio(self, symbol: str) -> bool:
        """Remove stock from portfolio"""
        try:
            if symbol in self.portfolio:
                del self.portfolio[symbol]
                self.logger.info(f"Removed {symbol} from portfolio")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove from portfolio: {e}")
            return False
            
    def add_to_watchlist(self, symbol: str) -> bool:
        """Add stock to watchlist"""
        try:
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                self.logger.info(f"Added {symbol} to watchlist")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add to watchlist: {e}")
            return False
            
    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove stock from watchlist"""
        try:
            if symbol in self.watchlist:
                self.watchlist.remove(symbol)
                self.logger.info(f"Removed {symbol} from watchlist")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove from watchlist: {e}")
            return False
            
    def update_portfolio_prices(self, price_updates: Dict[str, float]) -> None:
        """Update current prices in portfolio"""
        try:
            updated_count = 0
            for symbol, new_price in price_updates.items():
                if symbol in self.portfolio:
                    self.portfolio[symbol]['current_price'] = new_price
                    self.portfolio[symbol]['last_updated'] = datetime.now()
                    updated_count += 1
                    
            self.logger.info(f"Updated prices for {updated_count} portfolio stocks")
        except Exception as e:
            self.logger.error(f"Failed to update portfolio prices: {e}")
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics"""
        try:
            if not self.portfolio:
                return {
                    'total_investment': 0.0,
                    'total_value': 0.0,
                    'total_gain_loss': 0.0,
                    'total_return_pct': 0.0,
                    'num_stocks': 0
                }
                
            total_investment = 0.0
            total_value = 0.0
            
            for symbol, data in self.portfolio.items():
                investment = data['shares'] * data['buy_price']
                current_value = data['shares'] * data['current_price']
                total_investment += investment
                total_value += current_value
                
            total_gain_loss = total_value - total_investment
            total_return_pct = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0.0
            
            return {
                'total_investment': total_investment,
                'total_value': total_value,
                'total_gain_loss': total_gain_loss,
                'total_return_pct': total_return_pct,
                'num_stocks': len(self.portfolio)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
            
    def save_data(self) -> List[str]:
        """Save all data to files"""
        saved_files = []
        
        try:
            # Save portfolio
            if self.portfolio:
                with open(self.portfolio_file, 'wb') as f:
                    pickle.dump(self.portfolio, f)
                saved_files.append(self.portfolio_file)
                self.logger.info(f"Portfolio saved: {len(self.portfolio)} stocks")
        except Exception as e:
            self.logger.error(f"Failed to save portfolio: {e}")
            
        try:
            # Save watchlist
            if self.watchlist:
                with open(self.watchlist_file, 'wb') as f:
                    pickle.dump(self.watchlist, f)
                saved_files.append(self.watchlist_file)
                self.logger.info(f"Watchlist saved: {len(self.watchlist)} stocks")
        except Exception as e:
            self.logger.error(f"Failed to save watchlist: {e}")
            
        try:
            # Save stock data (optional - can be large)
            if self.stock_data:
                with open(self.data_file, 'wb') as f:
                    pickle.dump(self.stock_data, f)
                saved_files.append(self.data_file)
                self.logger.info(f"Stock data saved: {len(self.stock_data)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to save stock data: {e}")
            
        return saved_files
        
    def load_data(self) -> None:
        """Load all data from files"""
        try:
            # Load portfolio
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'rb') as f:
                    self.portfolio = pickle.load(f)
                self.logger.info(f"Portfolio loaded: {len(self.portfolio)} stocks")
        except Exception as e:
            self.logger.error(f"Failed to load portfolio: {e}")
            
        try:
            # Load watchlist
            if os.path.exists(self.watchlist_file):
                with open(self.watchlist_file, 'rb') as f:
                    self.watchlist = pickle.load(f)
                self.logger.info(f"Watchlist loaded: {len(self.watchlist)} stocks")
        except Exception as e:
            self.logger.error(f"Failed to load watchlist: {e}")
            
        try:
            # Load stock data
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    self.stock_data = pickle.load(f)
                self.logger.info(f"Stock data loaded: {len(self.stock_data)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to load stock data: {e}")
            
    def clear_cache(self) -> Dict[str, Any]:
        """Clear cached data and return statistics"""
        try:
            total_symbols = len(self.stock_data)
            total_data_points = sum(len(data) for data in self.stock_data.values())
            
            self.stock_data.clear()
            
            # Also clear the data file
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
                
            return {
                'symbols_cleared': total_symbols,
                'data_points_cleared': total_data_points,
                'cache_freed_mb': total_data_points * 0.001  # Rough estimate
            }
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return {'error': str(e)}
