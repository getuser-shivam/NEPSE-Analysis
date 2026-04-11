"""
Comprehensive test suite for NEPSE Analysis Application
Tests all major functionality including data fetching, portfolio management, and UI components
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
import sys
import tkinter as tk

# Add the main application directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import NEPSEAnalysisApp

class TestNEPSEAnalysisApp(unittest.TestCase):
    """Test suite for NEPSE Analysis Application"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock root window
        self.root = Mock()
        self.root.title = Mock()
        self.root.geometry = Mock()
        
        # Mock tkinter components
        with patch('tkinter.ttk.Frame'), \
             patch('tkinter.ttk.LabelFrame'), \
             patch('tkinter.ttk.Label'), \
             patch('tkinter.ttk.Entry'), \
             patch('tkinter.ttk.Button'), \
             patch('tkinter.ttk.Checkbutton'), \
             patch('tkinter.ttk.Treeview'), \
             patch('tkinter.ttk.Notebook'), \
             patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'), \
             patch('logging.basicConfig'), \
             patch('os.makedirs', return_value=None):
            
            # Create app instance with mocked dependencies
            self.app = NEPSEAnalysisApp.__new__(NEPSEAnalysisApp)
            
            # Initialize essential attributes
            self.app.root = self.root
            self.app.config = {
                'auto_save_interval': 300,
                'max_data_age_days': 7,
                'backup_enabled': True,
                'chart_style': 'seaborn-v0_8'
            }
            self.app.portfolio = {}
            self.app.watchlist = []
            self.app.stock_data = {}
            self.app.price_alerts = {}
            self.app.notifications = []
            self.app.performance_stats = {
                'data_fetches': 0,
                'chart_updates': 0,
                'errors_count': 0,
                'last_activity': datetime.now()
            }
            self.app.backup_manager = {
                'max_backups': 10,
                'backup_interval_hours': 24,
                'last_backup': None
            }
            self.app.memory_manager = {
                'max_cache_size_mb': 100,
                'auto_cleanup_interval': 300,
                'last_cleanup': datetime.now()
            }
            
            # Mock file paths
            self.app.data_file = os.path.join(self.test_dir, 'test_nepse_data.pkl')
            self.app.portfolio_file = os.path.join(self.test_dir, 'test_portfolio.pkl')
            self.app.watchlist_file = os.path.join(self.test_dir, 'test_watchlist.pkl')
            
            # Mock logger
            self.app.logger = Mock()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validate_symbol_valid(self):
        """Test valid stock symbol validation"""
        # Test valid symbols
        valid_symbols = ['NEPSE', 'NABIL', 'EBL', 'NICA', 'PRVU']
        for symbol in valid_symbols:
            self.assertTrue(self.app._validate_symbol(symbol), f"Symbol {symbol} should be valid")
    
    def test_validate_symbol_invalid(self):
        """Test invalid stock symbol validation"""
        # Test invalid symbols
        invalid_symbols = ['', 'TOOLONGSYMBOL123', 'A!B', 'SYMBOL#']
        for symbol in invalid_symbols:
            self.assertFalse(self.app._validate_symbol(symbol), f"Symbol {symbol} should be invalid")
    
    def test_validate_date_valid(self):
        """Test valid date validation"""
        # Test valid dates
        valid_dates = ['2023-01-01', '2023-12-31', '2024-06-15']
        for date in valid_dates:
            self.assertTrue(self.app._validate_date(date), f"Date {date} should be valid")
    
    def test_validate_date_invalid(self):
        """Test invalid date validation"""
        # Test invalid dates
        invalid_dates = ['2023-13-01', '2023-01-32', 'invalid-date', '01-01-2023']
        for date in invalid_dates:
            self.assertFalse(self.app._validate_date(date), f"Date {date} should be invalid")
    
    def test_validate_date_range_valid(self):
        """Test valid date range validation"""
        self.assertTrue(self.app._validate_date_range('2023-01-01', '2023-12-31'))
        self.assertTrue(self.app._validate_date_range('2023-06-01', '2023-06-30'))
    
    def test_validate_date_range_invalid(self):
        """Test invalid date range validation"""
        self.assertFalse(self.app._validate_date_range('2023-12-31', '2023-01-01'))
        self.assertFalse(self.app._validate_date_range('2023-06-30', '2023-06-01'))
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        
        rsi = self.app._calculate_rsi(prices)
        
        # RSI should be a pandas Series
        self.assertIsInstance(rsi, pd.Series)
        # RSI values should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107])
        
        macd_line, signal_line, histogram = self.app._calculate_macd(prices)
        
        # All should be pandas Series
        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108])
        
        upper_band, middle_band, lower_band = self.app._calculate_bollinger_bands(prices)
        
        # All should be pandas Series
        self.assertIsInstance(upper_band, pd.Series)
        self.assertIsInstance(middle_band, pd.Series)
        self.assertIsInstance(lower_band, pd.Series)
        
        # Upper band should be >= middle band >= lower band
        valid_data = ~(upper_band.isna() | middle_band.isna() | lower_band.isna())
        self.assertTrue(all(upper_band[valid_data] >= middle_band[valid_data]))
        self.assertTrue(all(middle_band[valid_data] >= lower_band[valid_data]))
    
    def test_calculate_stochastic(self):
        """Test Stochastic oscillator calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108])
        
        k_percent, d_percent = self.app._calculate_stochastic(prices)
        
        # Both should be pandas Series
        self.assertIsInstance(k_percent, pd.Series)
        self.assertIsInstance(d_percent, pd.Series)
        
        # Values should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_k))
        self.assertTrue(all(0 <= val <= 100 for val in valid_d))
    
    def test_calculate_williams_r(self):
        """Test Williams %R calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108])
        
        wr = self.app._calculate_williams_r(prices)
        
        # Should be a pandas Series
        self.assertIsInstance(wr, pd.Series)
        
        # Values should be between -100 and 0
        valid_wr = wr.dropna()
        self.assertTrue(all(-100 <= val <= 0 for val in valid_wr))
    
    def test_validate_data_quality_good_data(self):
        """Test data quality validation with good data"""
        # Create good quality data
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        })
        
        result = self.app._validate_data_quality(data, 'TEST')
        self.assertTrue(result)
    
    def test_validate_data_quality_empty_data(self):
        """Test data quality validation with empty data"""
        data = pd.DataFrame()
        
        result = self.app._validate_data_quality(data, 'TEST')
        self.assertFalse(result)
    
    def test_validate_data_quality_missing_columns(self):
        """Test data quality validation with missing columns"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103]
        })
        
        result = self.app._validate_data_quality(data, 'TEST')
        self.assertFalse(result)
    
    def test_update_performance_stats(self):
        """Test performance statistics update"""
        initial_fetches = self.app.performance_stats['data_fetches']
        initial_updates = self.app.performance_stats['chart_updates']
        initial_errors = self.app.performance_stats['errors_count']
        
        # Test data fetch update
        self.app._update_performance_stats('data_fetch', 'Test fetch')
        self.assertEqual(self.app.performance_stats['data_fetches'], initial_fetches + 1)
        
        # Test chart update
        self.app._update_performance_stats('chart_update', 'Test update')
        self.assertEqual(self.app.performance_stats['chart_updates'], initial_updates + 1)
        
        # Test error update
        self.app._update_performance_stats('error', 'Test error')
        self.assertEqual(self.app.performance_stats['errors_count'], initial_errors + 1)
    
    def test_get_memory_usage(self):
        """Test memory usage calculation"""
        # Add some test data
        self.app.stock_data['TEST'] = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        
        memory_usage = self.app._get_memory_usage()
        
        # Should return a dictionary
        self.assertIsInstance(memory_usage, dict)
        
        # Should contain expected keys (psutil might not be available)
        expected_keys = ['cache_symbols', 'cache_data_points', 'cache_size_mb']
        for key in expected_keys:
            self.assertIn(key, memory_usage)
        
        # Values should be reasonable
        self.assertEqual(memory_usage['cache_symbols'], 1)
        self.assertEqual(memory_usage['cache_data_points'], 5)
        self.assertGreater(memory_usage['cache_size_mb'], 0)
    
    def test_classify_sector(self):
        """Test sector classification"""
        # Test known sectors
        self.assertEqual(self.app._classify_sector('TECHCORP'), 'Technology')
        self.assertEqual(self.app._classify_sector('BANKFIN'), 'Financial Services')
        self.assertEqual(self.app._classify_sector('INSURANCE'), 'Financial Services')
        self.assertEqual(self.app._classify_sector('OILGAS'), 'Energy')
        self.assertEqual(self.app._classify_sector('NEPSE'), 'Tourism/Hospitality')
        
        # Test unknown symbol
        self.assertEqual(self.app._classify_sector('UNKNOWN'), 'Other')
    
    def test_portfolio_analytics_empty_portfolio(self):
        """Test portfolio analytics with empty portfolio"""
        analytics = self.app._calculate_portfolio_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertEqual(analytics, {})  # Empty portfolio returns empty dict
    
    def test_portfolio_analytics_with_data(self):
        """Test portfolio analytics with portfolio data"""
        # Add test portfolio data
        self.app.portfolio = {
            'TEST1': {
                'shares': 100,
                'buy_price': 100.0,
                'current_price': 110.0,
                'last_updated': datetime.now()
            },
            'TEST2': {
                'shares': 50,
                'buy_price': 200.0,
                'current_price': 180.0,
                'last_updated': datetime.now()
            }
        }
        
        analytics = self.app._calculate_portfolio_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertEqual(analytics['total_value'], 100 * 110.0 + 50 * 180.0)
        self.assertEqual(analytics['total_investment'], 100 * 100.0 + 50 * 200.0)
        self.assertEqual(analytics['total_gain_loss'], 1000.0 - 1000.0)  # Should be 0
        self.assertEqual(analytics['num_stocks'], 2)
        self.assertAlmostEqual(analytics['total_return_pct'], 0.0, places=2)

class TestDataImportExport(unittest.TestCase):
    """Test suite for data import/export functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample CSV data
        self.csv_data = """Symbol,Shares,Buy Price
NEPSE,100,1000.0
NABIL,50,2000.0
EBL,75,1500.0"""
        
        self.csv_file = os.path.join(self.test_dir, 'test_portfolio.csv')
        with open(self.csv_file, 'w') as f:
            f.write(self.csv_data)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_import_portfolio_csv_valid(self):
        """Test importing valid CSV portfolio"""
        # Mock the app instance
        app = Mock()
        app.portfolio = {}
        app.logger = Mock()
        app._validate_symbol = Mock(return_value=True)
        
        # Import the CSV
        from main import NEPSEAnalysisApp
        NEPSEAnalysisApp._import_portfolio_csv(app, self.csv_file)
        
        # Check if data was imported
        self.assertEqual(len(app.portfolio), 3)
        self.assertIn('NEPSE', app.portfolio)
        self.assertIn('NABIL', app.portfolio)
        self.assertIn('EBL', app.portfolio)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestNEPSEAnalysisApp))
    test_suite.addTest(unittest.makeSuite(TestDataImportExport))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
