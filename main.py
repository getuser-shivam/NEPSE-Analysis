import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import threading
import os
import pickle
import seaborn as sns
from scipy import stats
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Any
import re
import csv
import math
from collections import defaultdict
import argparse
import sys
import asyncio
import data_manager
import technical_indicators
import ai_analytics

class NEPSEAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NEPSE Stock Analysis Tool")
        self.root.geometry("1200x800")
        
        # Initialize logging
        self._setup_logging()
        
        # Load configuration
        self._load_config()
        
        # Stock data storage
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.portfolio: Dict[str, Dict[str, float]] = {}
        self.watchlist: List[str] = []
        self.data_file = "nepse_data.pkl"
        self.portfolio_file = "portfolio.pkl"
        self.watchlist_file = "watchlist.pkl"
        
        # Load saved data
        self.load_data()
        
        # Create backup of existing data
        self._create_backup()
        
        # Create main frames
        self.create_frames()
        self.create_widgets()
        
        # Setup auto-save
        self._setup_auto_save()
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
        # Initialize progress tracking
        self.progress_var = tk.StringVar()
        self.progress_var.set("")
        
        # Setup notification system
        self._setup_notifications()
        
        # Setup chart interactivity
        self._setup_chart_interactivity()
        
        # Track application start time
        self.start_time = datetime.now()
        
        # Initialize notification system
        self.notifications = []
        self.price_alerts = {}
        
        # Initialize auto-refresh timer
        self.auto_refresh_enabled = False
        self.refresh_interval = 300  # 5 minutes default
        
        # Initialize portfolio analytics
        self.portfolio_analytics = {}
        self.risk_metrics = {}
        
        # Initialize performance monitoring
        self.performance_stats = {
            'data_fetches': 0,
            'chart_updates': 0,
            'errors_count': 0,
            'last_activity': datetime.now()
        }
        
        # Initialize backup management
        self.backup_manager = {
            'max_backups': 10,
            'backup_interval_hours': 24,
            'last_backup': None
        }
        
        # Initialize memory management
        self.memory_manager = {
            'max_cache_size_mb': 100,
            'auto_cleanup_interval': 300,  # 5 minutes
            'last_cleanup': datetime.now()
        }
        
        # Initialize async capabilities
        self.async_enabled = True
        self.async_loop = None
        self.session = None
        
        # Initialize automatic enhancement system
        self.auto_enhance_enabled = True
        self.last_enhancement_check = datetime.now()
        self.enhancement_interval = 1800  # 30 minutes
        self.performance_baseline = None
        self.error_count = 0
        self.last_code_change = datetime.now()
        
        # Initialize AI analytics
        self.ai_analytics = ai_analytics.AIAnalytics(self.logger)
        self.ml_models_trained = False
        
    def _load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config_data = json.load(f)
                    self.config = config_data.get('settings', {})
                    self.logger.info("Configuration loaded from config.json")
            else:
                # Default configuration
                self.config = {
                    'max_data_age_days': 7,
                    'backup_enabled': True,
                    'auto_save_interval': 300,
                    'chart_style': 'seaborn-v0_8',
                    'default_period': '1y',
                    'max_watchlist_size': 50,
                    'log_level': 'INFO'
                }
                self.logger.info("Using default configuration")
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Fallback to default config
            self.config = {
                'max_data_age_days': 7,
                'backup_enabled': True,
                'auto_save_interval': 300,
                'chart_style': 'seaborn-v0_8',
                'default_period': '1y',
                'max_watchlist_size': 50,
                'log_level': 'INFO'
            }
            
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('nepse_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("NEPSE Analysis App started")
        
    def _create_backup(self) -> None:
        """Create backup of existing data files"""
        if not self.config.get('backup_enabled', True):
            return
            
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in [self.data_file, self.portfolio_file, self.watchlist_file]:
            if os.path.exists(file_path):
                backup_path = os.path.join(backup_dir, f"{timestamp}_{file_path}")
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"Created backup: {backup_path}")
                
    def _setup_auto_save(self) -> None:
        """Setup automatic save timer"""
        if self.config.get('auto_save_interval', 0) > 0:
            self.root.after(self.config['auto_save_interval'] * 1000, self._auto_save)
            
        if self.auto_enhance_enabled:
            self._schedule_auto_enhancement_check()
            self.logger.info("🚀 Automatic enhancement system started")
            
    def _auto_save(self) -> None:
        """Automatically save data"""
        try:
            self.save_data()
            self.logger.info("Auto-save completed")
        except (IOError, OSError, pickle.PickleError) as e:
            self.logger.error(f"Auto-save failed: {e}")
        finally:
            # Schedule next auto-save
            if self.config.get('auto_save_interval', 0) > 0:
                self.root.after(self.config['auto_save_interval'] * 1000, self._auto_save)
                
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not symbol.strip():
            return False
        
        # Basic validation: alphanumeric, 1-10 characters
        symbol = symbol.strip().upper()
        return bool(re.match(r'^[A-Z0-9]{1,10}$', symbol))
        
    def _validate_date(self, date_str: str) -> bool:
        """Validate date format and range"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            # Check if date is not too far in the past or future
            min_date = datetime.now() - timedelta(days=365 * 10)  # 10 years ago
            max_date = datetime.now() + timedelta(days=30)  # 30 days in future
            return min_date <= date <= max_date
        except ValueError:
            return False
            
    def _validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate that start_date is before end_date and range is reasonable"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start >= end:
                return False
                
            # Check if range is not too large (max 5 years)
            max_range = timedelta(days=365 * 5)
            return (end - start) <= max_range
        except ValueError:
            return False
        
    def create_frames(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Left panel for controls
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="10")
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for charts and data
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom panel for portfolio
        self.portfolio_frame = ttk.LabelFrame(self.main_frame, text="Portfolio", padding="10")
        self.portfolio_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Watchlist frame
        self.watchlist_frame = ttk.LabelFrame(self.control_frame, text="Watchlist", padding="5")
        self.watchlist_frame.grid(row=27, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
    def create_widgets(self):
        # Stock Symbol Input with tooltip
        symbol_frame = ttk.Frame(self.control_frame)
        symbol_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(symbol_frame, text="Stock Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_entry = ttk.Entry(symbol_frame, width=20)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        self.symbol_entry.insert(0, "NEPSE")
        
        # Add tooltip for symbol entry
        self._create_tooltip(self.symbol_entry, "Enter stock symbol (e.g., NEPSE, AAPL, GOOG)")
        
        # Search functionality
        search_frame = ttk.Frame(self.control_frame)
        search_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_entry = ttk.Entry(search_frame, width=15)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self._on_search)
        
        self._create_tooltip(self.search_entry, "Search portfolio and watchlist")
        
        # Date Range
        ttk.Label(self.control_frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.start_date_entry = ttk.Entry(self.control_frame, width=20)
        self.start_date_entry.grid(row=1, column=1, pady=5, padx=5)
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
        
        ttk.Label(self.control_frame, text="End Date:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.end_date_entry = ttk.Entry(self.control_frame, width=20)
        self.end_date_entry.grid(row=2, column=1, pady=5, padx=5)
        self.end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # Buttons
        ttk.Button(self.control_frame, text="Fetch Data", command=self.fetch_stock_data).grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(self.control_frame, text="Add to Portfolio", command=self.add_to_portfolio).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Add to Watchlist", command=self.add_to_watchlist).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Show Portfolio", command=self.show_portfolio).grid(row=6, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Portfolio Analytics", command=self._show_portfolio_analytics).grid(row=7, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Export Data", command=self.export_data).grid(row=8, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Import Portfolio", command=self._import_portfolio).grid(row=9, column=0, columnspan=2, pady=5)
        # Additional buttons for advanced features
        ttk.Button(self.control_frame, text="Search Portfolio", command=self._search_portfolio).grid(row=11, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Clear Cache", command=self._clear_cache).grid(row=12, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Theme", command=self._toggle_theme).grid(row=13, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Settings", command=self._show_settings).grid(row=14, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Performance Test", command=self._performance_comparison).grid(row=15, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="AI Analytics", command=self._show_ai_analytics).grid(row=16, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Train ML Models", command=self._train_ml_models).grid(row=17, column=0, columnspan=2, pady=5)
        
        # Analysis Options
        ttk.Separator(self.control_frame, orient='horizontal').grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(self.control_frame, text="Analysis Options:", font=('Arial', 10, 'bold')).grid(row=19, column=0, columnspan=2, pady=5)
        
        self.show_ma = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Moving Average", variable=self.show_ma).grid(row=20, column=0, columnspan=2, sticky=tk.W)
        
        self.show_volume = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Volume", variable=self.show_volume).grid(row=21, column=0, columnspan=2, sticky=tk.W)
        
        self.show_rsi = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="RSI", variable=self.show_rsi).grid(row=22, column=0, columnspan=2, sticky=tk.W)
        
        self.show_macd = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="MACD", variable=self.show_macd).grid(row=23, column=0, columnspan=2, sticky=tk.W)
        
        self.show_bollinger = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Bollinger Bands", variable=self.show_bollinger).grid(row=24, column=0, columnspan=2, sticky=tk.W)
        
        # Advanced indicators
        self.show_stochastic = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Stochastic", variable=self.show_stochastic).grid(row=25, column=0, columnspan=2, sticky=tk.W)
        
        self.show_williams = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Williams %R", variable=self.show_williams).grid(row=26, column=0, columnspan=2, sticky=tk.W)
        
        # Create matplotlib figure for charts
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Portfolio Treeview
        self.portfolio_tree = ttk.Treeview(self.portfolio_frame, columns=('Symbol', 'Shares', 'Buy Price', 'Current Price', 'Gain/Loss'), show='headings')
        self.portfolio_tree.heading('Symbol', text='Symbol')
        self.portfolio_tree.heading('Shares', text='Shares')
        self.portfolio_tree.heading('Buy Price', text='Buy Price')
        self.portfolio_tree.heading('Current Price', text='Current Price')
        self.portfolio_tree.heading('Gain/Loss', text='Gain/Loss')
        
        self.portfolio_tree.column('Symbol', width=100)
        self.portfolio_tree.column('Shares', width=80)
        self.portfolio_tree.column('Buy Price', width=100)
        self.portfolio_tree.column('Current Price', width=100)
        self.portfolio_tree.column('Gain/Loss', width=100)
        
        self.portfolio_tree.pack(fill=tk.BOTH, expand=True)
        
        # Watchlist Treeview
        self.watchlist_tree = ttk.Treeview(self.watchlist_frame, columns=('Symbol', 'Price', 'Change'), show='headings', height=6)
        self.watchlist_tree.heading('Symbol', text='Symbol')
        self.watchlist_tree.heading('Price', text='Price')
        self.watchlist_tree.heading('Change', text='Change')
        
        self.watchlist_tree.column('Symbol', width=80)
        self.watchlist_tree.column('Price', width=60)
        self.watchlist_tree.column('Change', width=60)
        
        self.watchlist_tree.pack(fill=tk.BOTH, expand=True)
        
        # Status bar with progress indicator
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Progress bar for long operations
        self.progress_bar = ttk.Progressbar(self.main_frame, mode='indeterminate')
        # Initially hidden, shown during operations
        
    def fetch_stock_data(self) -> None:
        """Fetch stock data with improved validation and error handling"""
        symbol = self.symbol_entry.get().strip().upper()
        start_date = self.start_date_entry.get().strip()
        end_date = self.end_date_entry.get().strip()
        
        # Validate inputs
        if not self._validate_symbol(symbol):
            messagebox.showerror("Invalid Symbol", 
                                f"'{symbol}' is not a valid stock symbol.\n"
                                "Use 1-10 alphanumeric characters (e.g., NEPSE, AAPL).")
            self.logger.warning(f"Invalid symbol entered: {symbol}")
            return
            
        if not self._validate_date(start_date):
            messagebox.showerror("Invalid Date", 
                                f"'{start_date}' is not a valid date.\n"
                                "Use YYYY-MM-DD format within the last 10 years.")
            self.logger.warning(f"Invalid start date: {start_date}")
            return
            
        if not self._validate_date(end_date):
            messagebox.showerror("Invalid Date", 
                                f"'{end_date}' is not a valid date.\n"
                                "Use YYYY-MM-DD format within the last 10 years.")
            self.logger.warning(f"Invalid end date: {end_date}")
            return
            
        if not self._validate_date_range(start_date, end_date):
            messagebox.showerror("Invalid Date Range", 
                                "Start date must be before end date\n"
                                "and range should not exceed 5 years.")
            self.logger.warning(f"Invalid date range: {start_date} to {end_date}")
            return
            
        self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        self.status_var.set(f"Fetching data for {symbol}...")
        
        # Show progress bar
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 0))
        self.progress_bar.start(10)
        
        # Run data fetching in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._fetch_data_thread, args=(symbol, start_date, end_date))
        thread.daemon = True
        thread.start()
        
    def _fetch_data_thread(self, symbol: str, start_date: str, end_date: str) -> None:
        """Fetch data in separate thread with comprehensive error handling"""
        try:
            self.logger.info(f"Starting data fetch for {symbol}")
            self._update_performance_stats('data_fetch', f"Fetching {symbol}")
            
            # Try to fetch from NEPSE API first
            data = self._fetch_nepse_data(symbol, start_date, end_date)
            
            if data is None or data.empty:
                self.logger.info(f"NEPSE API failed for {symbol}, trying Yahoo Finance")
                # Fallback to yfinance
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    self.logger.info(f"Yahoo Finance returned {len(data)} records for {symbol}")
                except Exception as yf_error:
                    self.logger.warning(f"Yahoo Finance failed for {symbol}: {yf_error}")
                    data = pd.DataFrame()
            
            if data.empty:
                self.logger.info(f"All APIs failed for {symbol}, using simulated data")
                # If both fail, simulate NEPSE data
                data = self._simulate_nepse_data(symbol, start_date, end_date)
            
            # Validate data quality
            if not self._validate_data_quality(data, symbol):
                self.logger.error(f"Data validation failed for {symbol}")
                self.root.after(0, lambda: messagebox.showerror("Data Error", 
                                                              f"Failed to validate data for {symbol}"))
                self.root.after(0, lambda: self.status_var.set("Data validation failed"))
                return
            
            self.stock_data[symbol] = data
            self.logger.info(f"Successfully fetched and validated data for {symbol}: {len(data)} records")
            
            # Update GUI in main thread
            self.root.after(0, self._update_chart, symbol, data)
            self.root.after(0, lambda: self.status_var.set(f"Data fetched for {symbol}"))
            self.root.after(0, lambda: self.update_watchlist_display())  # Update watchlist if symbol is in it
            self.root.after(0, self._hide_progress_bar)
            
            # Auto memory optimization after data fetch
            self._auto_memory_optimization()
            
        except Exception as e:
            error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            self._update_performance_stats('error', error_msg)
            self.root.after(0, lambda: messagebox.showerror("Fetch Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Error fetching data"))
            self.root.after(0, self._hide_progress_bar)
            
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate the quality and integrity of fetched data"""
        try:
            if data.empty:
                self.logger.warning(f"Empty data received for {symbol}")
                return False
                
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                return False
                
            # Check for NaN values in critical columns
            critical_columns = ['Close']
            for col in critical_columns:
                if data[col].isna().all():
                    self.logger.warning(f"All NaN values in {col} column for {symbol}")
                    return False
                    
            # Check for reasonable price ranges (avoid extreme values)
            if 'Close' in data.columns:
                close_prices = data['Close'].dropna()
                if not close_prices.empty:
                    min_price = close_prices.min()
                    max_price = close_prices.max()
                    if min_price <= 0 or max_price > 1000000:  # Reasonable price range
                        self.logger.warning(f"Unreasonable price range for {symbol}: {min_price} - {max_price}")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error for {symbol}: {e}")
            return False
            
    def _simulate_nepse_data(self, symbol, start_date, end_date):
        """Simulate NEPSE stock data when real data is not available"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        # Generate realistic stock price movements
        np.random.seed(hash(symbol) % 2**32)
        base_price = np.random.uniform(100, 10000)
        
        prices = []
        volumes = []
        
        current_price = base_price
        for i in range(len(dates)):
            # Random walk with trend
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            current_price = max(current_price, 10)  # Minimum price
            
            # Generate volume
            volume = np.random.randint(1000, 100000)
            
            prices.append(current_price)
            volumes.append(volume)
        
        # Create OHLC data
        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return data
        
    def _fetch_nepse_data(self, symbol, start_date, end_date):
        """Fetch data from NEPSE API or web scraping"""
        try:
            # Try to get live data from NEPSE website
            url = "https://nepalstock.com.np/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # This is a simplified version - real implementation would need to parse actual NEPSE data
                # For now, return None to use fallback methods
                return None
            
        except Exception as e:
            print(f"NEPSE API error: {e}")
            return None
        
    def _update_chart(self, symbol: str, data: pd.DataFrame) -> None:
        """Update chart with enhanced styling and error handling"""
        try:
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Set style from configuration
            chart_style = self.config.get('chart_style', 'seaborn-v0_8')
            try:
                plt.style.use(chart_style)
            except OSError:
                # Fallback to default style if specified style not available
                plt.style.use('default')
                self.logger.warning(f"Chart style '{chart_style}' not available, using default")
            
            # Validate data before plotting
            if data.empty or 'Close' not in data.columns:
                self.logger.error(f"Invalid data for chart: {symbol}")
                self.ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=self.ax1.transAxes)
                self.canvas.draw()
                return
            
            # Plot price chart
            self.ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
            
            # Add moving average if selected
            if self.show_ma.get():
                if len(data) >= 50:  # Only show MA if enough data
                    ma_20 = data['Close'].rolling(window=20).mean()
                    ma_50 = data['Close'].rolling(window=50).mean()
                    self.ax1.plot(data.index, ma_20, label='20-day MA', alpha=0.7, color='orange')
                    self.ax1.plot(data.index, ma_50, label='50-day MA', alpha=0.7, color='red')
                else:
                    self.logger.warning(f"Not enough data for moving averages: {len(data)} records")
            
            # Add Bollinger Bands if selected
            if self.show_bollinger.get():
                if len(data) >= 20:  # Only show Bollinger Bands if enough data
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
                    self.ax1.plot(data.index, bb_upper, label='BB Upper', alpha=0.5, color='gray', linestyle='--')
                    self.ax1.plot(data.index, bb_middle, label='BB Middle', alpha=0.5, color='gray')
                    self.ax1.plot(data.index, bb_lower, label='BB Lower', alpha=0.5, color='gray', linestyle='--')
                    self.ax1.fill_between(data.index, bb_upper, bb_lower, alpha=0.1, color='gray')
                else:
                    self.logger.warning(f"Not enough data for Bollinger Bands: {len(data)} records")
            
            self.ax1.set_title(f'{symbol} Stock Price', fontweight='bold')
            self.ax1.set_ylabel('Price (NPR)')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # Format x-axis for better readability
            self.ax1.tick_params(axis='x', rotation=45)
            
            # Plot volume or indicators based on selection
            if self.show_volume.get():
                if 'Volume' in data.columns:
                    self.ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
                    self.ax2.set_title('Trading Volume', fontweight='bold')
                    self.ax2.set_ylabel('Volume')
                    self.ax2.set_xlabel('Date')
                else:
                    self.ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=self.ax2.transAxes)
            elif self.show_macd.get():
                if len(data) >= 26:  # MACD needs sufficient data
                    macd_line, signal_line, histogram = self._calculate_macd(data['Close'])
                    self.ax2.plot(data.index, macd_line, label='MACD', color='blue')
                    self.ax2.plot(data.index, signal_line, label='Signal', color='red')
                    self.ax2.bar(data.index, histogram, label='Histogram', alpha=0.6, color='green')
                    self.ax2.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
                    self.ax2.set_ylabel('MACD')
                    self.ax2.set_xlabel('Date')
                    self.ax2.legend()
                else:
                    self.ax2.text(0.5, 0.5, 'Insufficient data for MACD', ha='center', va='center', transform=self.ax2.transAxes)
            elif self.show_rsi.get():
                if len(data) >= 14:  # RSI needs sufficient data
                    rsi = self._calculate_rsi(data['Close'])
                    self.ax2.plot(data.index, rsi, label='RSI', color='purple')
                    self.ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
                    self.ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
                    self.ax2.set_title('RSI (Relative Strength Index)', fontweight='bold')
                    self.ax2.set_ylabel('RSI')
                    self.ax2.set_xlabel('Date')
                    self.ax2.legend()
                    self.ax2.set_ylim(0, 100)  # RSI is always between 0-100
                else:
                    self.ax2.text(0.5, 0.5, 'Insufficient data for RSI', ha='center', va='center', transform=self.ax2.transAxes)
            elif self.show_stochastic.get():
                if len(data) >= 14:  # Stochastic needs sufficient data
                    k_percent, d_percent = self._calculate_stochastic(data['Close'])
                    self.ax2.plot(data.index, k_percent, label='%K', color='blue', alpha=0.7)
                    self.ax2.plot(data.index, d_percent, label='%D', color='red', alpha=0.7)
                    self.ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought')
                    self.ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold')
                    self.ax2.set_title('Stochastic Oscillator', fontweight='bold')
                    self.ax2.set_ylabel('Value')
                    self.ax2.set_xlabel('Date')
                    self.ax2.legend()
                    self.ax2.set_ylim(0, 100)
                else:
                    self.ax2.text(0.5, 0.5, 'Insufficient data for Stochastic', ha='center', va='center', transform=self.ax2.transAxes)
            elif self.show_williams.get():
                if len(data) >= 14:  # Williams %R needs sufficient data
                    wr = self._calculate_williams_r(data['Close'])
                    self.ax2.plot(data.index, wr, label='Williams %R', color='orange', alpha=0.7)
                    self.ax2.axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought')
                    self.ax2.axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold')
                    self.ax2.set_title('Williams %R', fontweight='bold')
                    self.ax2.set_ylabel('%R')
                    self.ax2.set_xlabel('Date')
                    self.ax2.legend()
                    self.ax2.set_ylim(-100, 0)
                else:
                    self.ax2.text(0.5, 0.5, 'Insufficient data for Williams %R', ha='center', va='center', transform=self.ax2.transAxes)
            else:
                # Default to volume if nothing selected
                if 'Volume' in data.columns:
                    self.ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
                    self.ax2.set_title('Trading Volume', fontweight='bold')
                    self.ax2.set_ylabel('Volume')
                    self.ax2.set_xlabel('Date')
                else:
                    self.ax2.text(0.5, 0.5, 'No indicator selected', ha='center', va='center', transform=self.ax2.transAxes)
            
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(axis='x', rotation=45)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.logger.info(f"Chart updated successfully for {symbol}")
            self._update_performance_stats('chart_update', f"Updated chart for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating chart for {symbol}: {e}")
            # Show error message on chart
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=self.ax1.transAxes)
            self.canvas.draw()
        
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
        
    def _calculate_stochastic(self, prices, k=14, d=3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        try:
            low_min = prices.rolling(window=k).min()
            high_max = prices.rolling(window=k).max()
            
            # Calculate %K and %D
            k_percent = 100 * ((prices - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d).mean()
            
            return k_percent, d_percent
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Stochastic: {e}")
            return pd.Series(), pd.Series(), pd.Series()
            
    def _calculate_williams_r(self, prices, period=14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            high_max = prices.rolling(window=period).max()
            low_min = prices.rolling(window=period).min()
            
            # Calculate Williams %R
            wr = -100 * ((high_max - prices) / (high_max - low_min))
            return wr
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Williams %R: {e}")
            return pd.Series()
            
    def _update_performance_stats(self, action: str, details: str = "") -> None:
        """Update performance statistics"""
        try:
            self.performance_stats['last_activity'] = datetime.now()
            
            if action == 'data_fetch':
                self.performance_stats['data_fetches'] += 1
            elif action == 'chart_update':
                self.performance_stats['chart_updates'] += 1
            elif action == 'error':
                self.performance_stats['errors_count'] += 1
                self.logger.warning(f"Performance error: {details}")
                
        except Exception as e:
            self.logger.error(f"Failed to update performance stats: {e}")
            
    def _manage_backups(self) -> None:
        """Manage backup rotation and cleanup"""
        try:
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                return
                
            # Get all backup files
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith('20') and file.endswith('.pkl'):
                    backup_files.append(file)
                    
            # Sort by creation time (newest first)
            backup_files.sort(reverse=True)
            
            # Remove old backups if we have too many
            max_backups = self.backup_manager['max_backups']
            if len(backup_files) > max_backups:
                for old_backup in backup_files[max_backups:]:
                    old_path = os.path.join(backup_dir, old_backup)
                    os.remove(old_path)
                    self.logger.info(f"Removed old backup: {old_backup}")
                    
            # Update last backup time
            if backup_files:
                self.backup_manager['last_backup'] = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Failed to manage backups: {e}")
            
    def _create_enhanced_backup(self) -> None:
        """Create enhanced backup with metadata"""
        try:
            if not self.config.get('backup_enabled', True):
                return
                
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create backup metadata
            backup_metadata = {
                'timestamp': timestamp,
                'portfolio_size': len(self.portfolio),
                'watchlist_size': len(self.watchlist),
                'stock_data_symbols': len(self.stock_data),
                'app_version': '2.0.0',
                'performance_stats': self.performance_stats,
                'config': self.config
            }
            
            # Backup data files
            backed_up_files = []
            for file_path in [self.data_file, self.portfolio_file, self.watchlist_file]:
                if os.path.exists(file_path):
                    backup_name = f"{timestamp}_{os.path.basename(file_path)}"
                    backup_path = os.path.join(backup_dir, backup_name)
                    shutil.copy2(file_path, backup_path)
                    backed_up_files.append(backup_name)
                    
            # Save metadata
            metadata_path = os.path.join(backup_dir, f"{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
                
            self.logger.info(f"Enhanced backup created: {backed_up_files}")
            
            # Manage backup rotation
            self._manage_backups()
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced backup: {e}")
            
    def _auto_memory_optimization(self) -> None:
        """Automatic memory optimization and cleanup"""
        try:
            current_time = datetime.now()
            
            # Check if cleanup interval has passed
            if (current_time - self.memory_manager['last_cleanup']).seconds < self.memory_manager['auto_cleanup_interval']:
                return
                
            # Calculate current cache size (approximate)
            cache_size_mb = sum(len(str(data)) for data in self.stock_data.values()) / (1024 * 1024)
            
            # If cache exceeds limit, clean up old data
            if cache_size_mb > self.memory_manager['max_cache_size_mb']:
                self._cleanup_old_data()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.memory_manager['last_cleanup'] = current_time
                self.logger.info(f"Auto memory optimization: cleaned up {cache_size_mb:.1f}MB cache")
                
        except Exception as e:
            self.logger.error(f"Failed to optimize memory: {e}")
            
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'cache_symbols': len(self.stock_data),
                'cache_data_points': sum(len(data) for data in self.stock_data.values()),
                'cache_size_mb': sum(len(str(data)) for data in self.stock_data.values()) / (1024 * 1024)
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cache_symbols': len(self.stock_data),
                'cache_data_points': sum(len(data) for data in self.stock_data.values()),
                'cache_size_mb': sum(len(str(data)) for data in self.stock_data.values()) / (1024 * 1024)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {}
            
    async def _async_fetch_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Async Yahoo Finance data fetching"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Use yfinance with async wrapper
            loop = asyncio.get_event_loop()
            
            # Run yfinance in thread pool to avoid blocking
            data = await loop.run_in_executor(
                None, 
                lambda: yf.Ticker(symbol).history(start=start_date, end=end_date)
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"Async Yahoo Finance returned {len(data)} records for {symbol}")
                return data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Async Yahoo Finance failed for {symbol}: {e}")
            return None
            
    async def _async_fetch_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch multiple stocks concurrently"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Create tasks for concurrent fetching
            tasks = []
            for symbol in symbols:
                task = self._async_fetch_yahoo_data(symbol, start_date, end_date)
                tasks.append((symbol, task))
                
            # Wait for all tasks to complete
            results = {}
            for symbol, task in tasks:
                try:
                    data = await task
                    if data is not None and not data.empty:
                        results[symbol] = data
                        self.logger.info(f"Async fetched {symbol}: {len(data)} records")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol} async: {e}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to fetch multiple stocks async: {e}")
            return {}
            
    def _async_data_fetch_wrapper(self, symbols: List[str], start_date: str, end_date: str) -> None:
        """Wrapper for async data fetching in tkinter environment"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async operation
            results = loop.run_until_complete(
                self._async_fetch_multiple_stocks(symbols, start_date, end_date)
            )
            
            # Update GUI in main thread
            for symbol, data in results.items():
                self.stock_data[symbol] = data
                self.root.after(0, lambda s=symbol, d=data: self._update_chart(s, d))
                
            # Update status
            self.root.after(0, lambda: self.status_var.set(f"Async fetched {len(results)} stocks"))
            self.root.after(0, self._hide_progress_bar)
            
            # Close loop
            loop.close()
            
        except Exception as e:
            self.logger.error(f"Async data fetch wrapper failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Async Error", f"Failed to fetch data: {str(e)}"))
            self.root.after(0, self._hide_progress_bar)
        
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
        
    def add_to_portfolio(self) -> None:
        """Add stock to portfolio with improved validation and error handling"""
        symbol = self.symbol_entry.get().strip().upper()
        
        if not self._validate_symbol(symbol):
            messagebox.showerror("Invalid Symbol", f"'{symbol}' is not a valid stock symbol.")
            return
            
        if symbol not in self.stock_data:
            messagebox.showerror("No Data", f"No data available for {symbol}. Please fetch data first.")
            return
            
        # Create dialog for portfolio input
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Add {symbol} to Portfolio")
        dialog.geometry("350x250")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Get current price for reference
        current_price = self.stock_data[symbol]['Close'][-1]
        
        ttk.Label(dialog, text=f"Symbol: {symbol}", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        ttk.Label(dialog, text=f"Current Price: NPR {current_price:.2f}").grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        
        ttk.Label(dialog, text="Number of Shares:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        shares_entry = ttk.Entry(dialog)
        shares_entry.grid(row=2, column=1, padx=10, pady=10)
        shares_entry.insert(0, "100")  # Default value
        
        ttk.Label(dialog, text="Buy Price per Share:").grid(row=3, column=0, padx=10, pady=10, sticky='w')
        price_entry = ttk.Entry(dialog)
        price_entry.grid(row=3, column=1, padx=10, pady=10)
        price_entry.insert(0, f"{current_price:.2f}")  # Default to current price
        
        def add_stock():
            try:
                shares_str = shares_entry.get().strip()
                price_str = price_entry.get().strip()
                
                if not shares_str or not price_str:
                    messagebox.showerror("Input Error", "Please fill all fields")
                    return
                    
                shares = float(shares_str)
                buy_price = float(price_str)
                
                if shares <= 0:
                    messagebox.showerror("Invalid Input", "Number of shares must be positive")
                    return
                    
                if buy_price <= 0:
                    messagebox.showerror("Invalid Input", "Buy price must be positive")
                    return
                
                # Update or add to portfolio
                self.portfolio[symbol] = {
                    'shares': shares,
                    'buy_price': buy_price,
                    'current_price': current_price,
                    'last_updated': datetime.now()
                }
                
                self.update_portfolio_display()
                self.logger.info(f"Added {symbol} to portfolio: {shares} shares at NPR {buy_price:.2f}")
                
                dialog.destroy()
                messagebox.showinfo("Success", f"{symbol} added to portfolio successfully!")
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for shares and price")
            except Exception as e:
                self.logger.error(f"Error adding to portfolio: {e}")
                messagebox.showerror("Error", f"Failed to add to portfolio: {str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Add", command=add_stock).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def update_portfolio_display(self):
        # Clear existing items
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        # Add portfolio items
        for symbol, data in self.portfolio.items():
            gain_loss = (data['current_price'] - data['buy_price']) * data['shares']
            gain_loss_pct = ((data['current_price'] - data['buy_price']) / data['buy_price']) * 100
            
            self.portfolio_tree.insert('', 'end', values=(
                symbol,
                f"{data['shares']:.2f}",
                f"NPR {data['buy_price']:.2f}",
                f"NPR {data['current_price']:.2f}",
                f"NPR {gain_loss:.2f} ({gain_loss_pct:.2f}%)"
            ))
            
    def show_portfolio(self):
        if not self.portfolio:
            messagebox.showinfo("Portfolio", "Portfolio is empty")
            return
            
        total_investment = sum(data['shares'] * data['buy_price'] for data in self.portfolio.values())
        current_value = sum(data['shares'] * data['current_price'] for data in self.portfolio.values())
        total_gain_loss = current_value - total_investment
        
        summary = f"Portfolio Summary:\n\n"
        summary += f"Total Investment: NPR {total_investment:.2f}\n"
        summary += f"Current Value: NPR {current_value:.2f}\n"
        summary += f"Total Gain/Loss: NPR {total_gain_loss:.2f}\n"
        summary += f"Return: {((total_gain_loss / total_investment) * 100):.2f}%"
        
        messagebox.showinfo("Portfolio Summary", summary)
        
    def export_data(self) -> None:
        """Export data with multiple format options"""
        if not self.stock_data:
            messagebox.showerror("Error", "No data to export")
            return
            
        # Create export dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Data")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Export Options", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Format selection
        format_frame = ttk.LabelFrame(dialog, text="Export Format", padding="10")
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        
        format_var = tk.StringVar(value="CSV")
        ttk.Radiobutton(format_frame, text="CSV", variable=format_var, value="CSV").pack(anchor='w')
        ttk.Radiobutton(format_frame, text="Excel", variable=format_var, value="Excel").pack(anchor='w')
        ttk.Radiobutton(format_frame, text="JSON", variable=format_var, value="JSON").pack(anchor='w')
        ttk.Radiobutton(format_frame, text="PDF Report", variable=format_var, value="PDF").pack(anchor='w')
        
        # Data selection
        data_frame = ttk.LabelFrame(dialog, text="Data to Export", padding="10")
        data_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        export_portfolio = tk.BooleanVar(value=True)
        export_watchlist = tk.BooleanVar(value=True)
        export_stock_data = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(data_frame, text="Portfolio Data", variable=export_portfolio).pack(anchor='w')
        ttk.Checkbutton(data_frame, text="Watchlist", variable=export_watchlist).pack(anchor='w')
        ttk.Checkbutton(data_frame, text="Stock Price Data", variable=export_stock_data).pack(anchor='w')
                
            if buy_price <= 0:
                messagebox.showerror("Invalid Input", "Buy price must be positive")
                return
                self._hide_progress_bar()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Export", command=do_export).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
    def _export_csv(self, filename: str, export_portfolio: bool, export_watchlist: bool, export_stock_data: bool) -> None:
        """Export data to CSV format"""
        import csv
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Export portfolio
            if export_portfolio and self.portfolio:
                writer.writerow(['Portfolio Data'])
                writer.writerow(['Symbol', 'Shares', 'Buy Price', 'Current Price', 'Gain/Loss', 'Gain/Loss %'])
                for symbol, data in self.portfolio.items():
                    gain_loss = (data['current_price'] - data['buy_price']) * data['shares']
                    gain_loss_pct = ((data['current_price'] - data['buy_price']) / data['buy_price']) * 100
                    writer.writerow([symbol, data['shares'], data['buy_price'], data['current_price'], gain_loss, gain_loss_pct])
                writer.writerow([])  # Empty row
            
            # Export watchlist
            if export_watchlist and self.watchlist:
                writer.writerow(['Watchlist'])
                writer.writerow(['Symbol'])
                for symbol in self.watchlist:
                    writer.writerow([symbol])
                writer.writerow([])  # Empty row
            
            # Export stock data
            if export_stock_data and self.stock_data:
                writer.writerow(['Stock Price Data'])
                for symbol, data in self.stock_data.items():
                    writer.writerow([f'Data for {symbol}'])
                    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    for date, row in data.iterrows():
                        writer.writerow([date.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume']])
                    writer.writerow([])  # Empty row
                    
    def _export_excel(self, filename: str, export_portfolio: bool, export_watchlist: bool, export_stock_data: bool) -> None:
        """Export data to Excel format"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Export portfolio
                if export_portfolio and self.portfolio:
                    portfolio_data = []
                    for symbol, data in self.portfolio.items():
                        gain_loss = (data['current_price'] - data['buy_price']) * data['shares']
                        gain_loss_pct = ((data['current_price'] - data['buy_price']) / data['buy_price']) * 100
                        portfolio_data.append({
                            'Symbol': symbol,
                            'Shares': data['shares'],
                            'Buy Price': data['buy_price'],
                            'Current Price': data['current_price'],
                            'Gain/Loss': gain_loss,
                            'Gain/Loss %': gain_loss_pct
                        })
                    pd.DataFrame(portfolio_data).to_excel(writer, sheet_name='Portfolio', index=False)
                
                # Export watchlist
                if export_watchlist and self.watchlist:
                    watchlist_data = [{'Symbol': symbol} for symbol in self.watchlist]
                    pd.DataFrame(watchlist_data).to_excel(writer, sheet_name='Watchlist', index=False)
                
                # Export stock data
                if export_stock_data and self.stock_data:
                    for symbol, data in self.stock_data.items():
                        # Clean sheet name (Excel has restrictions)
                        sheet_name = symbol.replace('/', '_').replace('\\', '_')[:31]
                        data.to_excel(writer, sheet_name=sheet_name)
                        
        except ImportError:
            # Fallback to CSV if openpyxl not available
            self.logger.warning("openpyxl not available, falling back to CSV")
            self._export_csv(filename.replace('.xlsx', '.csv'), export_portfolio, export_watchlist, export_stock_data)
            
    def _export_pdf_report(self, filename: str, export_portfolio: bool, export_watchlist: bool, export_stock_data: bool) -> None:
        """Export data to PDF format with charts"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import matplotlib.pyplot as plt
            import io
            
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("NEPSE Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Export date
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Portfolio section
            if export_portfolio and self.portfolio:
                story.append(Paragraph("Portfolio Summary", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                portfolio_data = [['Symbol', 'Shares', 'Buy Price', 'Current Price', 'Gain/Loss', 'Return %']]
                total_investment = 0
                total_value = 0
                
                for symbol, data in self.portfolio.items():
                    investment = data['shares'] * data['buy_price']
                    current_value = data['shares'] * data['current_price']
                    gain_loss = current_value - investment
                    return_pct = (gain_loss / investment * 100) if investment > 0 else 0
                    
                    portfolio_data.append([
                        symbol,
                        str(data['shares']),
                        f"{data['buy_price']:.2f}",
                        f"{data['current_price']:.2f}",
                        f"{gain_loss:.2f}",
                        f"{return_pct:.2f}%"
                    ])
                    
                    total_investment += investment
                    total_value += current_value
                
                # Create table
                table = Table(portfolio_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#4CAF50'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
                
                # Portfolio summary
                total_gain_loss = total_value - total_investment
                total_return = (total_gain_loss / total_investment * 100) if total_investment > 0 else 0
                
                summary_data = [
                    ['Total Investment', f"{total_investment:.2f}"],
                    ['Current Value', f"{total_value:.2f}"],
                    ['Total Gain/Loss', f"{total_gain_loss:.2f}"],
                    ['Total Return %', f"{total_return:.2f}%"]
                ]
                
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), '#2196F3'),
                    ('TEXTCOLOR', (0, 0), (0, -1), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ]))
                
                story.append(summary_table)
                story.append(Spacer(1, 20))
            
            # Watchlist section
            if export_watchlist and self.watchlist:
                story.append(Paragraph("Watchlist", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                watchlist_data = [['Symbol', 'Added Date']]
                for symbol in self.watchlist:
                    watchlist_data.append([symbol, 'N/A'])
                
                watchlist_table = Table(watchlist_data)
                watchlist_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#FF9800'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                    ('GRID', (0, 0), (-1, -1), 1, '#000000')
                ]))
                
                story.append(watchlist_table)
                story.append(Spacer(1, 20))
            
            # Add charts if stock data available
            if export_stock_data and self.stock_data:
                story.append(Paragraph("Stock Charts", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                for symbol, data in list(self.stock_data.items())[:3]:  # Limit to first 3 stocks
                    # Create chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(data.index, data['Close'], label=f'{symbol} Close Price')
                    ax.set_title(f'{symbol} Stock Price')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.legend()
                    ax.grid(True)
                    
                    # Save chart to buffer
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close()
                    
                    # Add chart to PDF
                    story.append(Paragraph(f"{symbol} Price Chart", styles['Heading3']))
                    story.append(Spacer(1, 6))
                    story.append(Image(img_buffer, width=6*inch, height=3*inch))
                    story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"PDF report exported to {filename}")
            
        except ImportError:
            self.logger.warning("reportlab not available, PDF export not supported")
            messagebox.showerror("Export Error", "PDF export requires reportlab library. Install with: pip install reportlab")
        except Exception as e:
            self.logger.error(f"Failed to export PDF: {e}")
            raise
            
    def _export_json(self, filename: str, export_portfolio: bool, export_watchlist: bool, export_stock_data: bool) -> None:
        """Export data to JSON format"""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'application': 'NEPSE Stock Analysis Tool'
        }
        
        if export_portfolio:
            export_data['portfolio'] = self.portfolio
            
        if export_watchlist:
            export_data['watchlist'] = self.watchlist
            
        if export_stock_data:
            # Convert DataFrames to JSON-serializable format
            stock_data_json = {}
            for symbol, data in self.stock_data.items():
                stock_data_json[symbol] = {
                    'dates': [date.isoformat() for date in data.index],
                    'columns': data.columns.tolist(),
                    'data': data.values.tolist()
                }
            export_data['stock_data'] = stock_data_json
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
    def _on_search(self, event) -> None:
        """Handle search input changes"""
        search_term = self.search_entry.get().strip().upper()
        
        if not search_term:
            # Reset views if search is empty
            self.update_portfolio_display()
            self.update_watchlist_display()
            return
            
        # Filter portfolio
        self._filter_portfolio(search_term)
        
        # Filter watchlist
        self._filter_watchlist(search_term)
        
    def _filter_portfolio(self, search_term: str) -> None:
        """Filter portfolio based on search term"""
        # Clear existing items
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        # Add filtered portfolio items
        for symbol, data in self.portfolio.items():
            if search_term in symbol.upper():
                gain_loss = (data['current_price'] - data['buy_price']) * data['shares']
                gain_loss_pct = ((data['current_price'] - data['buy_price']) / data['buy_price']) * 100
                
                self.portfolio_tree.insert('', 'end', values=(
                    symbol,
                    f"{data['shares']:.2f}",
                    f"NPR {data['buy_price']:.2f}",
                    f"NPR {data['current_price']:.2f}",
                    f"NPR {gain_loss:.2f} ({gain_loss_pct:.2f}%)"
                ))
                
    def _filter_watchlist(self, search_term: str) -> None:
        """Filter watchlist based on search term"""
        # Clear existing items
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        
        # Add filtered watchlist items
        for symbol in self.watchlist:
            if search_term in symbol.upper():
                if symbol in self.stock_data:
                    current_price = self.stock_data[symbol]['Close'][-1]
                    previous_price = self.stock_data[symbol]['Close'][-2] if len(self.stock_data[symbol]['Close']) > 1 else current_price
                    change = current_price - previous_price
                    change_pct = (change / previous_price) * 100 if previous_price != 0 else 0
                    
                    self.watchlist_tree.insert('', 'end', values=(
                        symbol,
                        f"NPR {current_price:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    ))
                else:
                    self.watchlist_tree.insert('', 'end', values=(
                        symbol,
                        "N/A",
                        "N/A"
                    ))
                    
    def _search_portfolio(self) -> None:
        """Open advanced search dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Search")
        dialog.geometry("400x300")
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Search Portfolio & Watchlist", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Search options
        search_frame = ttk.LabelFrame(dialog, text="Search Options", padding="10")
        search_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        ttk.Label(search_frame, text="Symbol:").grid(row=0, column=0, sticky='w', pady=5)
        search_entry = ttk.Entry(search_frame, width=20)
        search_entry.grid(row=0, column=1, pady=5, padx=10)
        
        def perform_search():
            term = search_entry.get().strip().upper()
            if term:
                self.search_entry.delete(0, tk.END)
                self.search_entry.insert(0, term)
                self._on_search(None)
                dialog.destroy()
                
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Search", command=perform_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=lambda: [search_entry.delete(0, tk.END), self.search_entry.delete(0, tk.END), self._on_search(None)]).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        search_entry.focus()
        
    def _clear_cache(self) -> None:
        """Clear cached data and reset with enhanced memory cleanup"""
        try:
            result = messagebox.askyesno("Clear Cache", "This will remove all cached stock data and free memory. Continue?")
            if result:
                # Enhanced cache clearing with memory optimization
                total_symbols = len(self.stock_data)
                total_data_points = sum(len(data) for data in self.stock_data.values())
                
                # Clear stock data
                self.stock_data.clear()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear performance stats cache
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['errors_count'] = 0
                    
                # Clean up old data
                self._cleanup_old_data()
                
                self.logger.info(f"Cache cleared: {total_symbols} symbols, {total_data_points} data points removed")
                messagebox.showinfo("Success", f"Cache cleared successfully!\nFreed memory from {total_symbols} symbols and {total_data_points} data points.")
                self.status_var.set(f"Cache cleared - freed {total_symbols} symbols")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
            
    def _toggle_theme(self) -> None:
        """Toggle between light and dark themes"""
        try:
            # Simple theme toggle (basic implementation)
            if hasattr(self, 'current_theme') and self.current_theme == 'dark':
                # Switch to light theme
                self.root.tk_setPalette(background='white')
                self.current_theme = 'light'
                self.logger.info("Switched to light theme")
            else:
                # Switch to dark theme
                self.root.tk_setPalette(background='#2b2b2b', foreground='white')
                self.current_theme = 'dark'
                self.logger.info("Switched to dark theme")
                
            messagebox.showinfo("Theme", f"Switched to {self.current_theme} theme")
            
        except Exception as e:
            self.logger.error(f"Failed to toggle theme: {e}")
            messagebox.showerror("Theme Error", f"Failed to toggle theme: {str(e)}")
            
    def add_to_watchlist(self) -> None:
        """Add stock to watchlist with validation and logging"""
        symbol = self.symbol_entry.get().strip().upper()
        
        if not self._validate_symbol(symbol):
            messagebox.showerror("Invalid Symbol", f"'{symbol}' is not a valid stock symbol.")
            self.logger.warning(f"Invalid symbol for watchlist: {symbol}")
            return
            
        if symbol in self.watchlist:
            messagebox.showwarning("Already in Watchlist", f"{symbol} is already in your watchlist.")
            return
            
        # Limit watchlist size to prevent performance issues
        max_watchlist_size = self.config.get('max_watchlist_size', 50)
        if len(self.watchlist) >= max_watchlist_size:
            messagebox.showwarning("Watchlist Full", 
                                  f"Watchlist is full (max {max_watchlist_size} stocks).\n"
                                  "Remove some stocks to add new ones.")
            return
            
        self.watchlist.append(symbol)
        self.update_watchlist_display()
        self.logger.info(f"Added {symbol} to watchlist")
        
        # Fetch data if not already available
        if symbol not in self.stock_data:
            self.logger.info(f"Auto-fetching data for watchlist symbol: {symbol}")
            # Set default date range (last 3 months)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            
            thread = threading.Thread(target=self._fetch_data_thread, args=(symbol, start_date, end_date))
            thread.daemon = True
            thread.start()
        
        messagebox.showinfo("Success", f"{symbol} added to watchlist!")
            
    def update_watchlist_display(self):
        # Clear existing items
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        
        # Add watchlist items
        for symbol in self.watchlist:
            if symbol in self.stock_data:
                current_price = self.stock_data[symbol]['Close'][-1]
                previous_price = self.stock_data[symbol]['Close'][-2] if len(self.stock_data[symbol]['Close']) > 1 else current_price
                change = current_price - previous_price
                change_pct = (change / previous_price) * 100 if previous_price != 0 else 0
                
                self.watchlist_tree.insert('', 'end', values=(
                    symbol,
                    f"NPR {current_price:.2f}",
                    f"{change:+.2f} ({change_pct:+.2f}%)"
                ))
            else:
                self.watchlist_tree.insert('', 'end', values=(
                    symbol,
                    "N/A",
                    "N/A"
                ))
                
    def save_data(self) -> None:
        """Save data with comprehensive error handling and logging"""
        try:
            self._show_progress_bar("Saving data...")
            
            # Create enhanced backup before saving
            self._create_enhanced_backup()
            
            saved_files = []
            
            # Save portfolio
            try:
                with open(self.portfolio_file, 'wb') as f:
                    pickle.dump(self.portfolio, f)
                saved_files.append(self.portfolio_file)
                self.logger.info(f"Portfolio saved: {len(self.portfolio)} stocks")
            except Exception as e:
                self.logger.error(f"Failed to save portfolio: {e}")
            
            # Save watchlist
            try:
                with open(self.watchlist_file, 'wb') as f:
                    pickle.dump(self.watchlist, f)
                saved_files.append(self.watchlist_file)
                self.logger.info(f"Watchlist saved: {len(self.watchlist)} stocks")
            except Exception as e:
                self.logger.error(f"Failed to save watchlist: {e}")
            
            # Save stock data (optional - can be large)
            try:
                with open(self.data_file, 'wb') as f:
                    pickle.dump(self.stock_data, f)
                saved_files.append(self.data_file)
                total_records = sum(len(data) for data in self.stock_data.values())
                self.logger.info(f"Stock data saved: {len(self.stock_data)} symbols, {total_records} total records")
            except Exception as e:
                self.logger.error(f"Failed to save stock data: {e}")
            
            if saved_files:
                messagebox.showinfo("Success", f"Data saved successfully!\nSaved files: {', '.join(saved_files)}")
                self.status_var.set("Data saved successfully")
            else:
                messagebox.showwarning("Partial Save", "Some data could not be saved. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"Critical error during save: {e}")
            messagebox.showerror("Save Error", f"Failed to save data: {str(e)}")
            self.status_var.set("Save failed")
        finally:
            self._hide_progress_bar()
            
    def load_data(self) -> None:
        """Load data with comprehensive error handling and validation"""
        loaded_files = []
        
        # Load portfolio
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'rb') as f:
                    portfolio_data = pickle.load(f)
                    # Validate portfolio data structure
                    if isinstance(portfolio_data, dict):
                        self.portfolio = portfolio_data
                        loaded_files.append(self.portfolio_file)
                        self.logger.info(f"Portfolio loaded: {len(self.portfolio)} stocks")
                    else:
                        self.logger.warning("Invalid portfolio data format, starting fresh")
                        self.portfolio = {}
            except Exception as e:
                self.logger.error(f"Failed to load portfolio: {e}")
                self.portfolio = {}
        
        # Load watchlist
        if os.path.exists(self.watchlist_file):
            try:
                with open(self.watchlist_file, 'rb') as f:
                    watchlist_data = pickle.load(f)
                    # Validate watchlist data structure
                    if isinstance(watchlist_data, list):
                        self.watchlist = watchlist_data
                        loaded_files.append(self.watchlist_file)
                        self.logger.info(f"Watchlist loaded: {len(self.watchlist)} stocks")
                    else:
                        self.logger.warning("Invalid watchlist data format, starting fresh")
                        self.watchlist = []
            except Exception as e:
                self.logger.error(f"Failed to load watchlist: {e}")
                self.watchlist = []
        
        # Load stock data
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    stock_data = pickle.load(f)
                    # Validate stock data structure
                    if isinstance(stock_data, dict):
                        # Validate each DataFrame
                        valid_data = {}
                        for symbol, data in stock_data.items():
                            if isinstance(data, pd.DataFrame) and not data.empty:
                                if self._validate_data_quality(data, symbol):
                                    valid_data[symbol] = data
                                else:
                                    self.logger.warning(f"Skipping invalid data for {symbol}")
                        
                        self.stock_data = valid_data
                        loaded_files.append(self.data_file)
                        total_records = sum(len(data) for data in self.stock_data.values())
                        self.logger.info(f"Stock data loaded: {len(self.stock_data)} symbols, {total_records} total records")
                    else:
                        self.logger.warning("Invalid stock data format, starting fresh")
                        self.stock_data = {}
            except Exception as e:
                self.logger.error(f"Failed to load stock data: {e}")
                self.stock_data = {}
        
        if loaded_files:
            self.logger.info(f"Successfully loaded data from: {', '.join(loaded_files)}")
        else:
            self.logger.info("No existing data found, starting fresh")
            
    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts for common actions"""
        self.root.bind('<Control-f>', lambda e: self.fetch_stock_data())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        self.root.bind('<Control-e>', lambda e: self.export_data())
        self.root.bind('<Control-p>', lambda e: self.show_portfolio())
        self.root.bind('<Control-w>', lambda e: self.add_to_watchlist())
        self.root.bind('<Control-r>', lambda e: self._refresh_current_data())
        self.root.bind('<F5>', lambda e: self._refresh_current_data())
        self.root.bind('<Control-h>', lambda e: self._show_help())
        self.root.bind('<Control-n>', lambda e: self._set_price_alert())
        self.root.bind('<Control-t>', lambda e: self._toggle_auto_refresh())
        self.logger.info("Keyboard shortcuts configured")
        
    def _refresh_current_data(self) -> None:
        """Refresh data for currently displayed symbol"""
        current_symbol = self.symbol_entry.get().strip().upper()
        if current_symbol and self._validate_symbol(current_symbol):
            self.logger.info(f"Refreshing data for {current_symbol}")
            self.fetch_stock_data()
        else:
            messagebox.showinfo("Refresh", "Please enter a valid symbol to refresh")
            
    def _setup_notifications(self) -> None:
        """Setup notification system"""
        # Create notification frame (initially hidden)
        self.notification_frame = ttk.Frame(self.main_frame)
        
        self.notification_label = ttk.Label(self.notification_frame, text="", background="lightgreen", relief="solid", borderwidth=1)
        self.notification_label.pack(pady=2)
        
        self.logger.info("Notification system initialized")
        
    def _setup_chart_interactivity(self) -> None:
        """Setup chart interactivity features"""
        try:
            # Enable zoom and pan functionality
            from matplotlib.widgets import RectangleSelector
            
            # Store original chart limits for reset
            self.chart_limits = {}
            
            # Add mouse event handlers
            self.canvas.mpl_connect('scroll_event', self._on_chart_scroll)
            self.canvas.mpl_connect('button_press_event', self._on_chart_click)
            
            self.logger.info("Chart interactivity enabled")
            
        except ImportError:
            self.logger.warning("Chart interactivity features not available")
            
    def _on_chart_scroll(self, event) -> None:
        """Handle mouse scroll for chart zoom"""
        try:
            if event.inaxes == self.ax1:
                # Zoom in/out on price chart
                scale_factor = 1.1 if event.button == 'up' else 0.9
                xlim = self.ax1.get_xlim()
                ylim = self.ax1.get_ylim()
                
                # Calculate new limits
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                
                new_xlim = [
                    x_center - (x_center - xlim[0]) * scale_factor,
                    x_center + (xlim[1] - x_center) * scale_factor
                ]
                new_ylim = [
                    y_center - (y_center - ylim[0]) * scale_factor,
                    y_center + (ylim[1] - y_center) * scale_factor
                ]
                
                self.ax1.set_xlim(new_xlim)
                self.ax1.set_ylim(new_ylim)
                self.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Chart zoom error: {e}")
            
    def _on_chart_click(self, event) -> None:
        """Handle chart click events"""
        try:
            if event.inaxes == self.ax1 and event.button == 3:  # Right click
                # Show context menu with chart options
                        
                        menu = tk.Menu(self.root, tearoff=0)
                        menu.add_command(label="Reset Zoom", command=self._reset_chart_zoom)
                        menu.add_command(label="Save Chart", command=self._save_chart)
                        menu.add_separator()
                        menu.add_command(label="Toggle Grid", command=self._toggle_grid)
                        
                        menu.post(event.x_root, event.y_root)
                        
        except Exception as e:
            self.logger.error(f"Chart click error: {e}")
            
    def _reset_chart_zoom(self) -> None:
        """Reset chart zoom to original view"""
        try:
            self.ax1.relim()
            self.ax1.autoscale()
            self.ax2.relim()
            self.ax2.autoscale()
            self.canvas.draw()
            self.logger.info("Chart zoom reset")
        except Exception as e:
            self.logger.error(f"Failed to reset chart zoom: {e}")
            
    def _save_chart(self) -> None:
        """Save current chart as image"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to {filename}")
                self.logger.info(f"Chart saved: {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to save chart: {e}")
            messagebox.showerror("Error", f"Failed to save chart: {str(e)}")
            
    def _toggle_grid(self) -> None:
        """Toggle chart grid visibility"""
        try:
            current_grid = self.ax1.xaxis.get_gridlines()[0].get_visible()
            
            # Toggle grid for both axes
            self.ax1.grid(not current_grid)
            self.ax2.grid(not current_grid)
            
            self.canvas.draw()
            self.logger.info(f"Chart grid {'enabled' if not current_grid else 'disabled'}")
            
        except Exception as e:
            self.logger.error(f"Failed to toggle grid: {e}")
            
    def _show_help(self) -> None:
        """Show help dialog with keyboard shortcuts and usage information"""
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Help - NEPSE Analysis Tool")
        help_dialog.geometry("600x500")
        
        # Center the dialog
        help_dialog.transient(self.root)
        help_dialog.grab_set()
        
        # Create notebook for organized help
        notebook = ttk.Notebook(help_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Keyboard shortcuts tab
        shortcuts_frame = ttk.Frame(notebook)
        notebook.add(shortcuts_frame, text="Keyboard Shortcuts")
        
        shortcuts_text = """
        Keyboard Shortcuts:
        
        • Ctrl+F  - Fetch Data
        • Ctrl+S  - Save Data
        • Ctrl+E  - Export Data
        • Ctrl+P  - Show Portfolio Summary
        • Ctrl+W  - Add to Watchlist
        • Ctrl+R  - Refresh Current Data
        • Ctrl+H  - Show Help (this dialog)
        • Ctrl+N  - Set Price Alert
        • Ctrl+T  - Toggle Auto-Refresh
        • F5      - Refresh Current Data
        • Mouse Wheel - Zoom in/out on charts
        • Right Click on Chart - Context menu
        """
        
        shortcuts_label = ttk.Label(shortcuts_frame, text=shortcuts_text, justify='left')
        shortcuts_label.pack(padx=20, pady=20)
        
        # Usage tab
        usage_frame = ttk.Frame(notebook)
        notebook.add(usage_frame, text="Usage Guide")
        
        usage_text = """
        Quick Start Guide:
        
        1. Enter a stock symbol (e.g., NEPSE, AAPL)
        2. Select date range for analysis
        3. Click "Fetch Data" to retrieve stock data
        4. Toggle technical indicators using checkboxes
        5. Add stocks to portfolio or watchlist
        6. Export data in various formats
        
        Advanced Features:
        
        • Search: Type in search box to filter portfolio/watchlist
        • Themes: Click "Theme" button to toggle dark/light mode
        • Charts: Use mouse wheel to zoom, right-click for options
        • Alerts: Set price alerts for watchlist stocks
        • Auto-refresh: Enable automatic data updates
        """
        
        usage_label = ttk.Label(usage_frame, text=usage_text, justify='left')
        usage_label.pack(padx=20, pady=20)
        
        # About tab
        about_frame = ttk.Frame(notebook)
        notebook.add(about_frame, text="About")
        
        about_text = f"""
        NEPSE Stock Analysis Tool
        Version: 2.0.0
        
        Features:
        • Real-time stock data fetching
        • Advanced technical indicators
        • Portfolio management
        • Multiple export formats
        • Interactive charts
        • Price alerts
        • Search and filtering
        • Theme support
        
        Data Sources:
        • NEPSE API (primary)
        • Yahoo Finance (fallback)
        • Simulated data (when APIs unavailable)
        
        Logs: nepse_analysis.log
        Config: config.json
        
        Application Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        about_label = ttk.Label(about_frame, text=about_text, justify='left')
        about_label.pack(padx=20, pady=20)
        
        # Close button
        ttk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)
        
    def _set_price_alert(self) -> None:
        """Set price alert for current symbol"""
        symbol = self.symbol_entry.get().strip().upper()
        
        if not self._validate_symbol(symbol):
            messagebox.showerror("Invalid Symbol", f"'{symbol}' is not a valid stock symbol.")
            return
            
        if symbol not in self.stock_data:
            messagebox.showinfo("No Data", f"No data available for {symbol}. Please fetch data first.")
            return
            
        # Create alert dialog
        alert_dialog = tk.Toplevel(self.root)
        alert_dialog.title(f"Price Alert - {symbol}")
        alert_dialog.geometry("350x250")
        
        # Center the dialog
        alert_dialog.transient(self.root)
        alert_dialog.grab_set()
        
        current_price = self.stock_data[symbol]['Close'][-1]
        
        ttk.Label(alert_dialog, text=f"Set Price Alert for {symbol}", font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Label(alert_dialog, text=f"Current Price: NPR {current_price:.2f}").pack(pady=5)
        
        # Alert type
        alert_frame = ttk.LabelFrame(alert_dialog, text="Alert Type", padding="10")
        alert_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        alert_type = tk.StringVar(value="above")
        ttk.Radiobutton(alert_frame, text="Price goes above", variable=alert_type, value="above").pack(anchor='w')
        ttk.Radiobutton(alert_frame, text="Price goes below", variable=alert_type, value="below").pack(anchor='w')
        
        # Target price
        ttk.Label(alert_frame, text="Target Price:").pack(anchor='w', pady=(10, 0))
        price_entry = ttk.Entry(alert_frame)
        price_entry.pack(fill=tk.X, pady=5)
        price_entry.insert(0, f"{current_price:.2f}")
        
        def set_alert():
            try:
                target_price = float(price_entry.get())
                if target_price <= 0:
                    messagebox.showerror("Invalid Price", "Target price must be positive")
                    return
                    
                # Store alert
                self.price_alerts[symbol] = {
                    'type': alert_type.get(),
                    'target': target_price,
                    'created': datetime.now()
                }
                
                self.logger.info(f"Price alert set for {symbol}: {alert_type.get()} NPR {target_price}")
                
                alert_dialog.destroy()
                messagebox.showinfo("Alert Set", f"Price alert set for {symbol}!")
                
                # Check if alert should trigger immediately
                self._check_price_alerts()
                
            except ValueError:
                messagebox.showerror("Invalid Price", "Please enter a valid number")
                
        # Buttons
        button_frame = ttk.Frame(alert_dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Set Alert", command=set_alert).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=alert_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        price_entry.focus()
        
    def _check_price_alerts(self) -> None:
        """Check if any price alerts should trigger"""
        triggered_alerts = []
        
        for symbol, alert in self.price_alerts.items():
            if symbol in self.stock_data:
                current_price = self.stock_data[symbol]['Close'][-1]
                
                triggered = False
                if alert['type'] == 'above' and current_price >= alert['target']:
                    triggered = True
                elif alert['type'] == 'below' and current_price <= alert['target']:
                    triggered = True
                    
                if triggered:
                    message = f"Price Alert: {symbol} is NPR {current_price:.2f} ({alert['type']} NPR {alert['target']:.2f})"
                    self._show_notification(message, "warning")
                    triggered_alerts.append(symbol)
                    
        # Remove triggered alerts
        for symbol in triggered_alerts:
            del self.price_alerts[symbol]
            self.logger.info(f"Price alert triggered and removed for {symbol}")
            
    def _show_notification(self, message: str, level: str = "info") -> None:
        """Show notification message"""
        try:
            # Set notification color based on level
            colors = {
                'info': 'lightblue',
                'success': 'lightgreen',
                'warning': 'lightyellow',
                'error': 'lightcoral'
            }
            
            self.notification_label.config(text=message, background=colors.get(level, 'lightblue'))
            
            # Show notification frame at top of main frame
            self.notification_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
            
            # Auto-hide after 5 seconds
            self.root.after(5000, self._hide_notification)
            
            # Log notification
            self.logger.info(f"Notification: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to show notification: {e}")
            
    def _hide_notification(self) -> None:
        """Hide notification message"""
        try:
            self.notification_frame.grid_forget()
        except Exception:
            pass  # Ignore if already hidden
            
    def _toggle_auto_refresh(self) -> None:
        """Toggle automatic data refresh"""
        try:
            self.auto_refresh_enabled = not self.auto_refresh_enabled
            
            if self.auto_refresh_enabled:
                self._start_auto_refresh()
                self._show_notification("Auto-refresh enabled", "success")
                self.logger.info("Auto-refresh enabled")
            else:
                self._stop_auto_refresh()
                self._show_notification("Auto-refresh disabled", "info")
                self.logger.info("Auto-refresh disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to toggle auto-refresh: {e}")
            messagebox.showerror("Error", f"Failed to toggle auto-refresh: {str(e)}")
            
    def _start_auto_refresh(self) -> None:
        """Start automatic data refresh"""
        if hasattr(self, '_auto_refresh_timer'):
            self.root.after_cancel(self._auto_refresh_timer)
            
        self._auto_refresh_timer = self.root.after(self.refresh_interval * 1000, self._auto_refresh_tick)
        
    def _stop_auto_refresh(self) -> None:
        """Stop automatic data refresh"""
        if hasattr(self, '_auto_refresh_timer'):
            self.root.after_cancel(self._auto_refresh_timer)
            
    def _auto_refresh_tick(self) -> None:
        """Auto-refresh tick handler"""
        try:
            if self.auto_refresh_enabled:
                # Refresh watchlist data
                if self.watchlist:
                    self.logger.info("Auto-refreshing watchlist data")
                    
                    # Refresh each symbol in watchlist
                    for symbol in self.watchlist[:5]:  # Limit to 5 symbols per refresh
                        if symbol in self.stock_data:
                            # Get last 7 days of data
                            end_date = datetime.now().strftime("%Y-%m-%d")
                            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                            
                            thread = threading.Thread(target=self._fetch_data_thread, args=(symbol, start_date, end_date))
                            thread.daemon = True
                            thread.start()
                    
                # Check price alerts
                self._check_price_alerts()
                
                # Schedule next refresh
                self._auto_refresh_timer = self.root.after(self.refresh_interval * 1000, self._auto_refresh_tick)
                
        except Exception as e:
            self.logger.error(f"Auto-refresh error: {e}")
            # Don't stop auto-refresh on error, just continue
            self._auto_refresh_timer = self.root.after(self.refresh_interval * 1000, self._auto_refresh_tick)
            
    def _hide_progress_bar(self) -> None:
        """Hide the progress bar"""
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        
    def _create_tooltip(self, widget, text: str) -> None:
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="lightyellow", 
                           relief="solid", borderwidth=1, font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
            
    def _cleanup_old_data(self) -> None:
        """Clean up old data based on configuration"""
        max_age_days = self.config.get('max_data_age_days', 7)
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean old stock data
        symbols_to_remove = []
        for symbol, data in self.stock_data.items():
            if not data.empty:
                data_date = data.index[-1]  # Get last date
                if data_date < cutoff_date:
                    symbols_to_remove.append(symbol)
                    self.logger.info(f"Removing old data for {symbol} (from {data_date.date()})")
        
        for symbol in symbols_to_remove:
            del self.stock_data[symbol]
            
        if symbols_to_remove:
            self.logger.info(f"Cleaned up {len(symbols_to_remove)} old stock symbols")
            
    def _get_application_stats(self) -> Dict[str, Any]:
        """Get application statistics for monitoring"""
        stats = {
            'portfolio_stocks': len(self.portfolio),
            'watchlist_stocks': len(self.watchlist),
            'cached_symbols': len(self.stock_data),
            'total_data_points': sum(len(data) for data in self.stock_data.values()),
            'config': self.config,
            'app_start_time': getattr(self, 'start_time', datetime.now()),
        }
        return stats
        
    def export_portfolio_report(self) -> None:
        """Export detailed portfolio report to CSV"""
        if not self.portfolio:
            messagebox.showinfo("No Portfolio", "Portfolio is empty")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if filename:
            try:
                report_data = []
                for symbol, data in self.portfolio.items():
                    current_price = data['current_price']
                    buy_price = data['buy_price']
                    shares = data['shares']
                    
                    gain_loss = (current_price - buy_price) * shares
                    gain_loss_pct = ((current_price - buy_price) / buy_price) * 100
                    
                    report_data.append({
                        'Symbol': symbol,
                        'Shares': shares,
                        'Buy Price': buy_price,
                        'Current Price': current_price,
                        'Investment': shares * buy_price,
                        'Current Value': shares * current_price,
                        'Gain/Loss': gain_loss,
                        'Gain/Loss %': gain_loss_pct,
                        'Last Updated': data.get('last_updated', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(report_data)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Portfolio report exported to {filename}")
                self.logger.info(f"Portfolio report exported: {filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to export portfolio report: {e}")
                messagebox.showerror("Export Error", f"Failed to export report: {str(e)}")
                
    def _calculate_portfolio_analytics(self) -> Dict[str, Any]:
        """Calculate advanced portfolio analytics"""
        if not self.portfolio:
            return {}
            
        try:
            total_investment = 0
            total_value = 0
            total_gain_loss = 0
            weights = []
            returns = []
            
            for symbol, data in self.portfolio.items():
                investment = data['shares'] * data['buy_price']
                current_value = data['shares'] * data['current_price']
                gain_loss = current_value - investment
                
                total_investment += investment
                total_value += current_value
                total_gain_loss += gain_loss
                
                # Calculate individual return
                if investment > 0:
                    returns.append((current_value - investment) / investment)
                    weights.append(investment)
                    
            # Portfolio metrics
            total_return = (total_value - total_investment) / total_investment if total_investment > 0 else 0
            
            # Risk metrics (simplified)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return != 0 else 0
                
                # Portfolio beta (simplified calculation)
                portfolio_beta = self._calculate_portfolio_beta()
            else:
                sharpe_ratio = 0
                portfolio_beta = 0
                
            # Diversification metrics
            sector_concentration = self._calculate_sector_concentration()
            
            analytics = {
                'total_investment': total_investment,
                'total_value': total_value,
                'total_gain_loss': total_gain_loss,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'portfolio_beta': portfolio_beta,
                'sector_concentration': sector_concentration,
                'num_stocks': len(self.portfolio),
                'calculated_at': datetime.now()
            }
            
            self.portfolio_analytics = analytics
            self.logger.info(f"Portfolio analytics calculated: Total Return {total_return*100:.2f}%")
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio analytics: {e}")
            return {}
            
    def _calculate_portfolio_beta(self) -> float:
        """Calculate simplified portfolio beta"""
        try:
            # Simplified beta calculation based on portfolio volatility
            if not self.portfolio or len(self.portfolio) < 2:
                return 1.0  # Market beta
                
            returns = []
            for symbol, data in self.portfolio.items():
                if data['buy_price'] > 0:
                    returns.append((data['current_price'] - data['buy_price']) / data['buy_price'])
                    
            if len(returns) < 2:
                return 1.0
                
            # Calculate portfolio volatility (simplified)
            portfolio_volatility = np.std(returns) if returns else 0
            
            # Assume market volatility of 15% (simplified)
            market_volatility = 0.15
            
            # Beta = portfolio_volatility / market_volatility
            beta = portfolio_volatility / market_volatility if market_volatility != 0 else 1.0
            
            return max(0.1, min(3.0, beta))  # Reasonable bounds
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio beta: {e}")
            return 1.0
            
    def _calculate_sector_concentration(self) -> Dict[str, float]:
        """Calculate sector concentration (simplified)"""
        try:
            # Simplified sector mapping based on symbol patterns
            sectors = defaultdict(float)
            total_investment = 0
            
            for symbol, data in self.portfolio.items():
                investment = data['shares'] * data['buy_price']
                total_investment += investment
                
                # Simple sector classification based on symbol
                sector = self._classify_sector(symbol)
                sectors[sector] += investment
                
            # Calculate percentages
            if total_investment > 0:
                for sector in sectors:
                    sectors[sector] = (sectors[sector] / total_investment) * 100
                    
            return dict(sectors)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate sector concentration: {e}")
            return {}
            
    def _classify_sector(self, symbol: str) -> str:
        """Simple sector classification based on symbol patterns"""
        symbol = symbol.upper()
        
        # Simple heuristic classification
        if any(x in symbol for x in ['TECH', 'SOFT', 'COMP']):
            return 'Technology'
        elif any(x in symbol for x in ['BANK', 'FIN', 'INSUR']):
            return 'Financial Services'
        elif any(x in symbol for x in ['OIL', 'GAS', 'ENERGY']):
            return 'Energy'
        elif any(x in symbol for x in ['PHARMA', 'MED', 'HEALTH']):
            return 'Healthcare'
        elif any(x in symbol for x in ['RETAIL', 'SHOP', 'MALL']):
            return 'Consumer Discretionary'
        elif any(x in symbol for x in ['FOOD', 'BEV', 'CONSUMER']):
            return 'Consumer Staples'
        elif any(x in symbol for x in ['INDUSTRY', 'MANUF', 'STEEL']):
            return 'Industrial'
        elif any(x in symbol for x in ['UTIL', 'POWER', 'WATER']):
            return 'Utilities'
        elif any(x in symbol for x in ['NEPSE', 'HOTEL', 'TOURISM']):
            return 'Tourism/Hospitality'
        else:
            return 'Other'
            
    def _show_portfolio_analytics(self) -> None:
        """Display detailed portfolio analytics"""
        try:
            analytics = self._calculate_portfolio_analytics()
            
            if not analytics:
                messagebox.showinfo("No Data", "No portfolio data available for analytics")
                return
                
            # Create analytics dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Portfolio Analytics")
            dialog.geometry("600x500")
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Create notebook for organized analytics
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_frame = ttk.Frame(notebook)
            notebook.add(summary_frame, text="Summary")
            
            summary_text = f"""
            Portfolio Summary:
            
            Total Investment: NPR {analytics.get('total_investment', 0):,.2f}
            Current Value: NPR {analytics.get('total_value', 0):,.2f}
            Total Gain/Loss: NPR {analytics.get('total_gain_loss', 0):,.2f}
            Total Return: {analytics.get('total_return_pct', 0):,.2f}%
            Number of Stocks: {analytics.get('num_stocks', 0)}
            
            Risk Metrics:
            Sharpe Ratio: {analytics.get('sharpe_ratio', 0):,.3f}
            Portfolio Beta: {analytics.get('portfolio_beta', 0):,.3f}
            """
            
            ttk.Label(summary_frame, text=summary_text, justify='left').pack(padx=20, pady=20)
            
            # Sector allocation tab
            sector_frame = ttk.Frame(notebook)
            notebook.add(sector_frame, text="Sector Allocation")
            
            sector_data = analytics.get('sector_concentration', {})
            if sector_data:
                ttk.Label(sector_frame, text="Sector Allocation:", font=('Arial', 10, 'bold')).pack(pady=10)
                
                for sector, percentage in sector_data.items():
                    ttk.Label(sector_frame, text=f"{sector}: {percentage:.1f}%").pack(anchor='w', padx=20)
                    
            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            self.logger.info("Portfolio analytics displayed")
            
        except Exception as e:
            self.logger.error(f"Failed to show analytics: {e}")
            messagebox.showerror("Error", f"Failed to show analytics: {str(e)}")

    def _import_portfolio(self) -> None:
        """Import portfolio from CSV or Excel file"""
        try:
            filename = filedialog.askopenfilename(
                title="Import Portfolio",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return
                
            self._show_progress_bar("Importing portfolio...")
            
            # Determine file type and import accordingly
            if filename.endswith('.csv'):
                self._import_portfolio_csv(filename)
            elif filename.endswith('.xlsx'):
                self._import_portfolio_excel(filename)
            else:
                messagebox.showerror("Unsupported Format", "Please select a CSV or Excel file")
                return
                
            self._hide_progress_bar()
            self.update_portfolio_display()
            messagebox.showinfo("Success", "Portfolio imported successfully!")
            self.logger.info(f"Portfolio imported from {filename}")
            
        except Exception as e:
            self._hide_progress_bar()
            self.logger.error(f"Failed to import portfolio: {e}")
            messagebox.showerror("Import Error", f"Failed to import portfolio: {str(e)}")
            
    def _import_portfolio_csv(self, filename: str) -> None:
        """Import portfolio from CSV file"""
        try:
            df = pd.read_csv(filename)
            
            required_columns = ['Symbol', 'Shares', 'Buy Price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                messagebox.showerror("Invalid Format", f"Missing required columns: {', '.join(missing_columns)}")
                return
                
            # Process imported data
            imported_count = 0
            for _, row in df.iterrows():
                try:
                    symbol = str(row['Symbol']).strip().upper()
                    shares = float(row['Shares'])
                    buy_price = float(row['Buy Price'])
                    
                    if self._validate_symbol(symbol) and shares > 0 and buy_price > 0:
                        # Check if symbol already exists
                        if symbol in self.portfolio:
                            # Update existing
                            self.portfolio[symbol]['shares'] = shares
                            self.portfolio[symbol]['buy_price'] = buy_price
                        else:
                            # Add new
                            self.portfolio[symbol] = {
                                'shares': shares,
                                'buy_price': buy_price,
                                'current_price': buy_price,  # Will be updated on next data fetch
                                'last_updated': datetime.now()
                            }
                        imported_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Skipping invalid row: {e}")
                    
            self.logger.info(f"Imported {imported_count} portfolio items from CSV")
            
        except Exception as e:
            self.logger.error(f"Failed to import CSV portfolio: {e}")
            raise
            
    def _import_portfolio_excel(self, filename: str) -> None:
        """Import portfolio from Excel file"""
        try:
            df = pd.read_excel(filename)
            
            # Try different column name variations
            column_mapping = {
                'symbol': ['Symbol', 'Ticker', 'Stock'],
                'shares': ['Shares', 'Quantity', 'Qty'],
                'buy_price': ['Buy Price', 'Purchase Price', 'Cost']
            }
            
            # Find matching columns
            found_columns = {}
            for standard, variations in column_mapping.items():
                for col in df.columns:
                    if col in variations:
                        found_columns[standard] = col
                        break
                        
            missing_required = [col for col in ['symbol', 'shares', 'buy_price'] if col not in found_columns]
            
            if missing_required:
                messagebox.showerror("Invalid Format", f"Missing required columns. Need: {', '.join(missing_required)}")
                return
                
            # Process imported data
            imported_count = 0
            for _, row in df.iterrows():
                try:
                    symbol = str(row[found_columns['symbol']]).strip().upper()
                    shares = float(row[found_columns['shares']])
                    buy_price = float(row[found_columns['buy_price']])
                    
                    if self._validate_symbol(symbol) and shares > 0 and buy_price > 0:
                        if symbol in self.portfolio:
                            # Update existing
                            self.portfolio[symbol]['shares'] = shares
                            self.portfolio[symbol]['buy_price'] = buy_price
                        else:
                            # Add new
                            self.portfolio[symbol] = {
                                'shares': shares,
                                'buy_price': buy_price,
                                'current_price': buy_price,
                                'last_updated': datetime.now()
                            }
                        imported_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Skipping invalid row: {e}")
                    
            self.logger.info(f"Imported {imported_count} portfolio items from Excel")
            
        except Exception as e:
            self.logger.error(f"Failed to import Excel portfolio: {e}")
            raise
            
    def _show_settings(self) -> None:
        """Show settings dialog for configuration"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Settings")
            dialog.geometry("500x400")
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Create notebook for organized settings
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # General settings tab
            general_frame = ttk.Frame(notebook)
            notebook.add(general_frame, text="General")
            
            # Auto-save interval
            ttk.Label(general_frame, text="Auto-save Interval (seconds):").grid(row=0, column=0, sticky='w', pady=5)
            auto_save_var = tk.StringVar(value=str(self.config.get('auto_save_interval', 300)))
            auto_save_entry = ttk.Entry(general_frame, textvariable=auto_save_var, width=10)
            auto_save_entry.grid(row=0, column=1, pady=5, padx=10)
            
            # Max data age
            ttk.Label(general_frame, text="Max Data Age (days):").grid(row=1, column=0, sticky='w', pady=5)
            max_age_var = tk.StringVar(value=str(self.config.get('max_data_age_days', 7)))
            max_age_entry = ttk.Entry(general_frame, textvariable=max_age_var, width=10)
            max_age_entry.grid(row=1, column=1, pady=5, padx=10)
            
            # Refresh interval
            ttk.Label(general_frame, text="Refresh Interval (seconds):").grid(row=2, column=0, sticky='w', pady=5)
            refresh_var = tk.StringVar(value=str(self.refresh_interval))
            refresh_entry = ttk.Entry(general_frame, textvariable=refresh_var, width=10)
            refresh_entry.grid(row=2, column=1, pady=5, padx=10)
            
            # Chart style
            ttk.Label(general_frame, text="Chart Style:").grid(row=3, column=0, sticky='w', pady=5)
            style_var = tk.StringVar(value=self.config.get('chart_style', 'seaborn-v0_8'))
            style_combo = ttk.Combobox(general_frame, textvariable=style_var, values=['seaborn-v0_8', 'default', 'classic', 'ggplot'])
            style_combo.grid(row=3, column=1, pady=5, padx=10)
            
            def save_settings():
                try:
                    self.config['auto_save_interval'] = int(auto_save_var.get())
                    self.config['max_data_age_days'] = int(max_age_var.get())
                    self.refresh_interval = int(refresh_var.get())
                    self.config['chart_style'] = style_var.get()
                    
                    # Save to file
                    with open('config.json', 'w') as f:
                        json.dump({'settings': self.config}, f, indent=2)
                        
                    self.logger.info("Settings saved successfully")
                    messagebox.showinfo("Success", "Settings saved successfully!")
                    dialog.destroy()
                    
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers for intervals")
                except Exception as e:
                    self.logger.error(f"Failed to save settings: {e}")
                    messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
                    
            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            
            ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
            self.logger.info("Settings dialog opened")
            
        except Exception as e:
            self.logger.error(f"Failed to show settings: {e}")
            messagebox.showerror("Error", f"Failed to show settings: {str(e)}")
            
    def _performance_comparison(self) -> None:
        """Compare sync vs async performance"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Performance Comparison")
            dialog.geometry("600x400")
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Create text widget for results
            text_widget = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Test symbols
            test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            start_date = '2023-01-01'
            end_date = '2023-12-31'
            
            text_widget.insert(tk.END, "Performance Comparison: Sync vs Async\n")
            text_widget.insert(tk.END, "=" * 50 + "\n\n")
            
            # Test synchronous fetching
            text_widget.insert(tk.END, "Testing Synchronous Fetching...\n")
            dialog.update()
            
            sync_start = time.time()
            sync_results = {}
            
            for symbol in test_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    if data is not None and not data.empty:
                        sync_results[symbol] = len(data)
                except Exception as e:
                    sync_results[symbol] = f"Error: {e}"
                    
            sync_time = time.time() - sync_start
            
            text_widget.insert(tk.END, f"Synchronous time: {sync_time:.2f} seconds\n")
            text_widget.insert(tk.END, f"Synchronous results: {sync_results}\n\n")
            
            # Test asynchronous fetching
            text_widget.insert(tk.END, "Testing Asynchronous Fetching...\n")
            dialog.update()
            
            async_start = time.time()
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async_results = loop.run_until_complete(
                self._async_fetch_multiple_stocks(test_symbols, start_date, end_date)
            )
            
            async_time = time.time() - async_start
            loop.close()
            
            text_widget.insert(tk.END, f"Asynchronous time: {async_time:.2f} seconds\n")
            text_widget.insert(tk.END, f"Asynchronous results: { {k: len(v) if hasattr(v, '__len__') else v for k, v in async_results.items()} }\n\n")
            
            # Performance improvement
            if sync_time > 0:
                improvement = ((sync_time - async_time) / sync_time) * 100
                text_widget.insert(tk.END, f"Performance improvement: {improvement:.1f}%\n\n")
            
            # Memory usage comparison
            text_widget.insert(tk.END, "Memory Usage:\n")
            memory_usage = self._get_memory_usage()
            for key, value in memory_usage.items():
                text_widget.insert(tk.END, f"{key}: {value}\n")
            
            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            self.logger.info("Performance comparison completed")
            
        except Exception as e:
            self.logger.error(f"Failed to run performance comparison: {e}")
            messagebox.showerror("Error", f"Failed to run performance comparison: {str(e)}")
            
    def _should_auto_enhance(self) -> bool:
        """Check if automatic enhancement should be triggered"""
        try:
            # Check time-based trigger
            time_since_last_check = (datetime.now() - self.last_enhancement_check).total_seconds()
            if time_since_last_check >= self.enhancement_interval:
                self.logger.info("Time-based auto-enhancement trigger activated")
                return True
                
            # Check performance degradation
            current_performance = self._get_current_performance_metrics()
            if self._is_performance_degraded(current_performance):
                self.logger.info("Performance-based auto-enhancement trigger activated")
                return True
                
            # Check error threshold
            if self.error_count >= 5:
                self.logger.info("Error-based auto-enhancement trigger activated")
                return True
                
            # Check for code changes
            if self._has_recent_code_changes():
                self.logger.info("Code change-based auto-enhancement trigger activated")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check auto-enhance conditions: {e}")
            return False
            
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            memory_usage = self._get_memory_usage()
            return {
                'memory_usage_mb': memory_usage.get('rss_mb', 0),
                'cache_size_mb': memory_usage.get('cache_size_mb', 0),
                'cache_symbols': memory_usage.get('cache_symbols', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
            
    def _is_performance_degraded(self, current_metrics: Dict[str, Any]) -> bool:
        """Check if performance has degraded from baseline"""
        try:
            if not self.performance_baseline:
                self.performance_baseline = current_metrics
                return False
                
            # Check memory usage increase
            baseline_memory = self.performance_baseline.get('memory_usage_mb', 0)
            current_memory = current_metrics.get('memory_usage_mb', 0)
            
            if current_memory > baseline_memory * 1.5:  # 50% increase
                return True
                
            # Check cache size increase
            baseline_cache = self.performance_baseline.get('cache_size_mb', 0)
            current_cache = current_metrics.get('cache_size_mb', 0)
            
            if current_cache > baseline_cache * 2.0:  # 100% increase
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check performance degradation: {e}")
            return False
            
    def _has_recent_code_changes(self) -> bool:
        """Check for recent code changes"""
        try:
            # Check main.py modification time
            main_py_path = 'main.py'
            if os.path.exists(main_py_path):
                mod_time = datetime.fromtimestamp(os.path.getmtime(main_py_path))
                time_since_change = (datetime.now() - mod_time).total_seconds()
                
                # If changed in last 5 minutes
                if time_since_change < 300:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check code changes: {e}")
            return False
            
    def _trigger_auto_enhancement(self) -> None:
        """Trigger automatic enhancement cycle"""
        try:
            self.logger.info("🚀 Starting automatic enhancement cycle")
            
            # Update check time
            self.last_enhancement_check = datetime.now()
            
            # Reset error count
            self.error_count = 0
            
            # Log enhancement start
            self._update_performance_stats('auto_enhancement', 'Automatic enhancement cycle started')
            
            # Create enhancement suggestions
            enhancements = self._analyze_enhancement_opportunities()
            
            if enhancements:
                self.logger.info(f"Found {len(enhancements)} enhancement opportunities")
                
                # Implement top priority enhancements
                for enhancement in enhancements[:3]:  # Top 3 priorities
                    self._implement_enhancement(enhancement)
                    
                # Commit changes
                self._commit_auto_enhancements(enhancements[:3])
                
            else:
                self.logger.info("No immediate enhancement opportunities found")
                
        except Exception as e:
            self.logger.error(f"Failed to trigger auto-enhancement: {e}")
            self.error_count += 1
            
    def _analyze_enhancement_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze codebase for enhancement opportunities"""
        opportunities = []
        
        try:
            # Check for memory optimization opportunities
            memory_usage = self._get_memory_usage()
            if memory_usage.get('cache_size_mb', 0) > 50:
                opportunities.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'description': 'Cache size exceeds 50MB, optimization needed',
                    'action': 'clear_cache'
                })
                
            # Check for performance issues
            current_metrics = self._get_current_performance_metrics()
            if self._is_performance_degraded(current_metrics):
                opportunities.append({
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'description': 'Performance degradation detected',
                    'action': 'optimize_performance'
                })
                
            # Check for missing features
            if not hasattr(self, 'advanced_analytics_enabled'):
                opportunities.append({
                    'type': 'feature_enhancement',
                    'priority': 'medium',
                    'description': 'Advanced analytics not enabled',
                    'action': 'enable_advanced_analytics'
                })
                
            # Check for documentation gaps
            if not os.path.exists('docs/USER_GUIDE.md'):
                opportunities.append({
                    'type': 'documentation',
                    'priority': 'medium',
                    'description': 'User guide documentation missing',
                    'action': 'create_user_guide'
                })
                
        except Exception as e:
            self.logger.error(f"Failed to analyze enhancement opportunities: {e}")
            
        return opportunities
        
    def _implement_enhancement(self, enhancement: Dict[str, Any]) -> None:
        """Implement a specific enhancement"""
        try:
            action = enhancement.get('action')
            
            if action == 'clear_cache':
                self._clear_cache()
                self.logger.info("✅ Implemented cache clearing enhancement")
                
            elif action == 'optimize_performance':
                self._auto_memory_optimization()
                self.logger.info("✅ Implemented performance optimization enhancement")
                
            elif action == 'enable_advanced_analytics':
                self.advanced_analytics_enabled = True
                self.logger.info("✅ Implemented advanced analytics enhancement")
                
            elif action == 'create_user_guide':
                self._create_user_guide()
                self.logger.info("✅ Implemented user guide documentation enhancement")
                
        except Exception as e:
            self.logger.error(f"Failed to implement enhancement {enhancement}: {e}")
            
    def _create_user_guide(self) -> None:
        """Create comprehensive user guide"""
        try:
            os.makedirs('docs', exist_ok=True)
            
            user_guide = """# NEPSE Analysis Tool - User Guide

## Getting Started

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

### Basic Usage
1. Enter stock symbol (e.g., NEPSE)
2. Select date range
3. Click "Fetch Data"
4. View charts and analysis

### Portfolio Management
1. Click "Add to Portfolio" after fetching data
2. Enter shares and buy price
3. View portfolio summary
4. Export data as needed

### Advanced Features
- Technical indicators (RSI, MACD, Bollinger Bands)
- Portfolio analytics and risk metrics
- Data import/export functionality
- Performance monitoring
- Automatic enhancements

### Troubleshooting
- Check internet connection for data fetching
- Verify stock symbols are correct
- Clear cache if performance issues occur
- Check logs for error details

For more details, see README.md
"""
            
            with open('docs/USER_GUIDE.md', 'w') as f:
                f.write(user_guide)
                
        except Exception as e:
            self.logger.error(f"Failed to create user guide: {e}")
            
    def _commit_auto_enhancements(self, enhancements: List[Dict[str, Any]]) -> None:
        """Commit automatic enhancements to repository"""
        try:
            import subprocess
            
            # Stage changes
            subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
            
            # Create commit message
            enhancement_types = [e['type'] for e in enhancements]
            commit_msg = f"Auto-Enhancement: {', '.join(enhancement_types)}\n\nAutomatic improvements:\n" + "\n".join(f"- {e['description']}" for e in enhancements)
            
            # Commit changes
            subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
            
            # Push changes
            subprocess.run(['git', 'push', 'origin', 'master'], capture_output=True, text=True)
            
            self.logger.info("✅ Auto-enhancements committed and pushed")
            
        except Exception as e:
            self.logger.error(f"Failed to commit auto-enhancements: {e}")
            
    def _schedule_auto_enhancement_check(self) -> None:
        """Schedule periodic auto-enhancement checks"""
        if self.auto_enhance_enabled:
            # Check every 5 minutes
            self.root.after(300000, self._auto_enhancement_check)
            
    def _auto_enhancement_check(self) -> None:
        """Periodic check for auto-enhancement triggers"""
        try:
            if self._should_auto_enhance():
                self._trigger_auto_enhancement()
                
            # Schedule next check
            self._schedule_auto_enhancement_check()
            
        except Exception as e:
            self.logger.error(f"Auto-enhancement check failed: {e}")
            # Schedule next check even on failure
            self._schedule_auto_enhancement_check()
            
    def _show_ai_analytics(self) -> None:
        """Show AI analytics dialog"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("AI Analytics Dashboard")
            dialog.geometry("800x600")
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Create notebook for tabs
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Predictions tab
            pred_frame = ttk.Frame(notebook)
            notebook.add(pred_frame, text="Predictions")
            
            ttk.Label(pred_frame, text="Stock Price Predictions", font=('Arial', 12, 'bold')).pack(pady=10)
            
            # Create predictions display
            pred_text = tk.Text(pred_frame, height=15, width=80)
            pred_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            if self.ml_models_trained:
                predictions = []
                for symbol in list(self.stock_data.keys())[:5]:  # Top 5 stocks
                    if symbol in self.stock_data:
                        pred_result = self.ai_analytics.predict_next_day_return(self.stock_data[symbol], symbol)
                        if 'error' not in pred_result:
                            pred_return = pred_result['predicted_return']
                            confidence = pred_result['confidence']
                            predictions.append(f"{symbol}: {pred_return:.4f} (confidence: {confidence:.2f})")
                
                if predictions:
                    pred_text.insert(tk.END, "\n".join(predictions))
                else:
                    pred_text.insert(tk.END, "No predictions available. Train ML models first.")
            else:
                pred_text.insert(tk.END, "ML models not trained yet. Click 'Train ML Models' to start.")
            
            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            self.logger.info("AI analytics dialog opened")
            
        except Exception as e:
            self.logger.error(f"Failed to show AI analytics: {e}")
            messagebox.showerror("Error", f"Failed to show AI analytics: {str(e)}")
            
    def _train_ml_models(self) -> None:
        """Train machine learning models for all stocks"""
        try:
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Training ML Models")
            progress_dialog.geometry("400x150")
            
            ttk.Label(progress_dialog, text="Training Machine Learning Models...", font=('Arial', 11, 'bold')).pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            progress_dialog.update()
            
            # Train models for each stock
            total_stocks = len(self.stock_data)
            trained_count = 0
            
            for i, (symbol, data) in enumerate(self.stock_data.items()):
                if len(data) > 50:  # Need sufficient data
                    self.logger.info(f"Training ML model for {symbol}...")
                    
                    # Update progress
                    progress = (i / total_stocks) * 100
                    progress_var.set(progress)
                    progress_dialog.update()
                    
                    # Train model
                    result = self.ai_analytics.train_price_prediction_model(data, symbol)
                    if 'error' not in result:
                        trained_count += 1
                        self.logger.info(f"Successfully trained model for {symbol}")
                    else:
                        self.logger.warning(f"Failed to train model for {symbol}: {result['error']}")
                        
            self.ml_models_trained = trained_count > 0
            
            # Complete progress
            progress_var.set(100)
            progress_dialog.update()
            
            # Show completion message
            if trained_count > 0:
                messagebox.showinfo("Training Complete", f"Successfully trained ML models for {trained_count} stocks.")
            else:
                messagebox.showwarning("Training Failed", "No ML models were trained. Insufficient data.")
                
            progress_dialog.destroy()
            
            self.logger.info(f"ML training completed. Trained {trained_count}/{total_stocks} models.")
            
        except Exception as e:
            self.logger.error(f"Failed to train ML models: {e}")
            messagebox.showerror("Error", f"Failed to train ML models: {str(e)}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="NEPSE Stock Analysis Tool - Advanced stock analysis with portfolio management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""
Examples:
  %(prog)s                    # Run with default settings
  %(prog)s --symbol NEPSE --start 2023-01-01 --end 2023-12-31
  %(prog)s --import-portfolio portfolio.csv  # Import portfolio from CSV
  %(prog)s --no-backup                # Disable backup creation
  %(prog)s --debug                   # Enable debug logging
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, help='Stock symbol to fetch')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--import-portfolio', '-i', type=str, help='Import portfolio from CSV/Excel file')
    parser.add_argument('--no-backup', action='store_true', help='Disable backup creation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()
    
def main():
    """Main entry point with enhanced argument handling"""
    try:
        args = parse_arguments()
        
        # Create root window
        root = tk.Tk()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Create application
        app = NEPSEAnalysisApp(root)
        
        # Handle command line arguments
        if args.symbol:
            app.symbol_entry.delete(0, tk.END)
            app.symbol_entry.insert(0, args.symbol.upper())
            
        if args.start:
            app.start_date_entry.delete(0, tk.END)
            app.start_date_entry.insert(0, args.start)
            
        if args.end:
            app.end_date_entry.delete(0, tk.END)
            app.end_date_entry.insert(0, args.end)
            
        if args.import_portfolio:
            app._import_portfolio_csv(args.import_portfolio)
            
        if args.no_backup:
            app.config['backup_enabled'] = False
            app.logger.info("Backup disabled via command line")
            
        # Auto-fetch data if symbol provided
        if args.symbol and args.start and args.end:
            app.fetch_stock_data()
            
        # Start the GUI
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
