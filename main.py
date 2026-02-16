import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        except Exception as e:
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
            
    def _auto_save(self) -> None:
        """Automatically save data"""
        try:
            self.save_data()
            self.logger.info("Auto-save completed")
        except Exception as e:
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
        self.watchlist_frame.grid(row=16, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
    def create_widgets(self):
        # Stock Symbol Input
        ttk.Label(self.control_frame, text="Stock Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_entry = ttk.Entry(self.control_frame, width=20)
        self.symbol_entry.grid(row=0, column=1, pady=5, padx=5)
        self.symbol_entry.insert(0, "NEPSE")
        
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
        ttk.Button(self.control_frame, text="Export Data", command=self.export_data).grid(row=7, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Save Data", command=self.save_data).grid(row=8, column=0, columnspan=2, pady=5)
        
        # Analysis Options
        ttk.Separator(self.control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(self.control_frame, text="Analysis Options:", font=('Arial', 10, 'bold')).grid(row=10, column=0, columnspan=2, pady=5)
        
        self.show_ma = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Moving Average", variable=self.show_ma).grid(row=11, column=0, columnspan=2, sticky=tk.W)
        
        self.show_volume = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Volume", variable=self.show_volume).grid(row=12, column=0, columnspan=2, sticky=tk.W)
        
        self.show_rsi = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="RSI", variable=self.show_rsi).grid(row=13, column=0, columnspan=2, sticky=tk.W)
        
        self.show_macd = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="MACD", variable=self.show_macd).grid(row=14, column=0, columnspan=2, sticky=tk.W)
        
        self.show_bollinger = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Bollinger Bands", variable=self.show_bollinger).grid(row=15, column=0, columnspan=2, sticky=tk.W)
        
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
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
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
        
        # Run data fetching in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._fetch_data_thread, args=(symbol, start_date, end_date))
        thread.daemon = True
        thread.start()
        
    def _fetch_data_thread(self, symbol: str, start_date: str, end_date: str) -> None:
        """Fetch data in separate thread with comprehensive error handling"""
        try:
            self.logger.info(f"Starting data fetch for {symbol}")
            
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
            
        except Exception as e:
            error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Fetch Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Error fetching data"))
            
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
        
    def export_data(self):
        if not self.stock_data:
            messagebox.showerror("Error", "No data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            # Combine all stock data
            all_data = pd.DataFrame()
            for symbol, data in self.stock_data.items():
                data_copy = data.copy()
                data_copy['Symbol'] = symbol
                all_data = pd.concat([all_data, data_copy])
            
            all_data.to_csv(filename)
            messagebox.showinfo("Success", f"Data exported to {filename}")
            
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
            # Create backup before saving
            if self.config.get('backup_enabled', True):
                self._create_backup()
            
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

def main():
    root = tk.Tk()
    app = NEPSEAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
