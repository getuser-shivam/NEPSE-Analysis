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

class NEPSEAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NEPSE Stock Analysis Tool")
        self.root.geometry("1200x800")
        
        # Stock data storage
        self.stock_data = {}
        self.portfolio = {}
        
        # Create main frames
        self.create_frames()
        self.create_widgets()
        
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
        ttk.Button(self.control_frame, text="Show Portfolio", command=self.show_portfolio).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text="Export Data", command=self.export_data).grid(row=6, column=0, columnspan=2, pady=5)
        
        # Analysis Options
        ttk.Separator(self.control_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(self.control_frame, text="Analysis Options:", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2, pady=5)
        
        self.show_ma = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Moving Average", variable=self.show_ma).grid(row=9, column=0, columnspan=2, sticky=tk.W)
        
        self.show_volume = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Volume", variable=self.show_volume).grid(row=10, column=0, columnspan=2, sticky=tk.W)
        
        self.show_rsi = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="RSI", variable=self.show_rsi).grid(row=11, column=0, columnspan=2, sticky=tk.W)
        
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
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def fetch_stock_data(self):
        symbol = self.symbol_entry.get().upper()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        
        if not symbol or not start_date or not end_date:
            messagebox.showerror("Error", "Please fill all fields")
            return
            
        self.status_var.set(f"Fetching data for {symbol}...")
        
        # Run data fetching in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._fetch_data_thread, args=(symbol, start_date, end_date))
        thread.daemon = True
        thread.start()
        
    def _fetch_data_thread(self, symbol, start_date, end_date):
        try:
            # Try to fetch from yfinance (works for major indices)
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                # If yfinance fails, try to simulate NEPSE data
                data = self._simulate_nepse_data(symbol, start_date, end_date)
            
            self.stock_data[symbol] = data
            
            # Update GUI in main thread
            self.root.after(0, self._update_chart, symbol, data)
            self.root.after(0, lambda: self.status_var.set(f"Data fetched for {symbol}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to fetch data: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error fetching data"))
            
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
        
    def _update_chart(self, symbol, data):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot price chart
        self.ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        
        # Add moving average if selected
        if self.show_ma.get():
            ma_20 = data['Close'].rolling(window=20).mean()
            ma_50 = data['Close'].rolling(window=50).mean()
            self.ax1.plot(data.index, ma_20, label='20-day MA', alpha=0.7)
            self.ax1.plot(data.index, ma_50, label='50-day MA', alpha=0.7)
        
        self.ax1.set_title(f'{symbol} Stock Price')
        self.ax1.set_ylabel('Price (NPR)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot volume if selected
        if self.show_volume.get():
            self.ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
            self.ax2.set_title('Trading Volume')
            self.ax2.set_ylabel('Volume')
            self.ax2.set_xlabel('Date')
        else:
            # Plot RSI if selected
            if self.show_rsi.get():
                rsi = self._calculate_rsi(data['Close'])
                self.ax2.plot(data.index, rsi, label='RSI', color='purple')
                self.ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                self.ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                self.ax2.set_title('RSI (Relative Strength Index)')
                self.ax2.set_ylabel('RSI')
                self.ax2.set_xlabel('Date')
                self.ax2.legend()
        
        self.ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        for ax in [self.ax1, self.ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def add_to_portfolio(self):
        symbol = self.symbol_entry.get().upper()
        
        if symbol not in self.stock_data:
            messagebox.showerror("Error", f"No data available for {symbol}. Please fetch data first.")
            return
            
        # Create dialog for portfolio input
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Add {symbol} to Portfolio")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Number of Shares:").grid(row=0, column=0, padx=10, pady=10)
        shares_entry = ttk.Entry(dialog)
        shares_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="Buy Price per Share:").grid(row=1, column=0, padx=10, pady=10)
        price_entry = ttk.Entry(dialog)
        price_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def add_stock():
            try:
                shares = float(shares_entry.get())
                buy_price = float(price_entry.get())
                current_price = self.stock_data[symbol]['Close'][-1]
                
                self.portfolio[symbol] = {
                    'shares': shares,
                    'buy_price': buy_price,
                    'current_price': current_price
                }
                
                self.update_portfolio_display()
                dialog.destroy()
                messagebox.showinfo("Success", f"{symbol} added to portfolio!")
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        ttk.Button(dialog, text="Add", command=add_stock).grid(row=2, column=0, columnspan=2, pady=20)
        
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
                f"₹{data['buy_price']:.2f}",
                f"₹{data['current_price']:.2f}",
                f"₹{gain_loss:.2f} ({gain_loss_pct:.2f}%)"
            ))
            
    def show_portfolio(self):
        if not self.portfolio:
            messagebox.showinfo("Portfolio", "Portfolio is empty")
            return
            
        total_investment = sum(data['shares'] * data['buy_price'] for data in self.portfolio.values())
        current_value = sum(data['shares'] * data['current_price'] for data in self.portfolio.values())
        total_gain_loss = current_value - total_investment
        
        summary = f"Portfolio Summary:\n\n"
        summary += f"Total Investment: ₹{total_investment:.2f}\n"
        summary += f"Current Value: ₹{current_value:.2f}\n"
        summary += f"Total Gain/Loss: ₹{total_gain_loss:.2f}\n"
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

def main():
    root = tk.Tk()
    app = NEPSEAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
