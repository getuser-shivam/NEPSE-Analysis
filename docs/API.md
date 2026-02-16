# API Documentation

## NEPSE Analysis Tool API Reference

### Core Classes

#### NEPSEAnalysisApp
The main application class that provides all functionality for stock analysis and portfolio management.

```python
class NEPSEAnalysisApp:
    def __init__(self, root):
        """Initialize the NEPSE Analysis application"""
        
    def fetch_stock_data(self) -> None:
        """Fetch stock data for the given symbol and date range"""
        
    def add_to_portfolio(self) -> None:
        """Add current stock to portfolio"""
        
    def show_portfolio(self) -> None:
        """Display portfolio summary"""
        
    def export_data(self) -> None:
        """Export data to various formats"""
```

### Data Validation Methods

#### Symbol Validation
```python
def _validate_symbol(self, symbol: str) -> bool:
    """Validate stock symbol format
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
```

#### Date Validation
```python
def _validate_date(self, date_str: str) -> bool:
    """Validate date format and range
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        True if valid, False otherwise
    """

def _validate_date_range(self, start_date: str, end_date: str) -> bool:
    """Validate that start_date is before end_date and range is reasonable
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        True if valid, False otherwise
    """
```

### Technical Indicators

#### RSI (Relative Strength Index)
```python
def _calculate_rsi(self, prices, window=14):
    """Calculate Relative Strength Index
    
    Args:
        prices: Pandas Series of price data
        window: RSI calculation period (default: 14)
        
    Returns:
        Pandas Series containing RSI values
    """
```

#### MACD
```python
def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Pandas Series of price data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
```

#### Bollinger Bands
```python
def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
    """Calculate Bollinger Bands
    
    Args:
        prices: Pandas Series of price data
        window: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
```

#### Stochastic Oscillator
```python
def _calculate_stochastic(self, prices, k=14, d=3):
    """Calculate Stochastic oscillator
    
    Args:
        prices: Pandas Series of price data
        k: %K period (default: 14)
        d: %D period (default: 3)
        
    Returns:
        Tuple of (%K, %D) pandas Series
    """
```

#### Williams %R
```python
def _calculate_williams_r(self, prices, period=14):
    """Calculate Williams %R
    
    Args:
        prices: Pandas Series of price data
        period: Calculation period (default: 14)
        
    Returns:
        Pandas Series containing Williams %R values
    """
```

### Portfolio Analytics

#### Portfolio Analytics
```python
def _calculate_portfolio_analytics(self) -> Dict[str, Any]:
    """Calculate advanced portfolio analytics
    
    Returns:
        Dictionary containing:
        - total_investment: Total amount invested
        - total_value: Current portfolio value
        - total_gain_loss: Total gain/loss amount
        - total_return_pct: Total return percentage
        - sharpe_ratio: Portfolio Sharpe ratio
        - portfolio_beta: Portfolio beta
        - sector_concentration: Sector concentration data
        - num_stocks: Number of stocks in portfolio
    """
```

#### Sector Classification
```python
def _classify_sector(self, symbol: str) -> str:
    """Simple sector classification based on symbol patterns
    
    Args:
        symbol: Stock symbol to classify
        
    Returns:
        Sector name as string
    """
```

### Data Import/Export

#### CSV Export
```python
def _export_csv(self, filename: str, export_portfolio: bool, 
                export_watchlist: bool, export_stock_data: bool) -> None:
    """Export data to CSV format
    
    Args:
        filename: Output filename
        export_portfolio: Whether to export portfolio data
        export_watchlist: Whether to export watchlist data
        export_stock_data: Whether to export stock data
    """
```

#### Excel Export
```python
def _export_excel(self, filename: str, export_portfolio: bool, 
                  export_watchlist: bool, export_stock_data: bool) -> None:
    """Export data to Excel format
    
    Args:
        filename: Output filename
        export_portfolio: Whether to export portfolio data
        export_watchlist: Whether to export watchlist data
        export_stock_data: Whether to export stock data
    """
```

#### JSON Export
```python
def _export_json(self, filename: str, export_portfolio: bool, 
                 export_watchlist: bool, export_stock_data: bool) -> None:
    """Export data to JSON format
    
    Args:
        filename: Output filename
        export_portfolio: Whether to export portfolio data
        export_watchlist: Whether to export watchlist data
        export_stock_data: Whether to export stock data
    """
```

### Memory Management

#### Memory Usage
```python
def _get_memory_usage(self) -> Dict[str, Any]:
    """Get current memory usage statistics
    
    Returns:
        Dictionary containing:
        - rss_mb: Resident set size in MB
        - vms_mb: Virtual memory size in MB
        - cache_symbols: Number of cached symbols
        - cache_data_points: Total cached data points
        - cache_size_mb: Cache size in MB
    """
```

#### Auto Memory Optimization
```python
def _auto_memory_optimization(self) -> None:
    """Automatic memory optimization and cleanup
    
    Performs automatic cleanup when cache exceeds limits
    """
```

### Configuration

#### Settings Management
```python
def _show_settings(self) -> None:
    """Show settings dialog for configuration
    
    Displays a tabbed dialog with:
    - Auto-save interval settings
    - Data age management
    - Refresh interval configuration
    - Chart style selection
    - Backup preferences
    """
```

### Error Handling

#### Data Quality Validation
```python
def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
    """Validate the quality and integrity of fetched data
    
    Args:
        data: DataFrame containing stock data
        symbol: Stock symbol for logging
        
    Returns:
        True if data quality is acceptable, False otherwise
    """
```

### Performance Monitoring

#### Performance Statistics
```python
def _update_performance_stats(self, action: str, details: str = "") -> None:
    """Update performance statistics
    
    Args:
        action: Type of action ('data_fetch', 'chart_update', 'error')
        details: Additional details for logging
    """
```

## Usage Examples

### Basic Stock Analysis
```python
import tkinter as tk
from main import NEPSEAnalysisApp

# Create application
root = tk.Tk()
app = NEPSEAnalysisApp(root)

# Fetch stock data
app.symbol_entry.insert(0, "NEPSE")
app.start_date_entry.insert(0, "2023-01-01")
app.end_date_entry.insert(0, "2023-12-31")
app.fetch_stock_data()

# Start GUI
root.mainloop()
```

### Portfolio Management
```python
# Add stock to portfolio
app.portfolio = {
    'NEPSE': {
        'shares': 100,
        'buy_price': 1000.0,
        'current_price': 1100.0,
        'last_updated': datetime.now()
    }
}

# Calculate analytics
analytics = app._calculate_portfolio_analytics()
print(f"Total Return: {analytics['total_return_pct']:.2f}%")
```

### Technical Indicators
```python
import pandas as pd
import numpy as np

# Create sample price data
prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])

# Calculate indicators
rsi = app._calculate_rsi(prices)
macd_line, signal_line, histogram = app._calculate_macd(prices)
upper_band, middle_band, lower_band = app._calculate_bollinger_bands(prices)
k_percent, d_percent = app._calculate_stochastic(prices)
wr = app._calculate_williams_r(prices)
```

## Data Structures

### Portfolio Data Format
```python
portfolio = {
    'SYMBOL': {
        'shares': float,           # Number of shares
        'buy_price': float,        # Purchase price per share
        'current_price': float,    # Current price per share
        'last_updated': datetime   # Last update timestamp
    }
}
```

### Stock Data Format
```python
stock_data = {
    'SYMBOL': pd.DataFrame({
        'Open': float,      # Opening price
        'High': float,      # Highest price
        'Low': float,       # Lowest price
        'Close': float,     # Closing price
        'Volume': int       # Trading volume
    })
}
```

### Configuration Format
```python
config = {
    'auto_save_interval': int,      # Auto-save interval in seconds
    'max_data_age_days': int,       # Maximum data age in days
    'backup_enabled': bool,         # Whether backup is enabled
    'chart_style': str,            # Chart style name
    'refresh_interval': int         # Auto-refresh interval in seconds
}
```

## Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| 1001 | Invalid stock symbol | Check symbol format and availability |
| 1002 | Invalid date format | Use YYYY-MM-DD format |
| 1003 | Data fetch failed | Check internet connection and API availability |
| 1004 | Portfolio save failed | Check disk space and permissions |
| 1005 | Export failed | Check file permissions and disk space |

## Thread Safety

The application uses threading for data fetching to prevent GUI freezing:

- `_fetch_data_thread()`: Runs in separate thread for data fetching
- GUI updates are scheduled using `root.after()`
- Thread-safe data structures are used for shared data

## Logging

The application uses Python's logging module with the following levels:

- **DEBUG**: Detailed debugging information
- **INFO**: General information messages
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for critical issues

Log files are saved to `nepse_analysis.log`.
