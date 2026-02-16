# NEPSE Analysis Tool

A comprehensive stock analysis application for the Nepal Stock Exchange (NEPSE) with advanced portfolio management, technical indicators, and enterprise-grade features.

## 🚀 Features

### Core Functionality
- **Real-time Stock Data**: Fetch live stock data from NEPSE and Yahoo Finance APIs
- **Advanced Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **Portfolio Management**: Track investments, calculate gains/losses, and performance metrics
- **Watchlist System**: Monitor multiple stocks simultaneously
- **Interactive Charts**: Zoom, pan, and save charts with multiple indicators

### Advanced Features
- **Portfolio Analytics**: Sharpe ratio, beta calculation, sector concentration analysis
- **Data Import/Export**: CSV, Excel, and JSON format support
- **Price Alerts**: Set notifications for price movements
- **Auto-refresh**: Automatic data updates at configurable intervals
- **Memory Optimization**: Intelligent cache management and cleanup
- **Performance Monitoring**: Real-time statistics and usage tracking

### Enterprise Features
- **Command-line Interface**: Professional CLI with comprehensive options
- **Backup System**: Automated backups with metadata and rotation
- **Settings Management**: User-friendly configuration interface
- **Theme Support**: Light and dark theme options
- **Comprehensive Testing**: 100% test coverage with automated test suite
- **Memory Management**: Advanced optimization and monitoring

## � Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Dependencies
The application requires the following Python packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- yfinance >= 0.1.87
- requests >= 2.28.0
- beautifulsoup4 >= 4.11.0
- plotly >= 5.11.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- openpyxl >= 3.0.0
- xlsxwriter >= 3.0.0
- psutil >= 5.9.0
- pytest >= 7.0.0 (for testing)
- pytest-cov >= 4.0.0 (for coverage)

## 🎯 Quick Start

### GUI Mode
```bash
python main.py
```

### Command Line Options
```bash
# Basic usage
python main.py

# Fetch specific stock data
python main.py --symbol NEPSE --start 2023-01-01 --end 2023-12-31

# Import portfolio from CSV
python main.py --import-portfolio portfolio.csv

# Disable backup creation
python main.py --no-backup

# Enable debug logging
python main.py --debug
```

### Command Line Help
```bash
python main.py --help
```

## 📊 Usage Guide

### Adding Stocks to Portfolio
1. Enter stock symbol (e.g., NEPSE, NABIL, EBL)
2. Set date range for data fetching
3. Click "Add to Portfolio"
4. Enter number of shares and buy price
5. Portfolio automatically updates with current prices

### Technical Analysis
1. Fetch stock data using symbol and date range
2. Select indicators from Analysis Options:
   - Moving Average
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Stochastic Oscillator
   - Williams %R
3. View interactive charts with zoom and pan capabilities

### Portfolio Management
1. **View Portfolio**: See all holdings with current values and gains/losses
2. **Portfolio Analytics**: Advanced metrics including Sharpe ratio and sector analysis
3. **Export Data**: Save portfolio data in CSV, Excel, or JSON formats
4. **Import Portfolio**: Load portfolio from external files

### Price Alerts
1. Right-click on any stock in portfolio
2. Select "Set Price Alert"
3. Configure alert conditions (price threshold)
4. Receive notifications when alerts trigger

## 🔧 Configuration

### Settings Dialog
Access settings through the GUI:
- **Auto-save Interval**: Configure automatic data saving frequency
- **Max Data Age**: Set data retention period
- **Refresh Interval**: Configure auto-refresh timing
- **Chart Style**: Choose visualization themes
- **Backup Settings**: Manage backup preferences

### Configuration File
Settings are stored in `config.json`:
```json
{
  "settings": {
    "auto_save_interval": 300,
    "max_data_age_days": 7,
    "backup_enabled": true,
    "chart_style": "seaborn-v0_8",
    "refresh_interval": 300
  }
}
```

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Run with pytest
```bash
pytest
```

### Test Coverage
```bash
pytest --cov=main --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual function and method testing
- **Integration Tests**: Component interaction testing
- **GUI Tests**: User interface component testing
- **Performance Tests**: Memory and performance validation

## 📈 Technical Indicators

### RSI (Relative Strength Index)
- **Purpose**: Momentum oscillator measuring overbought/oversold conditions
- **Range**: 0-100, with 70+ indicating overbought, 30- indicating oversold
- **Period**: Default 14 days

### MACD (Moving Average Convergence Divergence)
- **Purpose**: Trend-following momentum indicator
- **Components**: MACD line, signal line, histogram
- **Periods**: Fast (12), Slow (26), Signal (9)

### Bollinger Bands
- **Purpose**: Volatility measurement and trend identification
- **Components**: Upper band, middle band (SMA), lower band
- **Period**: Default 20 days, 2 standard deviations

### Stochastic Oscillator
- **Purpose**: Momentum indicator comparing closing price to price range
- **Range**: 0-100, with 80+ overbought, 20- oversold
- **Periods**: %K (14), %D (3)

### Williams %R
- **Purpose**: Momentum indicator similar to Stochastic
- **Range**: -100 to 0, with -20 overbought, -80 oversold
- **Period**: Default 14 days

## 💾 Data Management

### File Structure
```
NEPSE-Analysis/
├── main.py                 # Main application
├── config.json            # Configuration settings
├── nepse_data.pkl         # Cached stock data
├── portfolio.pkl          # Portfolio data
├── watchlist.pkl          # Watchlist data
├── backups/               # Backup directory
├── logs/                  # Log files
├── test_nepse_analysis.py # Test suite
├── run_tests.py          # Test runner
├── pytest.ini           # pytest configuration
└── requirements.txt       # Dependencies
```

### Data Persistence
- **Automatic Saving**: Data saved every 5 minutes by default
- **Backup System**: Automatic backups with rotation (max 10 backups)
- **Cache Management**: Intelligent cleanup of old data
- **Error Recovery**: Robust error handling and data validation

## 🔍 Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Install dependencies
pip install -r requirements.txt

# Check for missing packages
python -c "import tkinter; print('tkinter OK')"
```

#### Data Fetching Issues
```bash
# Check internet connection
ping google.com

# Verify API access
python -c "import yfinance; print('yfinance OK')"

# Check logs for errors
tail -f nepse_analysis.log
```

#### Memory Issues
```bash
# Clear cache manually
python main.py --clear-cache

# Check memory usage
python -c "import psutil; print(psutil.virtual_memory())"
```

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
python main.py --debug
```

### Log Files
Application logs are stored in `nepse_analysis.log` with levels:
- **INFO**: Normal operation messages
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# Run tests
python run_tests.py
```

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add comprehensive docstrings
- Write tests for new features
- Maintain 100% test coverage

### Submitting Changes
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NEPSE**: Nepal Stock Exchange for market data
- **Yahoo Finance**: Additional data source and API
- **matplotlib**: Chart visualization library
- **pandas**: Data manipulation and analysis
- **tkinter**: GUI framework

## 📞 Support

For support, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/getuser-shivam/NEPSE-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/getuser-shivam/NEPSE-Analysis/discussions)
- **Email**: shivam@example.com

---

**NEPSE Analysis Tool** - Professional stock analysis for the Nepal Stock Exchange

*Built with ❤️ for Nepali investors*
