# NEPSE Stock Analysis Tool

A comprehensive GUI application for analyzing Nepal Stock Exchange (NEPSE) data with advanced portfolio management capabilities and technical analysis tools.

## 🚀 Features

### Core Functionality
- **Stock Data Fetching**: Retrieve historical stock data with NEPSE API integration
- **Interactive Charts**: Beautiful, responsive charts with multiple visualization options
- **Portfolio Management**: Track investments with real-time gain/loss calculations
- **Data Export**: Export stock data to CSV for further analysis

### Technical Indicators
- **Moving Averages**: 20-day and 50-day moving averages
- **Volume Analysis**: Trading volume charts
- **RSI (Relative Strength Index)**: Momentum oscillator with overbought/oversold levels
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: Volatility bands with upper, middle, and lower bands

### Advanced Features
- **Watchlist**: Track multiple stocks with real-time price updates
- **Data Persistence**: Save and load portfolio and watchlist data
- **Enhanced GUI**: Modern interface with seaborn styling
- **Multi-source Data**: NEPSE API, Yahoo Finance, and simulated data fallback
- **Currency Support**: Proper NPR (Nepalese Rupee) formatting

## 🛠️ Installation

### Quick Start (Recommended)

**Windows:**
```bash
double-click run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### Manual Installation

1. Clone the repository:
```bash
git clone git@github.com:getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 📖 Usage Guide

### 1. **Fetching Stock Data**
   - Enter a stock symbol (e.g., "NEPSE" for the index)
   - Select date range for analysis
   - Click "Fetch Data" to retrieve historical data
   - Data is fetched from NEPSE API with Yahoo Finance fallback

### 2. **Technical Analysis**
   - Toggle indicators using checkboxes:
     - ✅ Moving Average (20/50 day)
     - ✅ Volume charts
     - ✅ RSI with overbought/oversold levels
     - ✅ MACD with signal and histogram
     - ✅ Bollinger Bands for volatility
   - Charts update automatically with your selections

### 3. **Portfolio Management**
   - Click "Add to Portfolio" after fetching stock data
   - Enter number of shares and buy price
   - View portfolio summary with:
     - Total investment and current value
     - Individual stock performance
     - Percentage returns

### 4. **Watchlist**
   - Add stocks to watchlist for quick monitoring
   - Real-time price updates and changes
   - Track multiple stocks simultaneously

### 5. **Data Management**
   - **Save Data**: Persist portfolio and watchlist
   - **Export Data**: Save all fetched data to CSV
   - **Auto-load**: Previous data loads on startup

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- yfinance >= 0.1.87
- requests >= 2.28.0
- beautifulsoup4 >= 4.11.0
- plotly >= 5.11.0
- seaborn >= 0.11.0
- scipy >= 1.9.0

*(tkinter is included with Python installation)*

## 🌐 Data Sources

This application uses multiple data sources with automatic fallback:

1. **Primary**: NEPSE API integration (live market data)
2. **Secondary**: Yahoo Finance API (international stocks)
3. **Fallback**: Simulated NEPSE data (when APIs are unavailable)

> **Note**: For production use with real-time NEPSE data, ensure you have proper API access and network connectivity.

## 🔧 Technical Details

### Architecture
- **GUI Framework**: tkinter with matplotlib integration
- **Data Processing**: pandas and numpy for analysis
- **Visualization**: matplotlib with seaborn styling
- **Technical Indicators**: Custom implementations using scipy
- **Persistence**: pickle-based data storage

### File Structure
```
NEPSE-Analysis/
├── main.py              # Main application
├── requirements.txt     # Dependencies
├── setup.py            # Auto-install script
├── run.bat             # Windows launcher
├── run.sh              # Linux/Mac launcher
├── README.md           # Documentation
├── .gitignore          # Git ignore rules
├── nepse_data.pkl      # Cached stock data
├── portfolio.pkl       # Saved portfolio
└── watchlist.pkl       # Saved watchlist
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with proper documentation
4. **Test thoroughly** on different platforms
5. **Submit a pull request** with a clear description

### Areas for Improvement
- Real-time NEPSE API integration
- Additional technical indicators
- Mobile app version
- Web-based interface
- Advanced portfolio analytics

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NEPSE (Nepal Stock Exchange) for market data
- Yahoo Finance for international stock data
- matplotlib, pandas, and numpy communities
- All contributors and users of this tool

---

**🚀 Get Started**: Double-click `run.bat` (Windows) or run `./run.sh` (Linux/Mac) to begin your NEPSE analysis journey!
