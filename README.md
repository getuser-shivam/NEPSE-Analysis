# NEPSE Stock Analysis Tool

A comprehensive GUI application for analyzing Nepal Stock Exchange (NEPSE) data with portfolio management capabilities.

## Features

- **Stock Data Fetching**: Retrieve historical stock data for NEPSE listed companies
- **Interactive Charts**: Visualize stock prices with moving averages, volume, and RSI indicators
- **Portfolio Management**: Track your stock investments and calculate gains/losses
- **Data Export**: Export stock data to CSV for further analysis
- **Technical Indicators**: 
  - Moving Averages (20-day and 50-day)
  - Volume analysis
  - RSI (Relative Strength Index)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:getuser-shivam/NEPSE-Analysis.git
cd NEPSE-Analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. **Fetching Stock Data**:
   - Enter a stock symbol (e.g., "NEPSE" for the index)
   - Select date range for analysis
   - Click "Fetch Data" to retrieve historical data

3. **Technical Analysis**:
   - Toggle moving averages, volume, and RSI indicators
   - Charts update automatically with your selections

4. **Portfolio Management**:
   - Click "Add to Portfolio" after fetching stock data
   - Enter number of shares and buy price
   - View portfolio summary and individual stock performance

5. **Export Data**:
   - Click "Export Data" to save all fetched data to CSV format

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- yfinance >= 0.1.87
- requests >= 2.28.0
- beautifulsoup4 >= 4.11.0
- plotly >= 5.11.0

## Note on Data Sources

This application uses multiple data sources:
- Primary: Yahoo Finance API (yfinance)
- Fallback: Simulated NEPSE data when real data is unavailable

For production use with real NEPSE data, you may need to integrate with official NEPSE APIs or data providers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
