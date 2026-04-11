/// Represents a stock with its market data.
///
/// This class encapsulates all relevant information about a stock
/// including its current price, daily statistics, and trading volume.
class Stock {
  /// The stock symbol (e.g., 'NABIL', 'EBL')
  final String symbol;

  /// The company name
  final String name;

  /// Current market price
  final double currentPrice;

  /// Opening price for the day
  final double openPrice;

  /// Highest price for the day
  final double highPrice;

  /// Lowest price for the day
  final double lowPrice;

  /// Previous day's closing price
  final double previousClose;

  /// Trading volume
  final int volume;

  /// Last update timestamp
  final DateTime lastUpdated;

  /// Creates a new [Stock] instance.
  const Stock({
    required this.symbol,
    required this.name,
    required this.currentPrice,
    required this.openPrice,
    required this.highPrice,
    required this.lowPrice,
    required this.previousClose,
    required this.volume,
    required this.lastUpdated,
  });

  /// Price change from previous close.
  double get priceChange => currentPrice - previousClose;

  /// Percentage change from previous close.
  double get percentChange =>
      previousClose != 0 ? (priceChange / previousClose) * 100 : 0;

  /// Whether the stock is gaining value.
  bool get isGaining => priceChange >= 0;

  /// Price range for the day.
  double get dayRange => highPrice - lowPrice;

  @override
  String toString() {
    return 'Stock(symbol: $symbol, price: $currentPrice, change: $priceChange)';
  }
}
