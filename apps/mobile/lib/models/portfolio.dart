/// Represents a holding of a specific stock in a portfolio.
class PortfolioHolding {
  /// The stock symbol
  final String symbol;

  /// Number of shares owned
  final double shares;

  /// Average buy price per share
  final double buyPrice;

  /// Current market price per share
  double currentPrice;

  /// Last update timestamp
  DateTime lastUpdated;

  /// Creates a new [PortfolioHolding] instance.
  PortfolioHolding({
    required this.symbol,
    required this.shares,
    required this.buyPrice,
    this.currentPrice = 0.0,
    DateTime? lastUpdated,
  }) : lastUpdated = lastUpdated ?? DateTime.now();

  /// Total investment amount.
  double get totalInvestment => shares * buyPrice;

  /// Current value of the holding.
  double get currentValue => shares * currentPrice;

  /// Gain or loss amount.
  double get gainLoss => currentValue - totalInvestment;

  /// Gain or loss percentage.
  double get gainLossPercent =>
      totalInvestment > 0 ? (gainLoss / totalInvestment) * 100 : 0;

  /// Alias for gainLossPercent.
  double get gainLossPct => gainLossPercent;
}

/// Represents a complete investment portfolio.
class Portfolio {
  /// Map of symbol to holding
  final Map<String, PortfolioHolding> holdings;

  /// Portfolio name
  final String name;

  /// Creates a new [Portfolio] instance.
  Portfolio({
    required this.name,
    Map<String, PortfolioHolding>? holdings,
  }) : holdings = holdings ?? {};

  /// Total investment across all holdings.
  double get totalInvestment =>
      holdings.values.fold(0, (sum, h) => sum + h.totalInvestment);

  /// Current value across all holdings.
  double get currentValue =>
      holdings.values.fold(0, (sum, h) => sum + h.currentValue);

  /// Total gain or loss.
  double get totalGainLoss => currentValue - totalInvestment;

  /// Total gain or loss percentage.
  double get totalGainLossPercent =>
      totalInvestment > 0 ? (totalGainLoss / totalInvestment) * 100 : 0;

  /// Number of holdings.
  int get holdingCount => holdings.length;

  /// Adds a holding to the portfolio.
  void addHolding(PortfolioHolding holding) {
    holdings[holding.symbol] = holding;
  }

  /// Removes a holding by symbol.
  void removeHolding(String symbol) {
    holdings.remove(symbol);
  }

  /// Gets a holding by symbol.
  PortfolioHolding? getHolding(String symbol) => holdings[symbol];

  /// Updates prices for all holdings.
  void updatePrices(Map<String, double> prices) {
    for (final entry in prices.entries) {
      final holding = holdings[entry.key];
      if (holding != null) {
        holding.currentPrice = entry.value;
        holding.lastUpdated = DateTime.now();
      }
    }
  }
}
