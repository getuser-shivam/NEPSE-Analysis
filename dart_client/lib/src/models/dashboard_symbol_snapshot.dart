import 'portfolio_holding.dart';
import 'price_snapshot.dart';

class DashboardSymbolSnapshot {
  const DashboardSymbolSnapshot({
    required this.symbol,
    required this.inPortfolio,
    required this.inWatchlist,
    required this.holding,
    required this.recentPrices,
  });

  final String symbol;
  final bool inPortfolio;
  final bool inWatchlist;
  final PortfolioHolding? holding;
  final List<PriceSnapshot> recentPrices;

  factory DashboardSymbolSnapshot.fromJson(Map<String, dynamic> json) {
    final rawRecentPrices = json['recentPrices'] as List<dynamic>? ?? const [];

    return DashboardSymbolSnapshot(
      symbol: json['symbol'] as String,
      inPortfolio: json['inPortfolio'] as bool? ?? false,
      inWatchlist: json['inWatchlist'] as bool? ?? false,
      holding: json['holding'] == null
          ? null
          : PortfolioHolding.fromJson(
              Map<String, dynamic>.from(json['holding'] as Map),
            ),
      recentPrices: rawRecentPrices
          .map(
            (item) => PriceSnapshot.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList(growable: false),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'inPortfolio': inPortfolio,
      'inWatchlist': inWatchlist,
      'holding': holding?.toJson(),
      'recentPrices': recentPrices.map((snapshot) => snapshot.toJson()).toList(),
    };
  }
}
