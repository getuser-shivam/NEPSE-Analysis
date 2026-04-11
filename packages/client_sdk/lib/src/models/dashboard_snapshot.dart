import 'app_settings.dart';
import 'dashboard_symbol_snapshot.dart';
import 'portfolio_overview.dart';
import 'watchlist_item.dart';

class DashboardSnapshot {
  const DashboardSnapshot({
    required this.generatedAt,
    required this.settings,
    required this.portfolio,
    required this.watchlist,
    required this.symbols,
  });

  final DateTime generatedAt;
  final AppSettings settings;
  final PortfolioOverview portfolio;
  final List<WatchlistItem> watchlist;
  final List<DashboardSymbolSnapshot> symbols;

  factory DashboardSnapshot.fromJson(Map<String, dynamic> json) {
    final rawWatchlist = json['watchlist'] as List<dynamic>? ?? const [];
    final rawSymbols = json['symbols'] as List<dynamic>? ?? const [];

    return DashboardSnapshot(
      generatedAt: DateTime.parse(json['generatedAt'] as String),
      settings: AppSettings.fromJson(
        Map<String, dynamic>.from(json['settings'] as Map),
      ),
      portfolio: PortfolioOverview.fromJson(
        Map<String, dynamic>.from(json['portfolio'] as Map),
      ),
      watchlist: rawWatchlist
          .map(
            (item) =>
                WatchlistItem.fromJson(Map<String, dynamic>.from(item as Map)),
          )
          .toList(growable: false),
      symbols: rawSymbols
          .map(
            (item) => DashboardSymbolSnapshot.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList(growable: false),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'generatedAt': generatedAt.toIso8601String(),
      'settings': settings.toJson(),
      'portfolio': portfolio.toJson(),
      'watchlist': watchlist.map((item) => item.toJson()).toList(),
      'symbols': symbols.map((item) => item.toJson()).toList(),
    };
  }
}
