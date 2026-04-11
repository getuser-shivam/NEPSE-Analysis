/// DashboardSnapshot Model Unit Tests
///
/// Tests for the DashboardSnapshot data model.

import 'package:nepse_client/src/models/app_settings.dart';
import 'package:nepse_client/src/models/dashboard_snapshot.dart';
import 'package:nepse_client/src/models/dashboard_symbol_snapshot.dart';
import 'package:nepse_client/src/models/portfolio_holding.dart';
import 'package:nepse_client/src/models/portfolio_overview.dart';
import 'package:nepse_client/src/models/portfolio_summary.dart';
import 'package:nepse_client/src/models/price_snapshot.dart';
import 'package:nepse_client/src/models/watchlist_item.dart';
import 'package:test/test.dart';

void main() {
  group('DashboardSnapshot Model', () {
    test('creates DashboardSnapshot with correct values', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final tradeDate = DateTime.utc(2024, 1, 1);

      final settings = AppSettings(
        id: 'settings-id',
        name: 'Test Settings',
        autoSaveInterval: 300,
        maxDataAgeDays: 30,
        backupEnabled: true,
        chartStyle: 'candlestick',
        defaultPeriod: '1D',
        maxWatchlistSize: 50,
        logLevel: 'info',
        refreshInterval: 60,
        createdAt: now,
        updatedAt: now,
      );

      final summary = PortfolioSummary(
        totalInvestment: 95000.0,
        totalValue: 100000.0,
        totalGainLoss: 5000.0,
        totalReturnPct: 5.26,
        numStocks: 1,
      );

      final holding = PortfolioHolding(
        id: 'holding-id',
        symbol: 'NABIL',
        shares: 100.0,
        buyPrice: 950.0,
        currentPrice: 1000.0,
        notes: null,
        lastUpdated: null,
        createdAt: now,
        updatedAt: now,
      );

      final portfolio = PortfolioOverview(
        holdings: [holding],
        summary: summary,
      );

      final watchlistItem = WatchlistItem(
        id: 'watchlist-id',
        symbol: 'NABIL',
        notes: 'Watch for breakout',
        createdAt: now,
        updatedAt: now,
      );

      final price = PriceSnapshot(
        id: 'price-id',
        symbol: 'NABIL',
        tradeDate: tradeDate,
        open: 950.0,
        high: 980.0,
        low: 940.0,
        close: 970.0,
        volume: 1000000.0,
        source: 'NEPSE',
        createdAt: now,
        updatedAt: now,
      );

      final symbolSnapshot = DashboardSymbolSnapshot(
        symbol: 'NABIL',
        inPortfolio: true,
        inWatchlist: true,
        holding: holding,
        recentPrices: [price],
      );

      final snapshot = DashboardSnapshot(
        generatedAt: now,
        settings: settings,
        portfolio: portfolio,
        watchlist: [watchlistItem],
        symbols: [symbolSnapshot],
      );

      expect(snapshot.generatedAt, equals(now));
      expect(snapshot.settings.name, equals('Test Settings'));
      expect(snapshot.portfolio.holdings.length, equals(1));
      expect(snapshot.watchlist.length, equals(1));
      expect(snapshot.symbols.length, equals(1));
    });

    test('creates DashboardSnapshot with empty lists', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);

      final settings = AppSettings(
        id: 'settings-id',
        name: 'Test Settings',
        autoSaveInterval: 300,
        maxDataAgeDays: 30,
        backupEnabled: true,
        chartStyle: 'candlestick',
        defaultPeriod: '1D',
        maxWatchlistSize: 50,
        logLevel: 'info',
        refreshInterval: 60,
        createdAt: now,
        updatedAt: now,
      );

      final summary = PortfolioSummary(
        totalInvestment: 0.0,
        totalValue: 0.0,
        totalGainLoss: 0.0,
        totalReturnPct: 0.0,
        numStocks: 0,
      );

      final portfolio = PortfolioOverview(
        holdings: [],
        summary: summary,
      );

      final snapshot = DashboardSnapshot(
        generatedAt: now,
        settings: settings,
        portfolio: portfolio,
        watchlist: [],
        symbols: [],
      );

      expect(snapshot.watchlist, isEmpty);
      expect(snapshot.symbols, isEmpty);
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final tradeDate = DateTime.utc(2024, 1, 1);

      final settings = AppSettings(
        id: 'settings-id',
        name: 'Test Settings',
        autoSaveInterval: 300,
        maxDataAgeDays: 30,
        backupEnabled: true,
        chartStyle: 'candlestick',
        defaultPeriod: '1D',
        maxWatchlistSize: 50,
        logLevel: 'info',
        refreshInterval: 60,
        createdAt: now,
        updatedAt: now,
      );

      final summary = PortfolioSummary(
        totalInvestment: 0.0,
        totalValue: 0.0,
        totalGainLoss: 0.0,
        totalReturnPct: 0.0,
        numStocks: 0,
      );

      final portfolio = PortfolioOverview(
        holdings: [],
        summary: summary,
      );

      final snapshot = DashboardSnapshot(
        generatedAt: now,
        settings: settings,
        portfolio: portfolio,
        watchlist: [],
        symbols: [],
      );

      final json = snapshot.toJson();

      expect(json['generatedAt'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['settings'], isNotNull);
      expect(json['portfolio'], isNotNull);
      expect(json['watchlist'], isEmpty);
      expect(json['symbols'], isEmpty);
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'generatedAt': '2024-01-01T12:00:00.000Z',
        'settings': {
          'id': 'settings-id',
          'name': 'Test Settings',
          'autoSaveInterval': 300,
          'maxDataAgeDays': 30,
          'backupEnabled': true,
          'chartStyle': 'candlestick',
          'defaultPeriod': '1D',
          'maxWatchlistSize': 50,
          'logLevel': 'info',
          'refreshInterval': 60,
          'createdAt': '2024-01-01T12:00:00.000Z',
          'updatedAt': '2024-01-01T12:00:00.000Z',
        },
        'portfolio': {
          'holdings': [],
          'summary': {
            'totalInvestment': 0,
            'totalValue': 0,
            'totalGainLoss': 0,
            'totalReturnPct': 0,
            'numStocks': 0,
          }
        },
        'watchlist': [],
        'symbols': []
      };

      final snapshot = DashboardSnapshot.fromJson(json);

      expect(snapshot.generatedAt, isNotNull);
      expect(snapshot.settings.name, equals('Test Settings'));
      expect(snapshot.watchlist, isEmpty);
      expect(snapshot.symbols, isEmpty);
    });

    test('handles missing list fields with defaults', () {
      final json = {
        'generatedAt': '2024-01-01T12:00:00.000Z',
        'settings': {
          'id': 'settings-id',
          'name': 'Test Settings',
          'autoSaveInterval': 300,
          'maxDataAgeDays': 30,
          'backupEnabled': true,
          'chartStyle': 'candlestick',
          'defaultPeriod': '1D',
          'maxWatchlistSize': 50,
          'logLevel': 'info',
          'refreshInterval': 60,
          'createdAt': '2024-01-01T12:00:00.000Z',
          'updatedAt': '2024-01-01T12:00:00.000Z',
        },
        'portfolio': {
          'holdings': [],
          'summary': {
            'totalInvestment': 0,
            'totalValue': 0,
            'totalGainLoss': 0,
            'totalReturnPct': 0,
            'numStocks': 0,
          }
        },
      };

      final snapshot = DashboardSnapshot.fromJson(json);

      expect(snapshot.watchlist, isEmpty);
      expect(snapshot.symbols, isEmpty);
    });
  });
}
