/// DashboardSymbolSnapshot Model Unit Tests
///
/// Tests for the DashboardSymbolSnapshot data model.

import 'package:nepse_client/src/models/dashboard_symbol_snapshot.dart';
import 'package:nepse_client/src/models/portfolio_holding.dart';
import 'package:nepse_client/src/models/price_snapshot.dart';
import 'package:test/test.dart';

void main() {
  group('DashboardSymbolSnapshot Model', () {
    test('creates DashboardSymbolSnapshot with correct values', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final tradeDate = DateTime.utc(2024, 1, 1);
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

      expect(symbolSnapshot.symbol, equals('NABIL'));
      expect(symbolSnapshot.inPortfolio, isTrue);
      expect(symbolSnapshot.inWatchlist, isTrue);
      expect(symbolSnapshot.holding, isNotNull);
      expect(symbolSnapshot.recentPrices.length, equals(1));
    });

    test('creates DashboardSymbolSnapshot with nullable holding', () {
      final symbolSnapshot = DashboardSymbolSnapshot(
        symbol: 'NABIL',
        inPortfolio: false,
        inWatchlist: true,
        holding: null,
        recentPrices: [],
      );

      expect(symbolSnapshot.inPortfolio, isFalse);
      expect(symbolSnapshot.holding, isNull);
      expect(symbolSnapshot.recentPrices, isEmpty);
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final tradeDate = DateTime.utc(2024, 1, 1);
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

      final json = symbolSnapshot.toJson();

      expect(json['symbol'], equals('NABIL'));
      expect(json['inPortfolio'], isTrue);
      expect(json['inWatchlist'], isTrue);
      expect(json['holding'], isNotNull);
      expect(json['recentPrices'], isNotEmpty);
    });

    test('serializes nullable holding to JSON correctly', () {
      final symbolSnapshot = DashboardSymbolSnapshot(
        symbol: 'NABIL',
        inPortfolio: false,
        inWatchlist: true,
        holding: null,
        recentPrices: [],
      );

      final json = symbolSnapshot.toJson();

      expect(json['holding'], isNull);
      expect(json['recentPrices'], isEmpty);
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'symbol': 'NABIL',
        'inPortfolio': true,
        'inWatchlist': true,
        'holding': {
          'id': 'holding-id',
          'symbol': 'NABIL',
          'shares': 100,
          'buyPrice': 950.0,
          'currentPrice': 1000.0,
          'notes': null,
          'lastUpdated': null,
          'createdAt': '2024-01-01T12:00:00.000Z',
          'updatedAt': '2024-01-01T12:00:00.000Z',
        },
        'recentPrices': [
          {
            'id': 'price-id',
            'symbol': 'NABIL',
            'tradeDate': '2024-01-01T00:00:00.000Z',
            'open': 950,
            'high': 980,
            'low': 940,
            'close': 970,
            'volume': 1000000,
            'source': 'NEPSE',
            'createdAt': '2024-01-01T12:00:00.000Z',
            'updatedAt': '2024-01-01T12:00:00.000Z',
          }
        ]
      };

      final symbolSnapshot = DashboardSymbolSnapshot.fromJson(json);

      expect(symbolSnapshot.symbol, equals('NABIL'));
      expect(symbolSnapshot.inPortfolio, isTrue);
      expect(symbolSnapshot.holding, isNotNull);
      expect(symbolSnapshot.recentPrices.length, equals(1));
    });

    test('handles missing boolean fields with defaults', () {
      final json = {
        'symbol': 'NABIL',
        'holding': null,
        'recentPrices': [],
      };

      final symbolSnapshot = DashboardSymbolSnapshot.fromJson(json);

      expect(symbolSnapshot.inPortfolio, isFalse);
      expect(symbolSnapshot.inWatchlist, isFalse);
    });

    test('handles missing list fields with defaults', () {
      final json = {
        'symbol': 'NABIL',
        'inPortfolio': false,
        'inWatchlist': false,
        'holding': null,
      };

      final symbolSnapshot = DashboardSymbolSnapshot.fromJson(json);

      expect(symbolSnapshot.recentPrices, isEmpty);
    });
  });
}
