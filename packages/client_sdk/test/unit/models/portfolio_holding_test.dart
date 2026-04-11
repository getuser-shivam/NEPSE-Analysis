/// PortfolioHolding Model Unit Tests
///
/// Tests for the PortfolioHolding data model.

import 'package:nepse_client/src/models/portfolio_holding.dart';
import 'package:test/test.dart';

void main() {
  group('PortfolioHolding Model', () {
    test('creates PortfolioHolding with correct values', () {
      final now = DateTime.now();
      final holding = PortfolioHolding(
        id: 'holding-id',
        symbol: 'NABIL',
        shares: 100.0,
        buyPrice: 950.0,
        currentPrice: 1000.0,
        notes: 'Bank stock',
        lastUpdated: now,
        createdAt: now,
        updatedAt: now,
      );

      expect(holding.id, equals('holding-id'));
      expect(holding.symbol, equals('NABIL'));
      expect(holding.shares, equals(100.0));
      expect(holding.buyPrice, equals(950.0));
      expect(holding.currentPrice, equals(1000.0));
      expect(holding.notes, equals('Bank stock'));
    });

    test('creates PortfolioHolding with nullable fields', () {
      final now = DateTime.now();
      final holding = PortfolioHolding(
        id: 'holding-id',
        symbol: 'NABIL',
        shares: 100.0,
        buyPrice: 950.0,
        currentPrice: null,
        notes: null,
        lastUpdated: null,
        createdAt: now,
        updatedAt: now,
      );

      expect(holding.currentPrice, isNull);
      expect(holding.notes, isNull);
      expect(holding.lastUpdated, isNull);
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final holding = PortfolioHolding(
        id: 'holding-id',
        symbol: 'NABIL',
        shares: 100.0,
        buyPrice: 950.0,
        currentPrice: 1000.0,
        notes: 'Bank stock',
        lastUpdated: now,
        createdAt: now,
        updatedAt: now,
      );

      final json = holding.toJson();

      expect(json['id'], equals('holding-id'));
      expect(json['symbol'], equals('NABIL'));
      expect(json['shares'], equals(100.0));
      expect(json['buyPrice'], equals(950.0));
      expect(json['currentPrice'], equals(1000.0));
      expect(json['notes'], equals('Bank stock'));
      expect(json['lastUpdated'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['createdAt'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['updatedAt'], equals('2024-01-01T12:00:00.000Z'));
    });

    test('serializes nullable fields to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final holding = PortfolioHolding(
        id: 'holding-id',
        symbol: 'NABIL',
        shares: 100.0,
        buyPrice: 950.0,
        currentPrice: null,
        notes: null,
        lastUpdated: null,
        createdAt: now,
        updatedAt: now,
      );

      final json = holding.toJson();

      expect(json['currentPrice'], isNull);
      expect(json['notes'], isNull);
      expect(json['lastUpdated'], isNull);
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'id': 'holding-id',
        'symbol': 'NABIL',
        'shares': 100,
        'buyPrice': 950.0,
        'currentPrice': 1000.0,
        'notes': 'Bank stock',
        'lastUpdated': '2024-01-01T12:00:00.000Z',
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final holding = PortfolioHolding.fromJson(json);

      expect(holding.id, equals('holding-id'));
      expect(holding.symbol, equals('NABIL'));
      expect(holding.shares, equals(100.0));
      expect(holding.buyPrice, equals(950.0));
      expect(holding.currentPrice, equals(1000.0));
      expect(holding.notes, equals('Bank stock'));
    });

    test('handles JSON with numeric types correctly', () {
      final json = {
        'id': 'holding-id',
        'symbol': 'NABIL',
        'shares': 100.5,
        'buyPrice': 950.75,
        'currentPrice': 1000.25,
        'notes': null,
        'lastUpdated': null,
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final holding = PortfolioHolding.fromJson(json);

      expect(holding.shares, equals(100.5));
      expect(holding.buyPrice, equals(950.75));
      expect(holding.currentPrice, equals(1000.25));
    });
  });
}
