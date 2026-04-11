/// WatchlistItem Model Unit Tests
///
/// Tests for the WatchlistItem data model.

import 'package:nepse_client/src/models/watchlist_item.dart';
import 'package:test/test.dart';

void main() {
  group('WatchlistItem Model', () {
    test('creates WatchlistItem with correct values', () {
      final now = DateTime.now();
      final item = WatchlistItem(
        id: 'watchlist-id',
        symbol: 'NABIL',
        notes: 'Watch for breakout',
        createdAt: now,
        updatedAt: now,
      );

      expect(item.id, equals('watchlist-id'));
      expect(item.symbol, equals('NABIL'));
      expect(item.notes, equals('Watch for breakout'));
    });

    test('creates WatchlistItem with nullable notes', () {
      final now = DateTime.now();
      final item = WatchlistItem(
        id: 'watchlist-id',
        symbol: 'NABIL',
        notes: null,
        createdAt: now,
        updatedAt: now,
      );

      expect(item.notes, isNull);
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final item = WatchlistItem(
        id: 'watchlist-id',
        symbol: 'NABIL',
        notes: 'Watch for breakout',
        createdAt: now,
        updatedAt: now,
      );

      final json = item.toJson();

      expect(json['id'], equals('watchlist-id'));
      expect(json['symbol'], equals('NABIL'));
      expect(json['notes'], equals('Watch for breakout'));
      expect(json['createdAt'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['updatedAt'], equals('2024-01-01T12:00:00.000Z'));
    });

    test('serializes nullable notes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final item = WatchlistItem(
        id: 'watchlist-id',
        symbol: 'NABIL',
        notes: null,
        createdAt: now,
        updatedAt: now,
      );

      final json = item.toJson();

      expect(json['notes'], isNull);
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'id': 'watchlist-id',
        'symbol': 'NABIL',
        'notes': 'Watch for breakout',
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final item = WatchlistItem.fromJson(json);

      expect(item.id, equals('watchlist-id'));
      expect(item.symbol, equals('NABIL'));
      expect(item.notes, equals('Watch for breakout'));
    });

    test('deserializes from JSON with null notes', () {
      final json = {
        'id': 'watchlist-id',
        'symbol': 'NABIL',
        'notes': null,
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final item = WatchlistItem.fromJson(json);

      expect(item.notes, isNull);
    });
  });
}
