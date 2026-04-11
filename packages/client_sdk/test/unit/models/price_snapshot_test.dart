/// PriceSnapshot Model Unit Tests
///
/// Tests for the PriceSnapshot data model.

import 'package:nepse_client/src/models/price_snapshot.dart';
import 'package:test/test.dart';

void main() {
  group('PriceSnapshot Model', () {
    test('creates PriceSnapshot with correct values', () {
      final tradeDate = DateTime.utc(2024, 1, 1);
      final now = DateTime.now();
      final snapshot = PriceSnapshot(
        id: 'snapshot-id',
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

      expect(snapshot.id, equals('snapshot-id'));
      expect(snapshot.symbol, equals('NABIL'));
      expect(snapshot.tradeDate, equals(tradeDate));
      expect(snapshot.open, equals(950.0));
      expect(snapshot.high, equals(980.0));
      expect(snapshot.low, equals(940.0));
      expect(snapshot.close, equals(970.0));
      expect(snapshot.volume, equals(1000000.0));
      expect(snapshot.source, equals('NEPSE'));
    });

    test('serializes to JSON correctly', () {
      final tradeDate = DateTime.utc(2024, 1, 1);
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final snapshot = PriceSnapshot(
        id: 'snapshot-id',
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

      final json = snapshot.toJson();

      expect(json['id'], equals('snapshot-id'));
      expect(json['symbol'], equals('NABIL'));
      expect(json['tradeDate'], equals('2024-01-01T00:00:00.000Z'));
      expect(json['open'], equals(950.0));
      expect(json['high'], equals(980.0));
      expect(json['low'], equals(940.0));
      expect(json['close'], equals(970.0));
      expect(json['volume'], equals(1000000.0));
      expect(json['source'], equals('NEPSE'));
      expect(json['createdAt'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['updatedAt'], equals('2024-01-01T12:00:00.000Z'));
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'id': 'snapshot-id',
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
      };

      final snapshot = PriceSnapshot.fromJson(json);

      expect(snapshot.id, equals('snapshot-id'));
      expect(snapshot.symbol, equals('NABIL'));
      expect(snapshot.open, equals(950.0));
      expect(snapshot.high, equals(980.0));
      expect(snapshot.low, equals(940.0));
      expect(snapshot.close, equals(970.0));
      expect(snapshot.volume, equals(1000000.0));
      expect(snapshot.source, equals('NEPSE'));
    });

    test('handles JSON with numeric types correctly', () {
      final json = {
        'id': 'snapshot-id',
        'symbol': 'NABIL',
        'tradeDate': '2024-01-01T00:00:00.000Z',
        'open': 950.5,
        'high': 980.75,
        'low': 940.25,
        'close': 970.125,
        'volume': 1000000.5,
        'source': 'NEPSE',
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final snapshot = PriceSnapshot.fromJson(json);

      expect(snapshot.open, equals(950.5));
      expect(snapshot.high, equals(980.75));
      expect(snapshot.low, equals(940.25));
      expect(snapshot.close, equals(970.125));
      expect(snapshot.volume, equals(1000000.5));
    });
  });
}
