/// Stock Model Unit Tests
///
/// Tests for the Stock data model.

import 'package:nepse_app/src/models/stock.dart';
import 'package:test/test.dart';

void main() {
  group('Stock Model', () {
    test('creates stock with correct values', () {
      final stock = Stock(
        symbol: 'NABIL',
        name: 'Nabil Bank Ltd.',
        currentPrice: 1000.0,
        openPrice: 990.0,
        highPrice: 1010.0,
        lowPrice: 985.0,
        previousClose: 995.0,
        volume: 10000,
      );

      expect(stock.symbol, equals('NABIL'));
      expect(stock.currentPrice, equals(1000.0));
    });

    test('calculates price change correctly', () {
      final stock = Stock(
        symbol: 'NABIL',
        name: 'Nabil Bank Ltd.',
        currentPrice: 1000.0,
        openPrice: 990.0,
        highPrice: 1010.0,
        lowPrice: 985.0,
        previousClose: 995.0,
        volume: 10000,
      );

      expect(stock.priceChange, equals(5.0));
    });

    test('calculates percentage change correctly', () {
      final stock = Stock(
        symbol: 'NABIL',
        name: 'Nabil Bank Ltd.',
        currentPrice: 1000.0,
        openPrice: 990.0,
        highPrice: 1010.0,
        lowPrice: 985.0,
        previousClose: 995.0,
        volume: 10000,
      );

      expect(stock.percentChange, closeTo(0.5025, 0.001));
    });
  });
}
