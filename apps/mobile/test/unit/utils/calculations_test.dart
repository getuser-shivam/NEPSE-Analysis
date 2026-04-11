/// Technical Indicators Unit Tests
///
/// Tests for technical indicator calculations.

import 'package:nepse_app/src/utils/calculations.dart';
import 'package:test/test.dart';

void main() {
  group('SMA Calculations', () {
    test('calculates SMA correctly', () {
      final prices = [100, 102, 101, 105, 108, 107, 110, 112, 111, 115];
      final sma = calculateSMA(prices, 5);

      expect(sma, isNotNull);
      expect(sma.length, equals(6));
      expect(sma.last, closeTo(111, 0.1));
    });
  });

  group('RSI Calculations', () {
    test('calculates RSI correctly', () {
      final prices = List.generate(20, (i) => 100 + i + (i % 3));
      final rsi = calculateRSI(prices, 14);

      expect(rsi, isNotNull);
      expect(rsi, greaterThanOrEqualTo(0));
      expect(rsi, lessThanOrEqualTo(100));
    });
  });

  group('MACD Calculations', () {
    test('calculates MACD correctly', () {
      final prices = List.generate(30, (i) => 100 + i + (i % 3));
      final macd = calculateMACD(prices);

      expect(macd, isNotNull);
      expect(macd.line, isNotEmpty);
      expect(macd.signal, isNotEmpty);
    });
  });
}
