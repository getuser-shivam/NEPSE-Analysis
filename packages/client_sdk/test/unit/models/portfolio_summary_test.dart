/// PortfolioSummary Model Unit Tests
///
/// Tests for the PortfolioSummary data model.

import 'package:nepse_client/src/models/portfolio_summary.dart';
import 'package:test/test.dart';

void main() {
  group('PortfolioSummary Model', () {
    test('creates PortfolioSummary with correct values', () {
      final summary = PortfolioSummary(
        totalInvestment: 100000.0,
        totalValue: 120000.0,
        totalGainLoss: 20000.0,
        totalReturnPct: 20.0,
        numStocks: 5,
      );

      expect(summary.totalInvestment, equals(100000.0));
      expect(summary.totalValue, equals(120000.0));
      expect(summary.totalGainLoss, equals(20000.0));
      expect(summary.totalReturnPct, equals(20.0));
      expect(summary.numStocks, equals(5));
    });

    test('creates PortfolioSummary with negative values', () {
      final summary = PortfolioSummary(
        totalInvestment: 100000.0,
        totalValue: 80000.0,
        totalGainLoss: -20000.0,
        totalReturnPct: -20.0,
        numStocks: 5,
      );

      expect(summary.totalGainLoss, equals(-20000.0));
      expect(summary.totalReturnPct, equals(-20.0));
    });

    test('serializes to JSON correctly', () {
      final summary = PortfolioSummary(
        totalInvestment: 100000.0,
        totalValue: 120000.0,
        totalGainLoss: 20000.0,
        totalReturnPct: 20.0,
        numStocks: 5,
      );

      final json = summary.toJson();

      expect(json['totalInvestment'], equals(100000.0));
      expect(json['totalValue'], equals(120000.0));
      expect(json['totalGainLoss'], equals(20000.0));
      expect(json['totalReturnPct'], equals(20.0));
      expect(json['numStocks'], equals(5));
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'totalInvestment': 100000,
        'totalValue': 120000,
        'totalGainLoss': 20000,
        'totalReturnPct': 20.0,
        'numStocks': 5,
      };

      final summary = PortfolioSummary.fromJson(json);

      expect(summary.totalInvestment, equals(100000.0));
      expect(summary.totalValue, equals(120000.0));
      expect(summary.totalGainLoss, equals(20000.0));
      expect(summary.totalReturnPct, equals(20.0));
      expect(summary.numStocks, equals(5));
    });

    test('handles JSON with numeric types correctly', () {
      final json = {
        'totalInvestment': 100000.5,
        'totalValue': 120000.75,
        'totalGainLoss': 20000.25,
        'totalReturnPct': 20.125,
        'numStocks': 5.0,
      };

      final summary = PortfolioSummary.fromJson(json);

      expect(summary.totalInvestment, equals(100000.5));
      expect(summary.totalValue, equals(120000.75));
      expect(summary.totalGainLoss, equals(20000.25));
      expect(summary.totalReturnPct, equals(20.125));
      expect(summary.numStocks, equals(5));
    });
  });
}
