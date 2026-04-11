/// PortfolioOverview Model Unit Tests
///
/// Tests for the PortfolioOverview data model.

import 'package:nepse_client/src/models/portfolio_holding.dart';
import 'package:nepse_client/src/models/portfolio_overview.dart';
import 'package:nepse_client/src/models/portfolio_summary.dart';
import 'package:test/test.dart';

void main() {
  group('PortfolioOverview Model', () {
    test('creates PortfolioOverview with correct values', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
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

      final summary = PortfolioSummary(
        totalInvestment: 95000.0,
        totalValue: 100000.0,
        totalGainLoss: 5000.0,
        totalReturnPct: 5.26,
        numStocks: 1,
      );

      final overview = PortfolioOverview(
        holdings: [holding],
        summary: summary,
      );

      expect(overview.holdings.length, equals(1));
      expect(overview.holdings.first.symbol, equals('NABIL'));
      expect(overview.summary.totalInvestment, equals(95000.0));
      expect(overview.summary.totalValue, equals(100000.0));
    });

    test('creates PortfolioOverview with empty holdings', () {
      final summary = PortfolioSummary(
        totalInvestment: 0.0,
        totalValue: 0.0,
        totalGainLoss: 0.0,
        totalReturnPct: 0.0,
        numStocks: 0,
      );

      final overview = PortfolioOverview(
        holdings: [],
        summary: summary,
      );

      expect(overview.holdings, isEmpty);
      expect(overview.summary.numStocks, equals(0));
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
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

      final summary = PortfolioSummary(
        totalInvestment: 95000.0,
        totalValue: 100000.0,
        totalGainLoss: 5000.0,
        totalReturnPct: 5.26,
        numStocks: 1,
      );

      final overview = PortfolioOverview(
        holdings: [holding],
        summary: summary,
      );

      final json = overview.toJson();

      expect(json['holdings'], isNotEmpty);
      expect(json['holdings'].length, equals(1));
      expect(json['summary'], isNotEmpty);
      expect(json['summary']['totalInvestment'], equals(95000.0));
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'holdings': [
          {
            'id': 'holding-id',
            'symbol': 'NABIL',
            'shares': 100,
            'buyPrice': 950.0,
            'currentPrice': 1000.0,
            'notes': null,
            'lastUpdated': null,
            'createdAt': '2024-01-01T12:00:00.000Z',
            'updatedAt': '2024-01-01T12:00:00.000Z',
          }
        ],
        'summary': {
          'totalInvestment': 95000,
          'totalValue': 100000,
          'totalGainLoss': 5000,
          'totalReturnPct': 5.26,
          'numStocks': 1,
        }
      };

      final overview = PortfolioOverview.fromJson(json);

      expect(overview.holdings.length, equals(1));
      expect(overview.holdings.first.symbol, equals('NABIL'));
      expect(overview.summary.totalInvestment, equals(95000.0));
    });

    test('handles empty holdings in JSON', () {
      final json = {
        'holdings': [],
        'summary': {
          'totalInvestment': 0,
          'totalValue': 0,
          'totalGainLoss': 0,
          'totalReturnPct': 0,
          'numStocks': 0,
        }
      };

      final overview = PortfolioOverview.fromJson(json);

      expect(overview.holdings, isEmpty);
    });
  });
}
