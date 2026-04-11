/// PortfolioSummaryCard Widget Tests
///
/// Tests for the PortfolioSummaryCard widget.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nepse_client/nepse_client.dart';
import 'package:nepse_app/widgets/dashboard/portfolio_summary_card.dart';

void main() {
  group('PortfolioSummaryCard Widget', () {
    late DashboardSnapshot snapshot;

    setUp(() {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      snapshot = DashboardSnapshot(
        generatedAt: now,
        settings: const AppSettings(
          id: 'settings-id',
          name: 'Test',
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
        ),
        portfolio: const PortfolioOverview(
          holdings: [],
          summary: PortfolioSummary(
            totalInvestment: 100000,
            totalValue: 120000,
            totalGainLoss: 20000,
            totalReturnPct: 20.0,
            numStocks: 5,
          ),
        ),
        watchlist: const [],
        symbols: const [],
      );
    });

    testWidgets('renders portfolio summary title', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('Portfolio summary'), findsOneWidget);
    });

    testWidgets('displays number of holdings', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('5 holdings tracked with current price snapshots and watchlist context.'), findsOneWidget);
    });

    testWidgets('displays investment value', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('Rs 100,000.00'), findsOneWidget);
    });

    testWidgets('displays current value', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('Rs 120,000.00'), findsOneWidget);
    });

    testWidgets('displays gain/loss for positive gain', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('Rs 20,000.00'), findsOneWidget);
    });

    testWidgets('displays return percentage', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('20.00%'), findsOneWidget);
    });

    testWidgets('displays all stat tiles', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: snapshot),
          ),
        ),
      );

      expect(find.text('Investment'), findsOneWidget);
      expect(find.text('Value'), findsOneWidget);
      expect(find.text('Gain / Loss'), findsOneWidget);
      expect(find.text('Return'), findsOneWidget);
    });

    testWidgets('handles zero holdings', (WidgetTester tester) async {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final emptySnapshot = DashboardSnapshot(
        generatedAt: now,
        settings: const AppSettings(
          id: 'settings-id',
          name: 'Test',
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
        ),
        portfolio: const PortfolioOverview(
          holdings: [],
          summary: PortfolioSummary(
            totalInvestment: 0,
            totalValue: 0,
            totalGainLoss: 0,
            totalReturnPct: 0,
            numStocks: 0,
          ),
        ),
        watchlist: const [],
        symbols: const [],
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: PortfolioSummaryCard(snapshot: emptySnapshot),
          ),
        ),
      );

      expect(find.text('0 holdings tracked with current price snapshots and watchlist context.'), findsOneWidget);
    });
  });
}
