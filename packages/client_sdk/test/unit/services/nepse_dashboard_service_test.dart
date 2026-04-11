/// NepseDashboardService Unit Tests
///
/// Tests for the NEPSE dashboard service.

import 'package:nepse_client/src/models/dashboard_snapshot.dart';
import 'package:nepse_client/src/nepse_api_client.dart';
import 'package:nepse_client/src/services/nepse_dashboard_service.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test/unit/nepse_api_client_test.mocks.dart';

@GenerateMocks([NepseApiClient])
void main() {
  group('NepseDashboardService', () {
    late NepseDashboardService service;
    late MockNepseApiClient mockApiClient;

    setUp(() {
      mockApiClient = MockNepseApiClient();
      service = NepseDashboardService(mockApiClient);
    });

    test('loadSnapshot calls API client with default limit', () async {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final snapshot = DashboardSnapshot(
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

      when(mockApiClient.getDashboard(limit: 5))
          .thenAnswer((_) async => snapshot);

      final result = await service.loadSnapshot();

      expect(result, equals(snapshot));
      verify(mockApiClient.getDashboard(limit: 5)).called(1);
    });

    test('loadSnapshot calls API client with custom limit', () async {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final snapshot = DashboardSnapshot(
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

      when(mockApiClient.getDashboard(limit: 10))
          .thenAnswer((_) async => snapshot);

      final result = await service.loadSnapshot(limit: 10);

      expect(result, equals(snapshot));
      verify(mockApiClient.getDashboard(limit: 10)).called(1);
    });
  });
}
