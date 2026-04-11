/// Dashboard Provider Unit Tests
///
/// Tests for the dashboard Riverpod provider.

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nepse_app/providers/dashboard_provider.dart';
import 'package:nepse_client/nepse_client.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:nepse_app/providers/api_client_provider.dart';

import '../../../../packages/client_sdk/test/unit/nepse_api_client_test.mocks.dart';

@GenerateMocks([NepseApiClient])
void main() {
  group('dashboardProvider', () {
    late ProviderContainer container;
    late MockNepseApiClient mockApiClient;

    setUp(() {
      mockApiClient = MockNepseApiClient();
      container = ProviderContainer(
        overrides: [
          apiClientProvider.overrideWithValue(mockApiClient),
        ],
      );
    });

    tearDown(() {
      container.dispose();
    });

    test('provides DashboardSnapshot from API client', () async {
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

      when(mockApiClient.getDashboard()).thenAnswer((_) async => snapshot);

      final result = await container.read(dashboardProvider.future);

      expect(result, equals(snapshot));
      verify(mockApiClient.getDashboard()).called(1);
    });

    test('throws error when API client fails', () async {
      when(mockApiClient.getDashboard()).thenThrow(Exception('API error'));

      expect(
        () => container.read(dashboardProvider.future),
        throwsA(isA<Exception>()),
      );
    });
  });
}
