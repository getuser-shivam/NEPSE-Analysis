/// Portfolio Provider Unit Tests
///
/// Tests for the portfolio Riverpod provider.

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nepse_app/providers/portfolio_provider.dart';
import 'package:nepse_client/nepse_client.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:nepse_app/providers/api_client_provider.dart';

import '../../../../packages/client_sdk/test/unit/nepse_api_client_test.mocks.dart';

@GenerateMocks([NepseApiClient])
void main() {
  group('portfolioProvider', () {
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

    test('provides PortfolioOverview from API client', () async {
      final portfolio = PortfolioOverview(
        holdings: const [],
        summary: const PortfolioSummary(
          totalInvestment: 100000,
          totalValue: 120000,
          totalGainLoss: 20000,
          totalReturnPct: 20.0,
          numStocks: 5,
        ),
      );

      when(mockApiClient.getPortfolio()).thenAnswer((_) async => portfolio);

      final result = await container.read(portfolioProvider.future);

      expect(result, equals(portfolio));
      verify(mockApiClient.getPortfolio()).called(1);
    });

    test('throws error when API client fails', () async {
      when(mockApiClient.getPortfolio()).thenThrow(Exception('API error'));

      expect(
        () => container.read(portfolioProvider.future),
        throwsA(isA<Exception>()),
      );
    });
  });
}
