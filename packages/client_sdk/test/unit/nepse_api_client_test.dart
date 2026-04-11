/// NepseApiClient Unit Tests
///
/// Tests for the NEPSE API client.

import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:nepse_client/src/nepse_api_client.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import 'nepse_api_client_test.mocks.dart';

@GenerateMocks([http.Client])
void main() {
  group('NepseApiClient', () {
    late NepseApiClient client;
    late MockClient mockHttpClient;

    setUp(() {
      mockHttpClient = MockClient();
      client = NepseApiClient(
        baseUrl: 'https://api.example.com',
        httpClient: mockHttpClient,
      );
    });

    tearDown(() {
      client.close();
    });

    group('Authentication', () {
      test('login stores auth token on successful response', () async {
        final response = {
          'accessToken': 'test-token',
          'user': {'id': '1', 'email': 'test@example.com'}
        };

        when(mockHttpClient.post(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final result = await client.login('test@example.com', 'password');

        expect(result['accessToken'], equals('test-token'));
        verify(mockHttpClient.post(
          argThat(contains('/api/auth/login')),
          headers: argThat(containsPair('Content-Type', 'application/json')),
          body: argThat(contains('test@example.com')),
        )).called(1);
      });

      test('login does not throw on missing accessToken', () async {
        final response = {'user': {'id': '1', 'email': 'test@example.com'}};

        when(mockHttpClient.post(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final result = await client.login('test@example.com', 'password');

        expect(result, isNotNull);
        expect(result.containsKey('accessToken'), isFalse);
      });

      test('setAuthToken updates the authorization header', () {
        client.setAuthToken('new-token');

        final headers = client._headers;
        expect(headers['Authorization'], equals('Bearer new-token'));
      });
    });

    group('Health Check', () {
      test('getHealth returns health data', () async {
        final response = {'status': 'healthy', 'version': '1.0.0'};

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final result = await client.getHealth();

        expect(result['status'], equals('healthy'));
        verify(mockHttpClient.get(
          argThat(contains('/health')),
          headers: argThat(containsPair('Accept', 'application/json')),
        )).called(1);
      });
    });

    group('Settings', () {
      test('getSettings returns AppSettings', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'id': 'settings-id',
            'name': 'Test Settings',
            'autoSaveInterval': 300,
            'maxDataAgeDays': 30,
            'backupEnabled': true,
            'chartStyle': 'candlestick',
            'defaultPeriod': '1D',
            'maxWatchlistSize': 50,
            'logLevel': 'info',
            'refreshInterval': 60,
            'createdAt': now.toIso8601String(),
            'updatedAt': now.toIso8601String(),
          }
        };

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final settings = await client.getSettings();

        expect(settings.name, equals('Test Settings'));
        expect(settings.autoSaveInterval, equals(300));
      });

      test('updateSettings returns updated AppSettings', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'id': 'settings-id',
            'name': 'Updated Settings',
            'autoSaveInterval': 600,
            'maxDataAgeDays': 60,
            'backupEnabled': false,
            'chartStyle': 'line',
            'defaultPeriod': '1W',
            'maxWatchlistSize': 100,
            'logLevel': 'debug',
            'refreshInterval': 120,
            'createdAt': now.toIso8601String(),
            'updatedAt': now.toIso8601String(),
          }
        };

        when(mockHttpClient.patch(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final settings = await client.updateSettings({'autoSaveInterval': 600});

        expect(settings.autoSaveInterval, equals(600));
      });
    });

    group('Dashboard', () {
      test('getDashboard returns DashboardSnapshot', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'generatedAt': now.toIso8601String(),
            'settings': {
              'id': 'settings-id',
              'name': 'Test',
              'autoSaveInterval': 300,
              'maxDataAgeDays': 30,
              'backupEnabled': true,
              'chartStyle': 'candlestick',
              'defaultPeriod': '1D',
              'maxWatchlistSize': 50,
              'logLevel': 'info',
              'refreshInterval': 60,
              'createdAt': now.toIso8601String(),
              'updatedAt': now.toIso8601String(),
            },
            'portfolio': {
              'holdings': [],
              'summary': {
                'totalInvestment': 0,
                'totalValue': 0,
                'totalGainLoss': 0,
                'totalReturnPct': 0,
                'numStocks': 0,
              }
            },
            'watchlist': [],
            'symbols': []
          }
        };

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final dashboard = await client.getDashboard(limit: 10);

        expect(dashboard.generatedAt, equals(now));
        verify(mockHttpClient.get(
          argThat(allOf([
            contains('/api/dashboard'),
            contains('limit=10'),
          ])),
          headers: anyNamed('headers'),
        )).called(1);
      });
    });

    group('Portfolio', () {
      test('getPortfolio returns PortfolioOverview', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'holdings': [],
            'summary': {
              'totalInvestment': 0,
              'totalValue': 0,
              'totalGainLoss': 0,
              'totalReturnPct': 0,
              'numStocks': 0,
            }
          }
        };

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final portfolio = await client.getPortfolio();

        expect(portfolio.holdings, isEmpty);
      });

      test('upsertHolding returns PortfolioHolding', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'id': 'holding-id',
            'symbol': 'NABIL',
            'shares': 100,
            'buyPrice': 950.0,
            'currentPrice': 1000.0,
            'notes': null,
            'lastUpdated': null,
            'createdAt': now.toIso8601String(),
            'updatedAt': now.toIso8601String(),
          }
        };

        when(mockHttpClient.post(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final holding = await client.upsertHolding(
          symbol: 'nabil',
          shares: 100,
          buyPrice: 950.0,
          currentPrice: 1000.0,
        );

        expect(holding.symbol, equals('NABIL'));
        expect(holding.shares, equals(100.0));
      });

      test('deleteHolding sends DELETE request', () async {
        when(mockHttpClient.delete(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('', 204));

        await client.deleteHolding('NABIL');

        verify(mockHttpClient.delete(
          argThat(contains('/api/portfolio/NABIL')),
          headers: anyNamed('headers'),
        )).called(1);
      });
    });

    group('Watchlist', () {
      test('getWatchlist returns list of WatchlistItem', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': [
            {
              'id': 'watchlist-id',
              'symbol': 'NABIL',
              'notes': 'Watch for breakout',
              'createdAt': now.toIso8601String(),
              'updatedAt': now.toIso8601String(),
            }
          ]
        };

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final watchlist = await client.getWatchlist();

        expect(watchlist.length, equals(1));
        expect(watchlist.first.symbol, equals('NABIL'));
      });

      test('addWatchlistItem returns WatchlistItem', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final response = {
          'data': {
            'id': 'watchlist-id',
            'symbol': 'NABIL',
            'notes': 'Watch for breakout',
            'createdAt': now.toIso8601String(),
            'updatedAt': now.toIso8601String(),
          }
        };

        when(mockHttpClient.post(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final item = await client.addWatchlistItem('nabil', notes: 'Watch for breakout');

        expect(item.symbol, equals('NABIL'));
      });

      test('removeWatchlistItem sends DELETE request', () async {
        when(mockHttpClient.delete(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('', 204));

        await client.removeWatchlistItem('NABIL');

        verify(mockHttpClient.delete(
          argThat(contains('/api/watchlist/NABIL')),
          headers: anyNamed('headers'),
        )).called(1);
      });
    });

    group('Price Snapshots', () {
      test('getPriceSnapshots returns list of PriceSnapshot', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final tradeDate = DateTime.utc(2024, 1, 1);
        final response = {
          'data': [
            {
              'id': 'price-id',
              'symbol': 'NABIL',
              'tradeDate': tradeDate.toIso8601String(),
              'open': 950.0,
              'high': 980.0,
              'low': 940.0,
              'close': 970.0,
              'volume': 1000000,
              'source': 'NEPSE',
              'createdAt': now.toIso8601String(),
              'updatedAt': now.toIso8601String(),
            }
          ]
        };

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final prices = await client.getPriceSnapshots('NABIL', limit: 10);

        expect(prices.length, equals(1));
        expect(prices.first.symbol, equals('NABIL'));
      });

      test('upsertPriceSnapshot returns PriceSnapshot', () async {
        final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
        final tradeDate = DateTime.utc(2024, 1, 1);
        final response = {
          'data': {
            'id': 'price-id',
            'symbol': 'NABIL',
            'tradeDate': tradeDate.toIso8601String(),
            'open': 950.0,
            'high': 980.0,
            'low': 940.0,
            'close': 970.0,
            'volume': 1000000,
            'source': 'manual',
            'createdAt': now.toIso8601String(),
            'updatedAt': now.toIso8601String(),
          }
        };

        when(mockHttpClient.post(
          any,
          headers: anyNamed('headers'),
          body: anyNamed('body'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode(response),
              200,
            ));

        final price = await client.upsertPriceSnapshot(
          symbol: 'nabil',
          tradeDate: tradeDate,
          open: 950.0,
          high: 980.0,
          low: 940.0,
          close: 970.0,
          volume: 1000000,
        );

        expect(price.symbol, equals('NABIL'));
      });
    });

    group('Error Handling', () {
      test('throws NepseApiException on non-200 response', () async {
        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response(
              jsonEncode({'error': 'Not found'}),
              404,
            ));

        expect(
          () => client.getHealth(),
          throwsA(isA<NepseApiException>()
              .having((e) => e.statusCode, 'statusCode', 404)
              .having((e) => e.message, 'message', 'Not found')),
        );
      });

      test('throws NepseApiException with default message on empty body', () async {
        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('', 500));

        expect(
          () => client.getHealth(),
          throwsA(isA<NepseApiException>()
              .having((e) => e.statusCode, 'statusCode', 500)
              .having((e) => e.message, 'message', 'Request failed')),
        );
      });

      test('throws NepseApiException on invalid JSON response', () async {
        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('not json', 200));

        expect(
          () => client.getHealth(),
          throwsA(isA<NepseApiException>()),
        );
      });
    });

    group('URL Building', () {
      test('normalizes base URL with trailing slash', () {
        final clientWithSlash = NepseApiClient(
          baseUrl: 'https://api.example.com/',
          httpClient: mockHttpClient,
        );

        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('{"status":"ok"}', 200));

        clientWithSlash.getHealth();

        verify(mockHttpClient.get(
          argThat(isNot(contains('//health'))),
          headers: anyNamed('headers'),
        )).called(1);

        clientWithSlash.close();
      });

      test('includes query parameters in URL', () async {
        when(mockHttpClient.get(
          any,
          headers: anyNamed('headers'),
        )).thenAnswer((_) async => http.Response('[]', 200));

        await client.getPriceSnapshots('NABIL', limit: 20);

        verify(mockHttpClient.get(
          argThat(contains('limit=20')),
          headers: anyNamed('headers'),
        )).called(1);
      });
    });
  });
}
