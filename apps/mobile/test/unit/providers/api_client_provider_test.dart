/// API Client Provider Unit Tests
///
/// Tests for the API client Riverpod providers.

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:nepse_app/providers/api_client_provider.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:nepse_client/nepse_client.dart';

void main() {
  group('secureStorageProvider', () {
    test('provides FlutterSecureStorage instance', () {
      final container = ProviderContainer();
      final storage = container.read(secureStorageProvider);

      expect(storage, isA<FlutterSecureStorage>());
      container.dispose();
    });
  });

  group('apiClientProvider', () {
    test('provides NepseApiClient instance', () {
      final container = ProviderContainer();
      final client = container.read(apiClientProvider);

      expect(client, isA<NepseApiClient>());
      expect(client.baseUrl, equals('http://localhost:3000'));
      container.dispose();
    });

    test('provides same instance on multiple reads', () {
      final container = ProviderContainer();
      final client1 = container.read(apiClientProvider);
      final client2 = container.read(apiClientProvider);

      expect(identical(client1, client2), isTrue);
      container.dispose();
    });
  });
}
