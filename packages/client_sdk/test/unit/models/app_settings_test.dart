/// AppSettings Model Unit Tests
///
/// Tests for the AppSettings data model.

import 'package:nepse_client/src/models/app_settings.dart';
import 'package:test/test.dart';

void main() {
  group('AppSettings Model', () {
    test('creates AppSettings with correct values', () {
      final now = DateTime.now();
      final settings = AppSettings(
        id: 'test-id',
        name: 'Test Settings',
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
      );

      expect(settings.id, equals('test-id'));
      expect(settings.name, equals('Test Settings'));
      expect(settings.autoSaveInterval, equals(300));
      expect(settings.maxDataAgeDays, equals(30));
      expect(settings.backupEnabled, isTrue);
      expect(settings.chartStyle, equals('candlestick'));
      expect(settings.defaultPeriod, equals('1D'));
      expect(settings.maxWatchlistSize, equals(50));
      expect(settings.logLevel, equals('info'));
      expect(settings.refreshInterval, equals(60));
    });

    test('serializes to JSON correctly', () {
      final now = DateTime.utc(2024, 1, 1, 12, 0, 0);
      final settings = AppSettings(
        id: 'test-id',
        name: 'Test Settings',
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
      );

      final json = settings.toJson();

      expect(json['id'], equals('test-id'));
      expect(json['name'], equals('Test Settings'));
      expect(json['autoSaveInterval'], equals(300));
      expect(json['maxDataAgeDays'], equals(30));
      expect(json['backupEnabled'], isTrue);
      expect(json['chartStyle'], equals('candlestick'));
      expect(json['defaultPeriod'], equals('1D'));
      expect(json['maxWatchlistSize'], equals(50));
      expect(json['logLevel'], equals('info'));
      expect(json['refreshInterval'], equals(60));
      expect(json['createdAt'], equals('2024-01-01T12:00:00.000Z'));
      expect(json['updatedAt'], equals('2024-01-01T12:00:00.000Z'));
    });

    test('deserializes from JSON correctly', () {
      final json = {
        'id': 'test-id',
        'name': 'Test Settings',
        'autoSaveInterval': 300,
        'maxDataAgeDays': 30,
        'backupEnabled': true,
        'chartStyle': 'candlestick',
        'defaultPeriod': '1D',
        'maxWatchlistSize': 50,
        'logLevel': 'info',
        'refreshInterval': 60,
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final settings = AppSettings.fromJson(json);

      expect(settings.id, equals('test-id'));
      expect(settings.name, equals('Test Settings'));
      expect(settings.autoSaveInterval, equals(300));
      expect(settings.maxDataAgeDays, equals(30));
      expect(settings.backupEnabled, isTrue);
      expect(settings.chartStyle, equals('candlestick'));
      expect(settings.defaultPeriod, equals('1D'));
      expect(settings.maxWatchlistSize, equals(50));
      expect(settings.logLevel, equals('info'));
      expect(settings.refreshInterval, equals(60));
    });

    test('handles JSON with numeric types correctly', () {
      final json = {
        'id': 'test-id',
        'name': 'Test Settings',
        'autoSaveInterval': 300.0,
        'maxDataAgeDays': 30.5,
        'backupEnabled': true,
        'chartStyle': 'candlestick',
        'defaultPeriod': '1D',
        'maxWatchlistSize': 50.0,
        'logLevel': 'info',
        'refreshInterval': 60.0,
        'createdAt': '2024-01-01T12:00:00.000Z',
        'updatedAt': '2024-01-01T12:00:00.000Z',
      };

      final settings = AppSettings.fromJson(json);

      expect(settings.autoSaveInterval, equals(300));
      expect(settings.maxDataAgeDays, equals(30));
      expect(settings.maxWatchlistSize, equals(50));
      expect(settings.refreshInterval, equals(60));
    });
  });
}
