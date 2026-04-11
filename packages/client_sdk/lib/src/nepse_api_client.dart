import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models/app_settings.dart';
import 'models/dashboard_snapshot.dart';
import 'models/portfolio_holding.dart';
import 'models/portfolio_overview.dart';
import 'models/price_snapshot.dart';
import 'models/watchlist_item.dart';

class NepseApiException implements Exception {
  NepseApiException({
    required this.statusCode,
    required this.message,
    this.details,
  });

  final int statusCode;
  final String message;
  final Object? details;

  @override
  String toString() {
    return 'NepseApiException(statusCode: $statusCode, message: $message, details: $details)';
  }
}

class NepseApiClient {
  NepseApiClient({required this.baseUrl, http.Client? httpClient})
    : _httpClient = httpClient ?? http.Client();

  final String baseUrl;
  final http.Client _httpClient;

  Future<Map<String, dynamic>> getHealth() async {
    return _requestJson(method: 'GET', path: '/health');
  }

  Future<AppSettings> getSettings() async {
    final json = await _requestJson(method: 'GET', path: '/api/settings');

    return AppSettings.fromJson(Map<String, dynamic>.from(json['data'] as Map));
  }

  Future<AppSettings> updateSettings(Map<String, dynamic> changes) async {
    final json = await _requestJson(
      method: 'PATCH',
      path: '/api/settings',
      body: changes,
    );

    return AppSettings.fromJson(Map<String, dynamic>.from(json['data'] as Map));
  }

  Future<DashboardSnapshot> getDashboard({int limit = 5}) async {
    final json = await _requestJson(
      method: 'GET',
      path: '/api/dashboard',
      queryParameters: {'limit': '$limit'},
    );

    return DashboardSnapshot.fromJson(
      Map<String, dynamic>.from(json['data'] as Map),
    );
  }

  Future<PortfolioOverview> getPortfolio() async {
    final json = await _requestJson(method: 'GET', path: '/api/portfolio');

    return PortfolioOverview.fromJson(
      Map<String, dynamic>.from(json['data'] as Map),
    );
  }

  Future<PortfolioHolding> upsertHolding({
    required String symbol,
    required double shares,
    required double buyPrice,
    double? currentPrice,
    String? notes,
    DateTime? lastUpdated,
  }) async {
    final json = await _requestJson(
      method: 'POST',
      path: '/api/portfolio',
      body: {
        'symbol': symbol.toUpperCase(),
        'shares': shares,
        'buyPrice': buyPrice,
        'currentPrice': currentPrice,
        'notes': notes,
        'lastUpdated': lastUpdated?.toIso8601String(),
      },
    );

    return PortfolioHolding.fromJson(
      Map<String, dynamic>.from(json['data'] as Map),
    );
  }

  Future<void> deleteHolding(String symbol) async {
    await _requestVoid(
      method: 'DELETE',
      path: '/api/portfolio/${Uri.encodeComponent(symbol.toUpperCase())}',
    );
  }

  Future<List<WatchlistItem>> getWatchlist() async {
    final json = await _requestJson(method: 'GET', path: '/api/watchlist');

    final rawItems = json['data'] as List<dynamic>? ?? const [];
    return rawItems
        .map(
          (item) =>
              WatchlistItem.fromJson(Map<String, dynamic>.from(item as Map)),
        )
        .toList(growable: false);
  }

  Future<WatchlistItem> addWatchlistItem(String symbol, {String? notes}) async {
    final json = await _requestJson(
      method: 'POST',
      path: '/api/watchlist',
      body: {'symbol': symbol.toUpperCase(), 'notes': notes},
    );

    return WatchlistItem.fromJson(
      Map<String, dynamic>.from(json['data'] as Map),
    );
  }

  Future<void> removeWatchlistItem(String symbol) async {
    await _requestVoid(
      method: 'DELETE',
      path: '/api/watchlist/${Uri.encodeComponent(symbol.toUpperCase())}',
    );
  }

  Future<List<PriceSnapshot>> getPriceSnapshots(
    String symbol, {
    int limit = 30,
  }) async {
    final json = await _requestJson(
      method: 'GET',
      path: '/api/prices/${Uri.encodeComponent(symbol.toUpperCase())}',
      queryParameters: {'limit': '$limit'},
    );

    final rawItems = json['data'] as List<dynamic>? ?? const [];
    return rawItems
        .map(
          (item) =>
              PriceSnapshot.fromJson(Map<String, dynamic>.from(item as Map)),
        )
        .toList(growable: false);
  }

  Future<PriceSnapshot> upsertPriceSnapshot({
    required String symbol,
    required DateTime tradeDate,
    required double open,
    required double high,
    required double low,
    required double close,
    double volume = 0,
    String source = 'manual',
  }) async {
    final json = await _requestJson(
      method: 'POST',
      path: '/api/prices',
      body: {
        'symbol': symbol.toUpperCase(),
        'tradeDate': tradeDate.toIso8601String(),
        'open': open,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'source': source,
      },
    );

    return PriceSnapshot.fromJson(
      Map<String, dynamic>.from(json['data'] as Map),
    );
  }

  void close() {
    _httpClient.close();
  }

  Future<Map<String, dynamic>> _requestJson({
    required String method,
    required String path,
    Map<String, String>? queryParameters,
    Map<String, dynamic>? body,
  }) async {
    final response = await _send(
      method: method,
      path: path,
      queryParameters: queryParameters,
      body: body,
    );

    if (response.body.isEmpty) {
      return const <String, dynamic>{};
    }

    final decoded = jsonDecode(response.body);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }

    if (decoded is Map) {
      return Map<String, dynamic>.from(decoded);
    }

    throw NepseApiException(
      statusCode: response.statusCode,
      message: 'Expected a JSON object response',
      details: decoded,
    );
  }

  Future<void> _requestVoid({
    required String method,
    required String path,
  }) async {
    await _send(method: method, path: path);
  }

  Future<http.Response> _send({
    required String method,
    required String path,
    Map<String, String>? queryParameters,
    Map<String, dynamic>? body,
  }) async {
    final uri = _buildUri(path, queryParameters);

    switch (method) {
      case 'GET':
        return _handleResponse(await _httpClient.get(uri, headers: _headers));
      case 'POST':
        return _handleResponse(
          await _httpClient.post(
            uri,
            headers: _headers,
            body: jsonEncode(body ?? const <String, dynamic>{}),
          ),
        );
      case 'PATCH':
        return _handleResponse(
          await _httpClient.patch(
            uri,
            headers: _headers,
            body: jsonEncode(body ?? const <String, dynamic>{}),
          ),
        );
      case 'DELETE':
        return _handleResponse(
          await _httpClient.delete(uri, headers: _headers),
        );
      default:
        throw ArgumentError.value(method, 'method', 'Unsupported HTTP method');
    }
  }

  http.Response _handleResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return response;
    }

    Object? details;
    String message = 'Request failed';
    if (response.body.isNotEmpty) {
      final decoded = jsonDecode(response.body);
      details = decoded;

      if (decoded is Map<String, dynamic>) {
        message = decoded['error'] as String? ?? message;
      } else if (decoded is Map) {
        final mapped = Map<String, dynamic>.from(decoded);
        message = mapped['error'] as String? ?? message;
        details = mapped;
      }
    }

    throw NepseApiException(
      statusCode: response.statusCode,
      message: message,
      details: details,
    );
  }

  Uri _buildUri(String path, Map<String, String>? queryParameters) {
    final normalizedBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    final uri = Uri.parse('$normalizedBaseUrl$path');
    return queryParameters == null
        ? uri
        : uri.replace(queryParameters: queryParameters);
  }

  Map<String, String> get _headers => const {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  };
}
