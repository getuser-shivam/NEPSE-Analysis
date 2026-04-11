import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/watchlist_models.dart';
import 'api_settings_service.dart';
import 'auth_service.dart';

class WatchlistService {
  final _apiSettingsService = ApiSettingsService();
  final _authService = AuthService();

  Future<List<Watchlist>> getWatchlists() async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/watchlists'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => Watchlist.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load watchlists');
    }
  }

  Future<Watchlist> createWatchlist(String name) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.post(
      Uri.parse('$baseUrl/api/watchlists'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: json.encode({'name': name}),
    );

    if (response.statusCode == 201) {
      return Watchlist.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to create watchlist');
    }
  }

  Future<WatchlistItem> addItem(
    String watchlistId,
    String stockId, {
    String? notes,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.post(
      Uri.parse('$baseUrl/api/watchlists/$watchlistId/items'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: json.encode({'stockId': stockId, 'notes': notes}),
    );

    if (response.statusCode == 201) {
      return WatchlistItem.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to add item to watchlist');
    }
  }
}

class AlertService {
  final _apiSettingsService = ApiSettingsService();
  final _authService = AuthService();

  Future<List<PriceAlert>> getAlerts() async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/alerts'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => PriceAlert.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load alerts');
    }
  }

  Future<PriceAlert> createAlert({
    required String stockId,
    required String type,
    required double target,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.post(
      Uri.parse('$baseUrl/api/alerts'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: json.encode({
        'stockId': stockId,
        'alertType': type,
        'targetValue': target,
      }),
    );

    if (response.statusCode == 201) {
      return PriceAlert.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to create alert');
    }
  }
}
