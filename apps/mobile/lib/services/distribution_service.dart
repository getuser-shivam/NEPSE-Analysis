import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/distribution_models.dart';
import 'api_settings_service.dart';
import 'auth_service.dart';

class DistributionService {
  final _apiSettingsService = ApiSettingsService();
  final _authService = AuthService();

  Future<Order> placeOrder({
    required List<Map<String, dynamic>> items,
    required String shippingAddress,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.post(
      Uri.parse('$baseUrl/api/dist/orders'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: json.encode({'items': items, 'shippingAddress': shippingAddress}),
    );

    if (response.statusCode == 201) {
      return Order.fromJson(json.decode(response.body));
    } else {
      final error = json.decode(response.body);
      throw Exception(error['message'] ?? 'Failed to place order');
    }
  }

  Future<InventoryForecast> getForecast(String productId) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/dist/inventory/forecast/$productId'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      return InventoryForecast.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load forecast');
    }
  }

  Future<Map<String, dynamic>> getAnalytics() async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/dist/analytics'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load analytics');
    }
  }

  Future<void> logActivity(
    String? productId,
    String action, [
    Map<String, dynamic>? metadata,
  ]) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    await http.post(
      Uri.parse('$baseUrl/api/dist/activities'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: json.encode({
        'productId': productId,
        'action': action,
        'metadata': metadata,
      }),
    );
  }
}
