import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/portfolio_analytics_models.dart';
import 'api_settings_service.dart';
import 'auth_service.dart';

class PortfolioAnalyticsService {
  final _apiSettingsService = ApiSettingsService();
  final _authService = AuthService();

  Future<PortfolioInsight> getInsights() async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/portfolio/insights'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      return PortfolioInsight.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load portfolio insights');
    }
  }

  Future<List<PortfolioSnapshotModel>> getHistory({int days = 30}) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    final response = await http.get(
      Uri.parse('$baseUrl/api/portfolio/history?days=$days'),
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => PortfolioSnapshotModel.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load performance history');
    }
  }

  Future<void> triggerSnapshot() async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final token = await _authService.getAccessToken();

    await http.post(
      Uri.parse('$baseUrl/api/portfolio/snapshot'),
      headers: {'Authorization': 'Bearer $token'},
    );
  }
}
