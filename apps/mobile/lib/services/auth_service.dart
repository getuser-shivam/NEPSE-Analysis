import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../models/user_auth.dart';
import 'api_settings_service.dart';

class AuthService {
  final _storage = const FlutterSecureStorage();
  final _apiSettingsService = ApiSettingsService();

  static const _accessTokenKey = 'access_token';
  static const _refreshTokenKey = 'refresh_token';

  Future<Map<String, dynamic>> login({
    required String email,
    required String password,
    String? mfaCode,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final response = await http.post(
      Uri.parse('$baseUrl/api/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'email': email,
        'password': password,
        if (mfaCode != null) 'mfaCode': mfaCode,
      }),
    );

    final data = json.decode(response.body);

    if (response.statusCode == 200) {
      if (data['mfaRequired'] == true) {
        return {'mfaRequired': true, 'userId': data['userId']};
      }

      await _saveTokens(data['accessToken'], data['refreshToken']);
      return {'user': UserAuth.fromJson(data['user'])};
    } else {
      throw Exception(data['message'] ?? 'Login failed');
    }
  }

  Future<void> register({
    required String email,
    required String password,
    String? name,
  }) async {
    final baseUrl = await _apiSettingsService.loadBaseUrl();
    final response = await http.post(
      Uri.parse('$baseUrl/api/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'email': email, 'password': password, 'name': name}),
    );

    if (response.statusCode != 201) {
      final data = json.decode(response.body);
      throw Exception(data['message'] ?? 'Registration failed');
    }
  }

  Future<String?> getAccessToken() => _storage.read(key: _accessTokenKey);

  Future<void> _saveTokens(String access, String refresh) async {
    await _storage.write(key: _accessTokenKey, value: access);
    await _storage.write(key: _refreshTokenKey, value: refresh);
  }

  Future<void> logout() async {
    await _storage.delete(key: _accessTokenKey);
    await _storage.delete(key: _refreshTokenKey);
  }
}
