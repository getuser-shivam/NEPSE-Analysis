import 'package:shared_preferences/shared_preferences.dart';

const defaultApiBaseUrl = String.fromEnvironment(
  'NEPSE_API_URL',
  defaultValue: 'http://192.168.1.79:4000',
);

class ApiSettingsService {
  static const _apiBaseUrlKey = 'api_base_url';

  Future<String> loadBaseUrl() async {
    final preferences = await SharedPreferences.getInstance();
    return preferences.getString(_apiBaseUrlKey) ?? defaultApiBaseUrl;
  }

  Future<void> saveBaseUrl(String value) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(_apiBaseUrlKey, value);
  }
}
