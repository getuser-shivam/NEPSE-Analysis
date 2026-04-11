import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/ai_config.dart';
import '../models/ai_provider.dart';

class AiSettingsService {
  static const _aiProviderKey = 'nepse_ai_provider';
  static const _aiUrlKey = 'nepse_ai_url';
  static const _aiModelKey = 'nepse_ai_model';
  static const _aiApiKeyKey = 'nepse_ai_api_key';

  final FlutterSecureStorage _secureStorage = const FlutterSecureStorage();
  SharedPreferences? _prefs;

  Future<void> init() async {
    if (_prefs != null) {
      return;
    }

    _prefs = await SharedPreferences.getInstance();
  }

  Future<AiConfig> loadConfig() async {
    await init();

    final provider = aiProviderFromKey(_prefs?.getString(_aiProviderKey));
    final savedUrl = _prefs?.getString(_aiUrlKey)?.trim();
    final savedModel = _prefs?.getString(_aiModelKey)?.trim();
    final apiKey = await _secureStorage.read(key: _aiApiKeyKey) ?? '';

    return AiConfig(
      provider: provider,
      url: (savedUrl == null || savedUrl.isEmpty)
          ? provider.defaultUrl
          : savedUrl,
      model: (savedModel == null || savedModel.isEmpty)
          ? provider.defaultModel
          : savedModel,
      apiKey: apiKey,
    );
  }

  Future<void> saveConfig(AiConfig config) async {
    await init();

    await _prefs?.setString(_aiProviderKey, config.provider.key);
    await _prefs?.setString(_aiUrlKey, config.url.trim());
    await _prefs?.setString(_aiModelKey, config.model.trim());

    final trimmedKey = config.apiKey.trim();
    if (trimmedKey.isEmpty) {
      await _secureStorage.delete(key: _aiApiKeyKey);
    } else {
      await _secureStorage.write(key: _aiApiKeyKey, value: trimmedKey);
    }
  }
}
