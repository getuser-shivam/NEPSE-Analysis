import 'ai_provider.dart';

class AiConfig {
  const AiConfig({
    required this.provider,
    required this.url,
    required this.model,
    required this.apiKey,
  });

  factory AiConfig.initial() {
    return AiConfig(
      provider: AiProvider.groqChat,
      url: AiProvider.groqChat.defaultUrl,
      model: AiProvider.groqChat.defaultModel,
      apiKey: '',
    );
  }

  final AiProvider provider;
  final String url;
  final String model;
  final String apiKey;

  bool get hasApiKey => apiKey.trim().isNotEmpty;

  AiConfig copyWith({
    AiProvider? provider,
    String? url,
    String? model,
    String? apiKey,
  }) {
    return AiConfig(
      provider: provider ?? this.provider,
      url: url ?? this.url,
      model: model ?? this.model,
      apiKey: apiKey ?? this.apiKey,
    );
  }
}
