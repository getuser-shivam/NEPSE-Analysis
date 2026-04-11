enum AiProvider { groqChat, pollinationsText }

extension AiProviderX on AiProvider {
  String get key => switch (this) {
    AiProvider.groqChat => 'groq',
    AiProvider.pollinationsText => 'pollinations',
  };

  String get label => switch (this) {
    AiProvider.groqChat => 'Groq',
    AiProvider.pollinationsText => 'Pollen / Pollinations',
  };

  String get defaultUrl => switch (this) {
    AiProvider.groqChat => 'https://api.groq.com/openai/v1/chat/completions',
    AiProvider.pollinationsText => 'https://text.pollinations.ai/',
  };

  String get defaultModel => switch (this) {
    AiProvider.groqChat => 'llama-3.1-8b-instant',
    AiProvider.pollinationsText => 'openai',
  };

  bool get requiresApiKey => switch (this) {
    AiProvider.groqChat => true,
    AiProvider.pollinationsText => false,
  };

  String get helperText => switch (this) {
    AiProvider.groqChat =>
      'Groq uses an OpenAI-compatible chat API and requires an API key.',
    AiProvider.pollinationsText =>
      'Pollinations can run without a key, but you can still keep one saved locally for compatibility.',
  };
}

AiProvider aiProviderFromKey(String? value) {
  return AiProvider.values.firstWhere(
    (provider) => provider.key == value,
    orElse: () => AiProvider.groqChat,
  );
}
