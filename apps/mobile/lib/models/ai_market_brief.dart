enum AiActionBias { bullish, bearish, neutral }

AiActionBias aiActionBiasFromKey(String? value) {
  return switch (value?.trim().toLowerCase()) {
    'bullish' || 'buy' || 'long' => AiActionBias.bullish,
    'bearish' || 'sell' || 'short' => AiActionBias.bearish,
    _ => AiActionBias.neutral,
  };
}

class AiMarketBrief {
  const AiMarketBrief({
    required this.providerLabel,
    required this.generatedAt,
    required this.actionBias,
    required this.marketView,
    required this.portfolioTake,
    required this.watchlistFocus,
    required this.riskNote,
    required this.nextStep,
  });

  final String providerLabel;
  final DateTime generatedAt;
  final AiActionBias actionBias;
  final String marketView;
  final String portfolioTake;
  final List<String> watchlistFocus;
  final String riskNote;
  final String nextStep;

  factory AiMarketBrief.fromJson(
    Map<String, dynamic> json, {
    required String providerLabel,
  }) {
    final rawFocus = json['watchlistFocus'];
    final focus = rawFocus is List
        ? rawFocus
              .map((item) => item.toString())
              .where((item) => item.isNotEmpty)
              .toList(growable: false)
        : <String>[];

    return AiMarketBrief(
      providerLabel: providerLabel,
      generatedAt: _parseGeneratedAt(json['generatedAt']),
      actionBias: aiActionBiasFromKey(json['actionBias']?.toString()),
      marketView: json['marketView']?.toString().trim().isNotEmpty == true
          ? json['marketView'].toString().trim()
          : 'No market view was returned.',
      portfolioTake: json['portfolioTake']?.toString().trim().isNotEmpty == true
          ? json['portfolioTake'].toString().trim()
          : 'No portfolio note was returned.',
      watchlistFocus: focus,
      riskNote: json['riskNote']?.toString().trim().isNotEmpty == true
          ? json['riskNote'].toString().trim()
          : 'No explicit risk note was returned.',
      nextStep: json['nextStep']?.toString().trim().isNotEmpty == true
          ? json['nextStep'].toString().trim()
          : 'No next step was returned.',
    );
  }

  static DateTime _parseGeneratedAt(dynamic value) {
    if (value is String) {
      return DateTime.tryParse(value) ?? DateTime.now();
    }
    return DateTime.now();
  }
}
