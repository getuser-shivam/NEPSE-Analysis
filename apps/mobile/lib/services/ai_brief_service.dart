import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:nepse_client/nepse_client.dart';

import '../models/ai_brief_focus.dart';
import '../models/ai_config.dart';
import '../models/ai_market_brief.dart';
import '../models/ai_provider.dart';
import '../models/ai_service_status.dart';

class AiBriefService {
  const AiBriefService();

  Future<AiServiceStatus> verifyConnection(AiConfig config) async {
    final normalized = _normalizeConfig(config);

    if (normalized.url.isEmpty) {
      return AiServiceStatus.notConfigured(
        providerLabel: normalized.provider.label,
        message: 'AI URL is not configured yet.',
      );
    }

    if (normalized.provider.requiresApiKey && !normalized.hasApiKey) {
      return AiServiceStatus.notConfigured(
        providerLabel: normalized.provider.label,
        message: 'AI API key is required for ${normalized.provider.label}.',
      );
    }

    try {
      final response = await http
          .post(
            Uri.parse(normalized.url),
            headers: _headers(normalized),
            body: jsonEncode(
              _chatPayload(
                normalized,
                systemPrompt:
                    'You are a connectivity check. Reply only with JSON.',
                userPrompt:
                    'Return exactly {"actionBias":"neutral","marketView":"ok","portfolioTake":"ok","watchlistFocus":[],"riskNote":"ok","nextStep":"ok"}.',
                maxTokens: 80,
              ),
            ),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode < 200 || response.statusCode >= 300) {
        return AiServiceStatus.attentionRequired(
          providerLabel: normalized.provider.label,
          checkedAt: DateTime.now(),
          message: 'HTTP ${response.statusCode}: ${_truncate(response.body)}',
        );
      }

      final content = _extractContent(response);
      if (content.trim().isEmpty) {
        return AiServiceStatus.attentionRequired(
          providerLabel: normalized.provider.label,
          checkedAt: DateTime.now(),
          message: 'Provider responded without content.',
        );
      }

      return AiServiceStatus.active(
        providerLabel: normalized.provider.label,
        checkedAt: DateTime.now(),
        message:
            '${normalized.provider.label} accepted the request and returned a model response.',
      );
    } catch (error) {
      return AiServiceStatus.attentionRequired(
        providerLabel: normalized.provider.label,
        checkedAt: DateTime.now(),
        message: 'Verification failed: $error',
      );
    }
  }

  Future<AiMarketBrief> generateBrief({
    required AiConfig config,
    required DashboardSnapshot snapshot,
    required AiBriefFocus focus,
  }) async {
    final normalized = _normalizeConfig(config);

    if (normalized.provider.requiresApiKey && !normalized.hasApiKey) {
      throw StateError(
        'Configure a ${normalized.provider.label} API key in Settings first.',
      );
    }

    final response = await http
        .post(
          Uri.parse(normalized.url),
          headers: _headers(normalized),
          body: jsonEncode(
            _chatPayload(
              normalized,
              systemPrompt:
                  'You are a NEPSE market analyst. Return concise JSON only.',
              userPrompt: _buildPrompt(snapshot, focus: focus),
              maxTokens: 500,
            ),
          ),
        )
        .timeout(const Duration(seconds: 30));

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw StateError(
        'AI request failed with HTTP ${response.statusCode}: ${_truncate(response.body)}',
      );
    }

    final raw = _extractContent(response);
    final payload = _extractJsonMap(raw);

    return AiMarketBrief.fromJson(
      payload,
      providerLabel: normalized.provider.label,
    );
  }

  AiConfig _normalizeConfig(AiConfig config) {
    final provider = config.provider;
    final url = config.url.trim().isEmpty
        ? provider.defaultUrl
        : config.url.trim();
    final requestedModel = config.model.trim().isEmpty
        ? provider.defaultModel
        : config.model.trim();
    final model =
        provider == AiProvider.groqChat &&
            requestedModel.toLowerCase() == 'openai'
        ? provider.defaultModel
        : requestedModel;

    return config.copyWith(
      url: url,
      model: model,
      apiKey: config.apiKey.trim(),
    );
  }

  Map<String, String> _headers(AiConfig config) {
    final headers = <String, String>{'Content-Type': 'application/json'};

    if (config.apiKey.trim().isNotEmpty) {
      headers['Authorization'] = 'Bearer ${config.apiKey.trim()}';
    }

    return headers;
  }

  Map<String, dynamic> _chatPayload(
    AiConfig config, {
    required String systemPrompt,
    required String userPrompt,
    required int maxTokens,
  }) {
    return {
      'model': config.model.trim().isEmpty
          ? config.provider.defaultModel
          : config.model.trim(),
      'messages': [
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': userPrompt},
      ],
      'temperature': 0.2,
      'max_tokens': maxTokens,
    };
  }

  String _buildPrompt(
    DashboardSnapshot snapshot, {
    required AiBriefFocus focus,
  }) {
    final symbolLines = snapshot.symbols
        .take(6)
        .map((symbol) {
          final closes = symbol.recentPrices
              .take(3)
              .map((price) => price.close.toStringAsFixed(2))
              .join(', ');
          final currentPrice =
              symbol.holding?.currentPrice?.toStringAsFixed(2) ??
              (symbol.recentPrices.isEmpty
                  ? 'n/a'
                  : symbol.recentPrices.first.close.toStringAsFixed(2));
          final shares = symbol.holding?.shares.toStringAsFixed(2) ?? '0.00';

          return '- ${symbol.symbol}: inPortfolio=${symbol.inPortfolio}, inWatchlist=${symbol.inWatchlist}, shares=$shares, current=$currentPrice, recentCloses=[$closes]';
        })
        .join('\n');

    final watchlist = snapshot.watchlist.map((item) => item.symbol).join(', ');

    return '''
Analyze this NEPSE dashboard snapshot and return JSON only.

Return exactly:
{"actionBias":"bullish|bearish|neutral","marketView":"1-2 sentence view","portfolioTake":"1 sentence portfolio note","watchlistFocus":["up to 3 short symbol notes"],"riskNote":"1 sentence risk note","nextStep":"1 sentence next step"}

Snapshot:
- Total investment: ${snapshot.portfolio.summary.totalInvestment.toStringAsFixed(2)}
- Total value: ${snapshot.portfolio.summary.totalValue.toStringAsFixed(2)}
- Total gain/loss: ${snapshot.portfolio.summary.totalGainLoss.toStringAsFixed(2)}
- Total return: ${snapshot.portfolio.summary.totalReturnPct.toStringAsFixed(2)}%
- Holdings count: ${snapshot.portfolio.summary.numStocks}
- Watchlist: ${watchlist.isEmpty ? 'none' : watchlist}
- Default period: ${snapshot.settings.defaultPeriod}
- Focus mode: ${focus.label}
- Focus instruction: ${focus.promptInstruction}

Tracked symbols:
$symbolLines

Focus on practical NEPSE commentary for a mobile investor dashboard.
''';
  }

  String _extractContent(http.Response response) {
    final body = response.body.trim();
    if (body.isEmpty) {
      return '';
    }

    try {
      final decoded = jsonDecode(body);
      if (decoded is Map<String, dynamic>) {
        final choices = decoded['choices'];
        if (choices is List && choices.isNotEmpty) {
          final first = choices.first;
          if (first is Map<String, dynamic>) {
            final message = first['message'];
            if (message is Map<String, dynamic>) {
              return message['content']?.toString() ?? body;
            }
          }
        }
      }
    } catch (_) {
      // Fall back to the raw body.
    }

    return body;
  }

  Map<String, dynamic> _extractJsonMap(String raw) {
    final normalized = raw.trim();
    if (normalized.isEmpty) {
      return const {
        'actionBias': 'neutral',
        'marketView': 'No analysis returned.',
        'portfolioTake': 'No portfolio note returned.',
        'watchlistFocus': <String>[],
        'riskNote': 'No risk note returned.',
        'nextStep': 'No next step returned.',
      };
    }

    try {
      final start = normalized.indexOf('{');
      final end = normalized.lastIndexOf('}');
      if (start != -1 && end > start) {
        return Map<String, dynamic>.from(
          jsonDecode(normalized.substring(start, end + 1)) as Map,
        );
      }
    } catch (_) {
      // Fall back below.
    }

    return {
      'actionBias': 'neutral',
      'marketView': normalized,
      'portfolioTake': 'Fallback analysis was used.',
      'watchlistFocus': <String>[],
      'riskNote': 'Review the raw response before acting.',
      'nextStep': 'Adjust the provider settings or prompt and try again.',
    };
  }

  String _truncate(String value, {int max = 180}) {
    if (value.length <= max) {
      return value;
    }
    return '${value.substring(0, max)}...';
  }
}
