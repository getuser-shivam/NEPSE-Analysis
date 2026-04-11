/// AI Service
///
/// Integration with Groq AI and Pollens AI APIs for predictive modeling
/// and trend analysis of stock market data.
library nepse_analysis_ai_service;

import 'dart:convert';
import 'package:http/http.dart' as http;

/// AI service configuration
class AIConfig {
  final String groqApiKey;
  final String pollensApiKey;
  final String groqBaseUrl;
  final String pollensBaseUrl;
  final bool enableGroq;
  final bool enablePollens;
  final int maxTokens;
  final double temperature;
  final int timeoutMs;

  AIConfig({
    required this.groqApiKey,
    required this.pollensApiKey,
    this.groqBaseUrl = 'https://api.groq.com/openai/v1',
    this.pollensBaseUrl = 'https://api.pollens.ai/v1',
    this.enableGroq = true,
    this.enablePollens = true,
    this.maxTokens = 500,
    this.temperature = 0.3,
    this.timeoutMs = 30000,
  });

  bool get isConfigured =>
      (enableGroq && groqApiKey.isNotEmpty) ||
      (enablePollens && pollensApiKey.isNotEmpty);
}

/// Result from Groq AI analysis
class GroqAnalysisResult {
  final String sentiment;
  final double confidence;
  final String prediction;
  final List<String> factors;
  final String? recommendation;

  GroqAnalysisResult({
    required this.sentiment,
    required this.confidence,
    required this.prediction,
    required this.factors,
    this.recommendation,
  });

  Map<String, dynamic> toJson() {
    return {
      'sentiment': sentiment,
      'confidence': confidence,
      'prediction': prediction,
      'factors': factors,
      'recommendation': recommendation,
    };
  }

  factory GroqAnalysisResult.fromJson(Map<String, dynamic> json) {
    return GroqAnalysisResult(
      sentiment: json['sentiment'] ?? 'neutral',
      confidence: (json['confidence'] ?? 0.5).toDouble(),
      prediction: json['prediction'] ?? 'No prediction available',
      factors: List<String>.from(json['factors'] ?? []),
      recommendation: json['recommendation'],
    );
  }
}

/// Result from Pollens AI prediction
class PollensPredictionResult {
  final List<double> predictions;
  final String trendDirection;
  final double confidence;
  final Map<String, dynamic> metadata;

  PollensPredictionResult({
    required this.predictions,
    required this.trendDirection,
    required this.confidence,
    required this.metadata,
  });

  Map<String, dynamic> toJson() {
    return {
      'predictions': predictions,
      'trendDirection': trendDirection,
      'confidence': confidence,
      'metadata': metadata,
    };
  }
}

/// AI Service for integrating with Groq and Pollens APIs
class AIService {
  AIConfig config;

  AIService({AIConfig? config})
    : config = config ?? AIConfig(groqApiKey: '', pollensApiKey: '');

  /// Analyze market data using Groq AI
  Future<GroqAnalysisResult> analyzeWithGroq({
    required String symbol,
    required Map<String, dynamic> marketData,
  }) async {
    if (!config.enableGroq || config.groqApiKey.isEmpty) {
      throw Exception('Groq AI is not configured');
    }

    try {
      final prompt = _buildGroqPrompt(symbol, marketData);

      final response = await http
          .post(
            Uri.parse('${config.groqBaseUrl}/chat/completions'),
            headers: {
              'Authorization': 'Bearer ${config.groqApiKey}',
              'Content-Type': 'application/json',
            },
            body: jsonEncode({
              'model': 'llama3-8b-8192',
              'messages': [
                {
                  'role': 'system',
                  'content':
                      'You are a financial analyst specializing in Nepal Stock Exchange (NEPSE). Provide concise, data-driven analysis in JSON format.',
                },
                {'role': 'user', 'content': prompt},
              ],
              'max_tokens': config.maxTokens,
              'temperature': config.temperature,
            }),
          )
          .timeout(Duration(milliseconds: config.timeoutMs));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final content = data['choices'][0]['message']['content'];
        return _parseGroqResponse(content);
      } else {
        throw Exception('Groq AI request failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Groq AI analysis failed: $e');
    }
  }

  /// Predict future prices using Pollens AI
  Future<PollensPredictionResult> predictWithPollens({
    required String symbol,
    required List<double> priceHistory,
    required int daysAhead,
  }) async {
    if (!config.enablePollens || config.pollensApiKey.isEmpty) {
      throw Exception('Pollens AI is not configured');
    }

    try {
      final response = await http
          .post(
            Uri.parse('${config.pollensBaseUrl}/predict'),
            headers: {
              'Authorization': 'Bearer ${config.pollensApiKey}',
              'Content-Type': 'application/json',
            },
            body: jsonEncode({
              'symbol': symbol,
              'priceHistory': priceHistory,
              'daysAhead': daysAhead,
              'model': 'lstm-v2',
            }),
          )
          .timeout(Duration(milliseconds: config.timeoutMs));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return PollensPredictionResult(
          predictions: List<double>.from(data['predictions']),
          trendDirection: data['trendDirection'],
          confidence: (data['confidence'] ?? 0.5).toDouble(),
          metadata: data['metadata'] ?? {},
        );
      } else {
        throw Exception('Pollens AI request failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Pollens AI prediction failed: $e');
    }
  }

  /// Build prompt for Groq AI
  String _buildGroqPrompt(String symbol, Map<String, dynamic> marketData) {
    final currentPrice = marketData['currentPrice'] ?? 0;
    final priceChange = marketData['priceChange'] ?? 0;
    final indicators = marketData['indicators'] ?? {};

    return '''
Analyze the following stock data for $symbol:

Current Price: $currentPrice
Price Change: $priceChange%

Technical Indicators:
- RSI (14): ${indicators['rsi14'] ?? 'N/A'}
- MACD: ${indicators['macdHistogram'] ?? 'N/A'}
- Signal: ${indicators['signal'] ?? 'N/A'}

Provide a JSON response with the following structure:
{
  "sentiment": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "prediction": "short description of expected price movement",
  "factors": ["factor1", "factor2", "factor3"],
  "recommendation": "buy|sell|hold"
}
''';
  }

  /// Parse Groq AI response
  GroqAnalysisResult _parseGroqResponse(String content) {
    try {
      // Extract JSON from the response
      final jsonStart = content.indexOf('{');
      final jsonEnd = content.lastIndexOf('}') + 1;

      if (jsonStart == -1 || jsonEnd == 0) {
        throw Exception('No JSON found in response');
      }

      final jsonString = content.substring(jsonStart, jsonEnd);
      final jsonData = jsonDecode(jsonString);

      return GroqAnalysisResult.fromJson(jsonData);
    } catch (e) {
      // Fallback to default response if parsing fails
      return GroqAnalysisResult(
        sentiment: 'neutral',
        confidence: 0.5,
        prediction: 'Unable to parse AI response',
        factors: ['Data insufficient for analysis'],
      );
    }
  }

  /// Update configuration
  void updateConfig(AIConfig newConfig) {
    config = newConfig;
  }

  /// Check if AI services are available
  bool isGroqAvailable() => config.enableGroq && config.groqApiKey.isNotEmpty;
  bool isPollensAvailable() =>
      config.enablePollens && config.pollensApiKey.isNotEmpty;
}
