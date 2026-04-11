/// Analysis Engine
///
/// Core business logic for analyzing stock market data, combining technical
/// indicators with AI-powered insights to generate comprehensive market analysis.
library nepse_analysis_analysis_engine;

import '../models/stock.dart';
import '../utils/calculations.dart';
import 'ai_service.dart';

/// Represents a comprehensive market analysis result
class MarketAnalysis {
  final String symbol;
  final double currentPrice;
  final TechnicalIndicators indicators;
  final AIAnalysisResult? aiAnalysis;
  final PricePrediction? pricePrediction;
  final TrendAnalysis? trendAnalysis;
  final String recommendation;
  final double confidence;
  final DateTime generatedAt;

  MarketAnalysis({
    required this.symbol,
    required this.currentPrice,
    required this.indicators,
    this.aiAnalysis,
    this.pricePrediction,
    this.trendAnalysis,
    required this.recommendation,
    required this.confidence,
    DateTime? generatedAt,
  }) : generatedAt = generatedAt ?? DateTime.now();

  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'currentPrice': currentPrice,
      'indicators': {
        'rsi14': indicators.rsi14,
        'macd': indicators.macdHistogram,
        'signal': indicators.signal,
      },
      'aiAnalysis': aiAnalysis?.toJson(),
      'recommendation': recommendation,
      'confidence': confidence,
      'generatedAt': generatedAt.toIso8601String(),
    };
  }
}

/// Result from AI analysis
class AIAnalysisResult {
  final String sentiment;
  final double confidence;
  final String prediction;
  final List<String> factors;
  final String? recommendation;

  AIAnalysisResult({
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
}

/// Core analysis engine for stock market data
class AnalysisEngine {
  final AIService _aiService;

  AnalysisEngine({AIService? aiService})
    : _aiService = aiService ?? AIService();

  /// Performs comprehensive market analysis for a stock
  ///
  /// [symbol] - Stock symbol
  /// [prices] - Historical price data
  /// [highs] - Historical high prices
  /// [lows] - Historical low prices
  /// [enableAI] - Whether to include AI analysis
  Future<MarketAnalysis> analyze({
    required String symbol,
    required List<double> prices,
    required List<double> highs,
    required List<double> lows,
    bool enableAI = true,
  }) async {
    if (prices.isEmpty || highs.isEmpty || lows.isEmpty) {
      throw ArgumentError('Price data cannot be empty');
    }

    // Calculate technical indicators
    final indicators = analyzeTechnicals(
      prices: prices,
      highs: highs,
      lows: lows,
    );

    // Get AI analysis if enabled
    AIAnalysisResult? aiAnalysis;
    if (enableAI && _aiService.config.isConfigured) {
      try {
        aiAnalysis = await _fetchAIAnalysis(symbol, prices, indicators);
      } catch (e) {
        // Continue without AI analysis on error
        print('AI analysis failed: $e');
      }
    }

    // Generate recommendation
    final recommendation = _generateRecommendation(indicators, aiAnalysis);
    final confidence = _calculateConfidence(indicators, aiAnalysis);

    return MarketAnalysis(
      symbol: symbol,
      currentPrice: prices.last,
      indicators: indicators,
      aiAnalysis: aiAnalysis,
      recommendation: recommendation,
      confidence: confidence,
    );
  }

  /// Fetches AI analysis from Groq
  Future<AIAnalysisResult> _fetchAIAnalysis(
    String symbol,
    List<double> prices,
    TechnicalIndicators indicators,
  ) async {
    final marketData = {
      'symbol': symbol,
      'currentPrice': prices.last,
      'priceChange': prices.length > 1
          ? ((prices.last - prices[prices.length - 2]) /
                    prices[prices.length - 2]) *
                100
          : 0,
      'indicators': {
        'rsi14': indicators.rsi14,
        'macd': indicators.macdHistogram,
        'signal': indicators.signal,
      },
    };

    final result = await _aiService.analyzeWithGroq(
      symbol: symbol,
      marketData: marketData,
    );

    return AIAnalysisResult(
      sentiment: result.sentiment,
      confidence: result.confidence,
      prediction: result.prediction,
      factors: result.factors,
      recommendation: result.recommendation,
    );
  }

  /// Generates trading recommendation based on indicators and AI
  String _generateRecommendation(
    TechnicalIndicators indicators,
    AIAnalysisResult? aiAnalysis,
  ) {
    // Start with technical signal
    var recommendation = indicators.signal;

    // Adjust based on AI sentiment if available
    if (aiAnalysis != null && aiAnalysis.confidence > 0.6) {
      if (aiAnalysis.sentiment == 'bullish' &&
          recommendation.contains('sell')) {
        recommendation = 'hold';
      } else if (aiAnalysis.sentiment == 'bearish' &&
          recommendation.contains('buy')) {
        recommendation = 'hold';
      }
    }

    return recommendation;
  }

  /// Calculates overall confidence score
  double _calculateConfidence(
    TechnicalIndicators indicators,
    AIAnalysisResult? aiAnalysis,
  ) {
    var confidence = 0.5;

    // Technical indicator confidence
    if (indicators.rsi14 != null) {
      // Higher confidence when RSI is in clear overbought/oversold
      if (indicators.rsi14! < 30 || indicators.rsi14! > 70) {
        confidence += 0.2;
      }
    }

    // Trend strength confidence
    if (indicators.trendStrength.abs() > 0.6) {
      confidence += 0.15;
    }

    // AI confidence if available
    if (aiAnalysis != null) {
      confidence = (confidence + aiAnalysis.confidence) / 2;
    }

    return confidence.clamp(0.0, 1.0);
  }

  /// Analyzes market trend direction and strength
  TrendAnalysis analyzeTrend(List<double> prices) {
    if (prices.length < 20) {
      return TrendAnalysis(direction: 'neutral', strength: 0, confidence: 0.3);
    }

    final direction = _detectTrendDirection(prices);
    final strength = _calculateTrendStrength(prices);

    return TrendAnalysis(
      direction: direction,
      strength: strength,
      confidence: strength,
    );
  }

  /// Detects trend direction from price history
  String _detectTrendDirection(List<double> prices) {
    if (prices.length < 10) return 'neutral';

    final recent = prices.sublist(prices.length - 10);
    final firstHalf = recent.sublist(0, 5).reduce((a, b) => a + b) / 5;
    final secondHalf = recent.sublist(5).reduce((a, b) => a + b) / 5;

    final change = ((secondHalf - firstHalf) / firstHalf) * 100;

    if (change > 2) return 'up';
    if (change < -2) return 'down';
    return 'neutral';
  }

  /// Calculates trend strength (0-1)
  double _calculateTrendStrength(List<double> prices) {
    if (prices.length < 10) return 0.5;

    int upDays = 0;
    int downDays = 0;

    for (int i = 1; i < prices.length; i++) {
      if (prices[i] > prices[i - 1]) {
        upDays++;
      } else if (prices[i] < prices[i - 1]) {
        downDays++;
      }
    }

    final total = prices.length - 1;
    return (upDays - downDays).abs() / total;
  }
}

/// Trend analysis result
class TrendAnalysis {
  final String direction;
  final double strength;
  final double confidence;

  TrendAnalysis({
    required this.direction,
    required this.strength,
    required this.confidence,
  });
}
