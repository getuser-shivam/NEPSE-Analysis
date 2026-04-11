/// Recommendation Engine
///
/// Generates personalized investment recommendations based on technical analysis,
/// AI insights, and user risk preferences.
library nepse_analysis_recommendation_engine;

import '../models/portfolio.dart';
import '../models/stock.dart';
import 'analysis_engine.dart';

/// User risk profile levels
enum RiskProfile { conservative, moderate, aggressive }

/// Investment recommendation action
enum RecommendationAction { buy, sell, hold, watch }

/// Represents a personalized investment recommendation
class InvestmentRecommendation {
  final String symbol;
  final String stockName;
  final RecommendationAction action;
  final double confidence;
  final double? targetPrice;
  final double? stopLoss;
  final String timeHorizon;
  final String rationale;
  final RiskLevel riskLevel;
  final List<String> technicalFactors;
  final String? aiInsights;
  final DateTime generatedAt;

  InvestmentRecommendation({
    required this.symbol,
    required this.stockName,
    required this.action,
    required this.confidence,
    this.targetPrice,
    this.stopLoss,
    required this.timeHorizon,
    required this.rationale,
    required this.riskLevel,
    required this.technicalFactors,
    this.aiInsights,
    DateTime? generatedAt,
  }) : generatedAt = generatedAt ?? DateTime.now();

  /// Converts recommendation to JSON
  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'stockName': stockName,
      'action': action.name.toUpperCase(),
      'confidence': confidence,
      'targetPrice': targetPrice,
      'stopLoss': stopLoss,
      'timeHorizon': timeHorizon,
      'rationale': rationale,
      'riskLevel': riskLevel.name,
      'technicalFactors': technicalFactors,
      'aiInsights': aiInsights,
      'generatedAt': generatedAt.toIso8601String(),
    };
  }

  @override
  String toString() {
    return 'InvestmentRecommendation(${action.name.toUpperCase()} $symbol @ $confidence confidence)';
  }
}

/// Risk level classification
enum RiskLevel { low, medium, high }

/// Recommendation engine for generating personalized investment advice
class RecommendationEngine {
  final AnalysisEngine _analysisEngine;

  RecommendationEngine({AnalysisEngine? analysisEngine})
    : _analysisEngine = analysisEngine ?? AnalysisEngine();

  /// Generates personalized recommendation for a stock
  ///
  /// [stock] - Stock to analyze
  /// [prices] - Historical prices
  /// [highs] - Historical highs
  /// [lows] - Historical lows
  /// [userRiskProfile] - User's risk tolerance
  /// [userPortfolio] - User's current portfolio (optional)
  Future<InvestmentRecommendation> generateRecommendation({
    required Stock stock,
    required List<double> prices,
    required List<double> highs,
    required List<double> lows,
    RiskProfile userRiskProfile = RiskProfile.moderate,
    Portfolio? userPortfolio,
  }) async {
    if (prices.isEmpty) {
      throw ArgumentError('Price data cannot be empty');
    }

    final currentPrice = prices.last;

    // Perform market analysis
    final analysis = await _analysisEngine.analyze(
      symbol: stock.symbol,
      prices: prices,
      highs: highs,
      lows: lows,
    );

    // Determine action based on analysis
    final action = _determineAction(
      analysis,
      userRiskProfile,
      userPortfolio,
      stock.symbol,
    );

    // Calculate target and stop-loss prices
    final (targetPrice, stopLoss) = _calculatePriceTargets(
      currentPrice,
      analysis.indicators,
      action,
    );

    // Determine risk level
    final riskLevel = _assessRiskLevel(
      analysis.indicators.volatility,
      userRiskProfile,
    );

    // Generate rationale
    final rationale = _generateRationale(action, analysis, riskLevel);

    // Extract technical factors
    final technicalFactors = _extractTechnicalFactors(analysis);

    // Determine time horizon
    final timeHorizon = _determineTimeHorizon(action, userRiskProfile);

    return InvestmentRecommendation(
      symbol: stock.symbol,
      stockName: stock.name,
      action: action,
      confidence: analysis.confidence,
      targetPrice: targetPrice,
      stopLoss: stopLoss,
      timeHorizon: timeHorizon,
      rationale: rationale,
      riskLevel: riskLevel,
      technicalFactors: technicalFactors,
      aiInsights: analysis.aiAnalysis?.prediction,
    );
  }

  /// Determines the recommended action
  RecommendationAction _determineAction(
    MarketAnalysis analysis,
    RiskProfile riskProfile,
    Portfolio? portfolio,
    String symbol,
  ) {
    // Check if already in portfolio
    final inPortfolio = portfolio?.holdings.containsKey(symbol) ?? false;

    // Start with technical signal
    final signal = analysis.indicators.signal;

    // Map technical signal to action
    if (signal.contains('buy')) {
      if (inPortfolio) {
        // Already have it, check if should add more
        return analysis.confidence > 0.7
            ? RecommendationAction.buy
            : RecommendationAction.hold;
      }
      return RecommendationAction.buy;
    } else if (signal.contains('sell')) {
      if (inPortfolio) {
        return RecommendationAction.sell;
      }
      return RecommendationAction.watch;
    }

    return inPortfolio ? RecommendationAction.hold : RecommendationAction.watch;
  }

  /// Calculates target price and stop-loss
  (double?, double?) _calculatePriceTargets(
    double currentPrice,
    TechnicalIndicators indicators,
    RecommendationAction action,
  ) {
    double? targetPrice;
    double? stopLoss;

    if (action == RecommendationAction.buy) {
      // Target based on Bollinger Bands or 5-10% upside
      if (indicators.bbUpper != null) {
        targetPrice = indicators.bbUpper;
      } else {
        targetPrice = currentPrice * 1.08; // 8% upside
      }

      // Stop-loss at lower Bollinger Band or 5% downside
      if (indicators.bbLower != null &&
          indicators.bbLower! < currentPrice * 0.95) {
        stopLoss = indicators.bbLower;
      } else {
        stopLoss = currentPrice * 0.95; // 5% stop-loss
      }
    } else if (action == RecommendationAction.sell) {
      // For sell recommendations, target is current (sell now)
      // Stop-loss is the target if price goes back up
      targetPrice = currentPrice * 0.95; // 5% downside expected
      stopLoss = currentPrice * 1.05; // Sell if goes up 5%
    }

    return (targetPrice, stopLoss);
  }

  /// Assesses risk level based on volatility and user profile
  RiskLevel _assessRiskLevel(double volatility, RiskProfile riskProfile) {
    // Base risk on volatility
    RiskLevel baseRisk;
    if (volatility < 2) {
      baseRisk = RiskLevel.low;
    } else if (volatility < 5) {
      baseRisk = RiskLevel.medium;
    } else {
      baseRisk = RiskLevel.high;
    }

    // Adjust for user risk profile
    switch (riskProfile) {
      case RiskProfile.conservative:
        // Increase perceived risk for conservative users
        if (baseRisk == RiskLevel.medium) return RiskLevel.high;
        if (baseRisk == RiskLevel.high) return RiskLevel.high;
        return baseRisk;
      case RiskProfile.aggressive:
        // Decrease perceived risk for aggressive users
        if (baseRisk == RiskLevel.high) return RiskLevel.medium;
        if (baseRisk == RiskLevel.medium) return RiskLevel.low;
        return baseRisk;
      case RiskProfile.moderate:
        return baseRisk;
    }
  }

  /// Generates human-readable rationale
  String _generateRationale(
    RecommendationAction action,
    MarketAnalysis analysis,
    RiskLevel riskLevel,
  ) {
    final parts = <String>[];

    // Technical analysis summary
    if (analysis.indicators.rsi14 != null) {
      if (analysis.indicators.rsi14! < 30) {
        parts.add(
          'RSI indicates oversold conditions (${analysis.indicators.rsi14!.toStringAsFixed(1)})',
        );
      } else if (analysis.indicators.rsi14! > 70) {
        parts.add(
          'RSI indicates overbought conditions (${analysis.indicators.rsi14!.toStringAsFixed(1)})',
        );
      }
    }

    // Signal explanation
    if (analysis.indicators.signal.contains('buy')) {
      parts.add('Technical indicators suggest bullish momentum');
    } else if (analysis.indicators.signal.contains('sell')) {
      parts.add('Technical indicators suggest bearish pressure');
    } else {
      parts.add('Technical indicators are neutral');
    }

    // Risk assessment
    parts.add('Risk level: ${riskLevel.name}');

    // AI insights if available
    if (analysis.aiAnalysis != null &&
        analysis.aiAnalysis!.factors.isNotEmpty) {
      parts.add(
        'AI analysis highlights: ${analysis.aiAnalysis!.factors.take(2).join(', ')}',
      );
    }

    return parts.join('. ') + '.';
  }

  /// Extracts key technical factors
  List<String> _extractTechnicalFactors(MarketAnalysis analysis) {
    final factors = <String>[];

    if (analysis.indicators.rsi14 != null) {
      factors.add('RSI: ${analysis.indicators.rsi14!.toStringAsFixed(1)}');
    }

    if (analysis.indicators.macdHistogram != null) {
      final macdSignal = analysis.indicators.macdHistogram! > 0
          ? 'Bullish'
          : 'Bearish';
      factors.add('MACD: $macdSignal');
    }

    if (analysis.indicators.sma20 != null &&
        analysis.indicators.sma50 != null) {
      final trend = analysis.indicators.sma20! > analysis.indicators.sma50!
          ? 'Uptrend'
          : 'Downtrend';
      factors.add('Moving Averages: $trend');
    }

    if (analysis.indicators.trendStrength.abs() > 0.5) {
      final strength = analysis.indicators.trendStrength > 0
          ? 'Strong'
          : 'Weak';
      factors.add('Trend: $strength');
    }

    return factors;
  }

  /// Determines appropriate time horizon
  String _determineTimeHorizon(
    RecommendationAction action,
    RiskProfile riskProfile,
  ) {
    if (action == RecommendationAction.sell) {
      return 'Immediate to 1 week';
    }

    switch (riskProfile) {
      case RiskProfile.conservative:
        return '3-6 months';
      case RiskProfile.moderate:
        return '1-3 months';
      case RiskProfile.aggressive:
        return '2-4 weeks';
    }
  }

  /// Generates recommendations for multiple stocks
  Future<List<InvestmentRecommendation>> generateRecommendationsForWatchlist({
    required List<Stock> watchlist,
    required Map<String, List<double>> priceData,
    RiskProfile riskProfile = RiskProfile.moderate,
    Portfolio? portfolio,
  }) async {
    final recommendations = <InvestmentRecommendation>[];

    for (final stock in watchlist) {
      final prices = priceData[stock.symbol];
      if (prices != null && prices.length >= 30) {
        try {
          final recommendation = await generateRecommendation(
            stock: stock,
            prices: prices,
            highs: prices.map((p) => p * 1.01).toList(), // Estimate highs
            lows: prices.map((p) => p * 0.99).toList(), // Estimate lows
            userRiskProfile: riskProfile,
            userPortfolio: portfolio,
          );
          recommendations.add(recommendation);
        } catch (e) {
          print('Failed to generate recommendation for ${stock.symbol}: $e');
        }
      }
    }

    // Sort by confidence (highest first)
    recommendations.sort((a, b) => b.confidence.compareTo(a.confidence));

    return recommendations;
  }

  /// Calculates position size based on risk profile and volatility
  double calculatePositionSize({
    required RiskProfile riskProfile,
    required double volatility,
    required double availableCapital,
    double? currentPrice,
  }) {
    // Base position size by risk profile
    final baseSize = switch (riskProfile) {
      RiskProfile.conservative => 0.05, // 5%
      RiskProfile.moderate => 0.10, // 10%
      RiskProfile.aggressive => 0.20, // 20%
    };

    // Adjust for volatility (reduce size for high volatility)
    final volatilityFactor = volatility > 5 ? 0.5 : 1.0;

    // Calculate position size
    final positionSize = availableCapital * baseSize * volatilityFactor;

    // Limit to percentage of capital
    final maxPosition = availableCapital * 0.25; // Max 25% in single stock
    final minPosition = availableCapital * 0.01; // Min 1%

    return positionSize.clamp(minPosition, maxPosition);
  }
}
