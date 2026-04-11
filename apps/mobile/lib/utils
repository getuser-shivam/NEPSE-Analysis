/// Technical Indicator Calculations
/// 
/// This library provides comprehensive technical analysis calculations
/// for stock market data including moving averages, momentum indicators,
/// volatility measures, and trend detection.
library nepse_analysis_calculations;

import 'dart:math';

/// Result container for technical indicator calculations.
class TechnicalIndicators {
  /// Relative Strength Index (14-period)
  final double? rsi14;
  
  /// Relative Strength Index (7-period)
  final double? rsi7;
  
  /// Simple Moving Average (20-period)
  final double? sma20;
  
  /// Simple Moving Average (50-period)
  final double? sma50;
  
  /// Exponential Moving Average (12-period)
  final double? ema12;
  
  /// Exponential Moving Average (26-period)
  final double? ema26;
  
  /// MACD line value
  final double? macdLine;
  
  /// MACD signal line
  final double? macdSignal;
  
  /// MACD histogram
  final double? macdHistogram;
  
  /// Bollinger Bands upper band
  final double? bbUpper;
  
  /// Bollinger Bands middle band (SMA)
  final double? bbMiddle;
  
  /// Bollinger Bands lower band
  final double? bbLower;
  
  /// Stochastic %K
  final double? stochK;
  
  /// Stochastic %D
  final double? stochD;
  
  /// Williams %R
  final double? williamsR;
  
  /// Average True Range
  final double? atr;
  
  /// Trend strength (0-100)
  final double trendStrength;
  
  /// Volatility measure
  final double volatility;
  
  /// Trading signal (buy, sell, hold)
  final String signal;

  const TechnicalIndicators({
    this.rsi14,
    this.rsi7,
    this.sma20,
    this.sma50,
    this.ema12,
    this.ema26,
    this.macdLine,
    this.macdSignal,
    this.macdHistogram,
    this.bbUpper,
    this.bbMiddle,
    this.bbLower,
    this.stochK,
    this.stochD,
    this.williamsR,
    this.atr,
    this.trendStrength = 0,
    this.volatility = 0,
    this.signal = 'hold',
  });

  @override
  String toString() {
    return 'TechnicalIndicators(RSI14: $rsi14, Signal: $signal)';
  }
}

/// Calculates the Simple Moving Average (SMA).
/// 
/// [data] - List of price values
/// [period] - Number of periods to average
/// Returns a list of SMA values aligned with input data.
List<double> calculateSMA(List<double> data, int period) {
  if (data.length < period) return [];
  
  final result = <double>[];
  for (int i = period - 1; i < data.length; i++) {
    double sum = 0;
    for (int j = 0; j < period; j++) {
      sum += data[i - j];
    }
    result.add(sum / period);
  }
  return result;
}

/// Calculates the Exponential Moving Average (EMA).
/// 
/// [data] - List of price values
/// [period] - Number of periods for calculation
/// Returns a list of EMA values.
List<double> calculateEMA(List<double> data, int period) {
  if (data.length < period) return [];
  
  final multiplier = 2 / (period + 1);
  final result = <double>[];
  
  // First EMA is the SMA of the first 'period' values
  double ema = data.sublist(0, period).reduce((a, b) => a + b) / period;
  result.add(ema);
  
  // Calculate subsequent EMAs
  for (int i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
    result.add(ema);
  }
  
  return result;
}

/// Calculates the Relative Strength Index (RSI).
/// 
/// [prices] - List of closing prices
/// [period] - RSI period (default 14)
/// Returns the RSI value (0-100).
double calculateRSI(List<double> prices, {int period = 14}) {
  if (prices.length < period + 1) return 50; // Neutral
  
  double gains = 0;
  double losses = 0;
  
  // Calculate initial averages
  for (int i = 1; i <= period; i++) {
    double change = prices[i] - prices[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses += change.abs();
    }
  }
  
  double avgGain = gains / period;
  double avgLoss = losses / period;
  
  // Calculate smoothed averages
  for (int i = period + 1; i < prices.length; i++) {
    double change = prices[i] - prices[i - 1];
    if (change > 0) {
      avgGain = (avgGain * (period - 1) + change) / period;
      avgLoss = (avgLoss * (period - 1)) / period;
    } else {
      avgGain = (avgGain * (period - 1)) / period;
      avgLoss = (avgLoss * (period - 1) + change.abs()) / period;
    }
  }
  
  if (avgLoss == 0) return 100;
  double rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

/// Calculates MACD (Moving Average Convergence Divergence).
/// 
/// [prices] - List of closing prices
/// Returns MACD values including line, signal, and histogram.
({List<double> macdLine, List<double> signalLine, List<double> histogram}) calculateMACD(
  List<double> prices, {
  int fastPeriod = 12,
  int slowPeriod = 26,
  int signalPeriod = 9,
}) {
  if (prices.length < slowPeriod + signalPeriod) {
    return (macdLine: [], signalLine: [], histogram: []);
  }
  
  final fastEMA = calculateEMA(prices, fastPeriod);
  final slowEMA = calculateEMA(prices, slowPeriod);
  
  // Align EMAs (they start at different indices)
  final startIndex = slowPeriod - fastPeriod;
  final alignedFast = fastEMA.sublist(startIndex);
  
  // Calculate MACD line
  final macdLine = <double>[];
  for (int i = 0; i < alignedFast.length && i < slowEMA.length; i++) {
    macdLine.add(alignedFast[i] - slowEMA[i]);
  }
  
  // Calculate signal line (EMA of MACD)
  final signalLine = calculateEMA(macdLine, signalPeriod);
  
  // Calculate histogram
  final histogram = <double>[];
  final startIdx = macdLine.length - signalLine.length;
  for (int i = 0; i < signalLine.length; i++) {
    histogram.add(macdLine[startIdx + i] - signalLine[i]);
  }
  
  return (macdLine: macdLine, signalLine: signalLine, histogram: histogram);
}

/// Calculates Bollinger Bands.
/// 
/// [prices] - List of closing prices
/// [period] - Period for SMA (default 20)
/// [stdDev] - Number of standard deviations (default 2)
/// Returns upper, middle, and lower bands.
({List<double> upperBand, List<double> middleBand, List<double> lowerBand}) calculateBollingerBands(
  List<double> prices, {
  int period = 20,
  int stdDev = 2,
}) {
  if (prices.length < period) {
    return (upperBand: [], middleBand: [], lowerBand: []);
  }
  
  final sma = calculateSMA(prices, period);
  final upperBand = <double>[];
  final lowerBand = <double>[];
  
  for (int i = period - 1; i < prices.length; i++) {
    // Calculate standard deviation for this window
    double sum = 0;
    for (int j = 0; j < period; j++) {
      sum += pow(prices[i - j] - sma[i - period + 1], 2);
    }
    double standardDeviation = sqrt(sum / period);
    
    upperBand.add(sma[i - period + 1] + (standardDeviation * stdDev));
    lowerBand.add(sma[i - period + 1] - (standardDeviation * stdDev));
  }
  
  return (upperBand: upperBand, middleBand: sma, lowerBand: lowerBand);
}

/// Calculates Stochastic Oscillator.
/// 
/// [highs] - List of high prices
/// [lows] - List of low prices
/// [closes] - List of closing prices
/// [kPeriod] - %K period (default 14)
/// [dPeriod] - %D period (default 3)
/// Returns %K and %D values.
({List<double> kValues, List<double> dValues}) calculateStochastic(
  List<double> highs,
  List<double> lows,
  List<double> closes, {
  int kPeriod = 14,
  int dPeriod = 3,
}) {
  if (highs.length < kPeriod || lows.length < kPeriod || closes.length < kPeriod) {
    return (kValues: [], dValues: []);
  }
  
  final kValues = <double>[];
  
  for (int i = kPeriod - 1; i < closes.length; i++) {
    double highestHigh = highs.sublist(i - kPeriod + 1, i + 1).reduce(max);
    double lowestLow = lows.sublist(i - kPeriod + 1, i + 1).reduce(min);
    
    if (highestHigh == lowestLow) {
      kValues.add(50); // Neutral when no range
    } else {
      kValues.add(((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100);
    }
  }
  
  final dValues = calculateSMA(kValues, dPeriod);
  
  return (kValues: kValues, dValues: dValues);
}

/// Calculates Williams %R.
/// 
/// [highs] - List of high prices
/// [lows] - List of low prices
/// [closes] - List of closing prices
/// [period] - Lookback period (default 14)
/// Returns Williams %R values (-100 to 0).
List<double> calculateWilliamsR(
  List<double> highs,
  List<double> lows,
  List<double> closes, {
  int period = 14,
}) {
  if (highs.length < period || lows.length < period || closes.length < period) {
    return [];
  }
  
  final result = <double>[];
  
  for (int i = period - 1; i < closes.length; i++) {
    double highestHigh = highs.sublist(i - period + 1, i + 1).reduce(max);
    double lowestLow = lows.sublist(i - period + 1, i + 1).reduce(min);
    
    if (highestHigh == lowestLow) {
      result.add(-50); // Neutral
    } else {
      result.add(((highestHigh - closes[i]) / (highestHigh - lowestLow)) * -100);
    }
  }
  
  return result;
}

/// Calculates Average True Range (ATR).
/// 
/// [highs] - List of high prices
/// [lows] - List of low prices
/// [closes] - List of closing prices
/// [period] - ATR period (default 14)
/// Returns ATR values.
List<double> calculateATR(
  List<double> highs,
  List<double> lows,
  List<double> closes, {
  int period = 14,
}) {
  if (highs.length < period + 1 || lows.length < period + 1 || closes.length < period + 1) {
    return [];
  }
  
  final trueRanges = <double>[];
  
  for (int i = 1; i < highs.length; i++) {
    double tr1 = highs[i] - lows[i];
    double tr2 = (highs[i] - closes[i - 1]).abs();
    double tr3 = (lows[i] - closes[i - 1]).abs();
    trueRanges.add([tr1, tr2, tr3].reduce(max));
  }
  
  return calculateEMA(trueRanges, period);
}

/// Calculates percentage change between two values.
/// 
/// [current] - Current value
/// [previous] - Previous value
/// Returns percentage change.
double calculatePercentChange(double current, double previous) {
  if (previous == 0) return 0;
  return ((current - previous) / previous) * 100;
}

/// Detects trend direction and strength.
/// 
/// [prices] - List of price values
/// [period] - Analysis period (default 20)
/// Returns trend strength (positive = uptrend, negative = downtrend).
double detectTrend(List<double> prices, {int period = 20}) {
  if (prices.length < period) return 0;
  
  final recent = prices.sublist(prices.length - period);
  final sma = recent.reduce((a, b) => a + b) / period;
  
  // Calculate price position relative to SMA
  double trendStrength = 0;
  for (int i = 1; i < recent.length; i++) {
    if (recent[i] > recent[i - 1]) {
      trendStrength++;
    } else if (recent[i] < recent[i - 1]) {
      trendStrength--;
    }
  }
  
  return trendStrength / period; // Normalized to -1 to 1
}

/// Calculates volatility as standard deviation of returns.
/// 
/// [prices] - List of price values
/// Returns volatility percentage.
double calculateVolatility(List<double> prices) {
  if (prices.length < 2) return 0;
  
  // Calculate returns
  final returns = <double>[];
  for (int i = 1; i < prices.length; i++) {
    returns.add((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  
  // Calculate mean
  final mean = returns.reduce((a, b) => a + b) / returns.length;
  
  // Calculate variance
  double variance = 0;
  for (final ret in returns) {
    variance += pow(ret - mean, 2);
  }
  variance /= returns.length;
  
  // Return standard deviation as percentage
  return sqrt(variance) * 100;
}

/// Comprehensive technical analysis combining all indicators.
/// 
/// [prices] - List of closing prices
/// [highs] - List of high prices
/// [lows] - List of low prices
/// Returns complete technical indicators analysis.
TechnicalIndicators analyzeTechnicals({
  required List<double> prices,
  required List<double> highs,
  required List<double> lows,
}) {
  if (prices.isEmpty || highs.isEmpty || lows.isEmpty) {
    return const TechnicalIndicators();
  }
  
  // Calculate individual indicators
  final rsi14 = calculateRSI(prices, period: 14);
  final rsi7 = calculateRSI(prices, period: 7);
  
  final sma20List = calculateSMA(prices, 20);
  final sma50List = calculateSMA(prices, 50);
  final sma20 = sma20List.isNotEmpty ? sma20List.last : null;
  final sma50 = sma50List.isNotEmpty ? sma50List.last : null;
  
  final ema12List = calculateEMA(prices, 12);
  final ema26List = calculateEMA(prices, 26);
  final ema12 = ema12List.isNotEmpty ? ema12List.last : null;
  final ema26 = ema26List.isNotEmpty ? ema26List.last : null;
  
  final macd = calculateMACD(prices);
  final macdLine = macd.macdLine.isNotEmpty ? macd.macdLine.last : null;
  final macdSignal = macd.signalLine.isNotEmpty ? macd.signalLine.last : null;
  final macdHistogram = macd.histogram.isNotEmpty ? macd.histogram.last : null;
  
  final bb = calculateBollingerBands(prices);
  final bbUpper = bb.upperBand.isNotEmpty ? bb.upperBand.last : null;
  final bbMiddle = bb.middleBand.isNotEmpty ? bb.middleBand.last : null;
  final bbLower = bb.lowerBand.isNotEmpty ? bb.lowerBand.last : null;
  
  final stoch = calculateStochastic(highs, lows, prices);
  final stochK = stoch.kValues.isNotEmpty ? stoch.kValues.last : null;
  final stochD = stoch.dValues.isNotEmpty ? stoch.dValues.last : null;
  
  final williamsRList = calculateWilliamsR(highs, lows, prices);
  final williamsR = williamsRList.isNotEmpty ? williamsRList.last : null;
  
  final atrList = calculateATR(highs, lows, prices);
  final atr = atrList.isNotEmpty ? atrList.last : null;
  
  final trendStrength = detectTrend(prices) * 100;
  final volatility = calculateVolatility(prices);
  
  // Generate signal based on multiple indicators
  final signal = _generateSignal(
    rsi14: rsi14,
    macdHistogram: macdHistogram,
    price: prices.last,
    bbUpper: bbUpper,
    bbLower: bbLower,
    sma20: sma20,
    sma50: sma50,
  );
  
  return TechnicalIndicators(
    rsi14: rsi14,
    rsi7: rsi7,
    sma20: sma20,
    sma50: sma50,
    ema12: ema12,
    ema26: ema26,
    macdLine: macdLine,
    macdSignal: macdSignal,
    macdHistogram: macdHistogram,
    bbUpper: bbUpper,
    bbMiddle: bbMiddle,
    bbLower: bbLower,
    stochK: stochK,
    stochD: stochD,
    williamsR: williamsR,
    atr: atr,
    trendStrength: trendStrength,
    volatility: volatility,
    signal: signal,
  );
}

/// Generates trading signal based on technical indicators.
String _generateSignal({
  required double rsi14,
  double? macdHistogram,
  required double price,
  double? bbUpper,
  double? bbLower,
  double? sma20,
  double? sma50,
}) {
  int buyScore = 0;
  int sellScore = 0;
  
  // RSI signals
  if (rsi14 < 30) buyScore += 2;
  if (rsi14 > 70) sellScore += 2;
  if (rsi14 < 40) buyScore += 1;
  if (rsi14 > 60) sellScore += 1;
  
  // MACD signals
  if (macdHistogram != null) {
    if (macdHistogram > 0) buyScore += 1;
    if (macdHistogram < 0) sellScore += 1;
  }
  
  // Bollinger Bands signals
  if (bbUpper != null && price > bbUpper) sellScore += 2;
  if (bbLower != null && price < bbLower) buyScore += 2;
  
  // Moving average signals
  if (sma20 != null && sma50 != null) {
    if (sma20 > sma50) buyScore += 1;
    if (sma20 < sma50) sellScore += 1;
    if (price > sma20) buyScore += 1;
    if (price < sma20) sellScore += 1;
  }
  
  // Generate final signal
  if (buyScore >= 4) return 'strong_buy';
  if (buyScore >= 2) return 'buy';
  if (sellScore >= 4) return 'strong_sell';
  if (sellScore >= 2) return 'sell';
  return 'hold';
}
