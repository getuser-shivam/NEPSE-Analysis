class PriceSnapshot {
  const PriceSnapshot({
    required this.id,
    required this.symbol,
    required this.tradeDate,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
    required this.source,
    required this.createdAt,
    required this.updatedAt,
  });

  final String id;
  final String symbol;
  final DateTime tradeDate;
  final double open;
  final double high;
  final double low;
  final double close;
  final double volume;
  final String source;
  final DateTime createdAt;
  final DateTime updatedAt;

  factory PriceSnapshot.fromJson(Map<String, dynamic> json) {
    return PriceSnapshot(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      tradeDate: DateTime.parse(json['tradeDate'] as String),
      open: (json['open'] as num).toDouble(),
      high: (json['high'] as num).toDouble(),
      low: (json['low'] as num).toDouble(),
      close: (json['close'] as num).toDouble(),
      volume: (json['volume'] as num).toDouble(),
      source: json['source'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
      updatedAt: DateTime.parse(json['updatedAt'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'tradeDate': tradeDate.toIso8601String(),
      'open': open,
      'high': high,
      'low': low,
      'close': close,
      'volume': volume,
      'source': source,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }
}
