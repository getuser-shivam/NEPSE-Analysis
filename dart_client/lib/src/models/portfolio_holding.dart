class PortfolioHolding {
  const PortfolioHolding({
    required this.id,
    required this.symbol,
    required this.shares,
    required this.buyPrice,
    required this.currentPrice,
    required this.notes,
    required this.lastUpdated,
    required this.createdAt,
    required this.updatedAt,
  });

  final String id;
  final String symbol;
  final double shares;
  final double buyPrice;
  final double? currentPrice;
  final String? notes;
  final DateTime? lastUpdated;
  final DateTime createdAt;
  final DateTime updatedAt;

  factory PortfolioHolding.fromJson(Map<String, dynamic> json) {
    return PortfolioHolding(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      shares: (json['shares'] as num).toDouble(),
      buyPrice: (json['buyPrice'] as num).toDouble(),
      currentPrice: (json['currentPrice'] as num?)?.toDouble(),
      notes: json['notes'] as String?,
      lastUpdated: json['lastUpdated'] == null
          ? null
          : DateTime.parse(json['lastUpdated'] as String),
      createdAt: DateTime.parse(json['createdAt'] as String),
      updatedAt: DateTime.parse(json['updatedAt'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'shares': shares,
      'buyPrice': buyPrice,
      'currentPrice': currentPrice,
      'notes': notes,
      'lastUpdated': lastUpdated?.toIso8601String(),
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }
}
