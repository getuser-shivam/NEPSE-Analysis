class WatchlistItem {
  const WatchlistItem({
    required this.id,
    required this.symbol,
    required this.notes,
    required this.createdAt,
    required this.updatedAt,
  });

  final String id;
  final String symbol;
  final String? notes;
  final DateTime createdAt;
  final DateTime updatedAt;

  factory WatchlistItem.fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      notes: json['notes'] as String?,
      createdAt: DateTime.parse(json['createdAt'] as String),
      updatedAt: DateTime.parse(json['updatedAt'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'notes': notes,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }
}
