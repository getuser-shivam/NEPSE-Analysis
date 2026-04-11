import 'package:freezed_annotation/freezed_annotation.dart';

part 'watchlist_models.freezed.dart';
part 'watchlist_models.g.dart';

@freezed
class Watchlist with _$Watchlist {
  const factory Watchlist({
    required String id,
    required String name,
    String? description,
    @Default(false) bool isDefault,
    required List<WatchlistItem> items,
    DateTime? createdAt,
  }) = _Watchlist;

  factory Watchlist.fromJson(Map<String, dynamic> json) =>
      _$WatchlistFromJson(json);
}

@freezed
class WatchlistItem with _$WatchlistItem {
  const factory WatchlistItem({
    required String id,
    required String watchlistId,
    required String stockId,
    String? notes,
    double? alertPrice,
    StockSummary? stock,
  }) = _WatchlistItem;

  factory WatchlistItem.fromJson(Map<String, dynamic> json) =>
      _$WatchlistItemFromJson(json);
}

@freezed
class StockSummary with _$StockSummary {
  const factory StockSummary({
    required String symbol,
    required String name,
    double? lastPrice,
    double? change,
    double? changePercent,
  }) = _StockSummary;

  factory StockSummary.fromJson(Map<String, dynamic> json) =>
      _$StockSummaryFromJson(json);
}

@freezed
class PriceAlert with _$PriceAlert {
  const factory PriceAlert({
    required String id,
    required String stockId,
    required String alertType,
    required double targetValue,
    @Default('ACTIVE') String status,
    StockSummary? stock,
    DateTime? triggeredAt,
  }) = _PriceAlert;

  factory PriceAlert.fromJson(Map<String, dynamic> json) =>
      _$PriceAlertFromJson(json);
}
