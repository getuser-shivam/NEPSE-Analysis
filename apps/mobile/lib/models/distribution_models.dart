import 'package:freezed_annotation/freezed_annotation.dart';

part 'distribution_models.freezed.dart';
part 'distribution_models.g.dart';

@freezed
class Order with _$Order {
  const factory Order({
    required String id,
    required String userId,
    required String status,
    required double totalAmount,
    @Default('NPR') String currency,
    String? shippingAddress,
    required List<OrderItem> items,
    Shipment? shipment,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) = _Order;

  factory Order.fromJson(Map<String, dynamic> json) => _$OrderFromJson(json);
}

@freezed
class OrderItem with _$OrderItem {
  const factory OrderItem({
    required String id,
    required String productId,
    required int quantity,
    required double price,
  }) = _OrderItem;

  factory OrderItem.fromJson(Map<String, dynamic> json) => _$OrderItemFromJson(json);
}

@freezed
class Shipment with _$Shipment {
  const factory Shipment({
    required String id,
    String? trackingNumber,
    String? carrier,
    required String status,
    DateTime? estimatedArrival,
    DateTime? shippedAt,
    DateTime? deliveredAt,
  }) = _Shipment;

  factory Shipment.fromJson(Map<String, dynamic> json) => _$ShipmentFromJson(json);
}

@freezed
class InventoryForecast with _$InventoryForecast {
  const factory InventoryForecast({
    required int currentStock,
    required String avgDailySales,
    required String daysRemaining,
    required bool belowThreshold,
  }) = _InventoryForecast;

  factory InventoryForecast.fromJson(Map<String, dynamic> json) => _$InventoryForecastFromJson(json);
}
