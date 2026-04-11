/// Alert Manager Service
///
/// Manages price alerts for stocks including creation, monitoring,
/// and triggering of alerts based on various conditions.
library nepse_analysis_alert_manager;

import 'dart:async';
import '../models/stock.dart';

/// Types of price alerts available
enum AlertType {
  priceAbove,
  priceBelow,
  percentChange,
  volumeSpike,
  indicatorSignal,
}

/// Current status of an alert
enum AlertStatus { active, triggered, disabled, expired }

/// Represents a single price alert
class PriceAlert {
  final String id;
  final String symbol;
  final AlertType type;
  final double targetValue;
  final String? note;
  AlertStatus status;
  DateTime? triggeredAt;
  double? triggeredPrice;
  final DateTime createdAt;
  DateTime updatedAt;

  PriceAlert({
    required this.id,
    required this.symbol,
    required this.type,
    required this.targetValue,
    this.note,
    this.status = AlertStatus.active,
    this.triggeredAt,
    this.triggeredPrice,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) : createdAt = createdAt ?? DateTime.now(),
       updatedAt = updatedAt ?? DateTime.now();

  /// Checks if the alert should be triggered based on current price
  bool shouldTrigger(double currentPrice, double previousPrice) {
    if (status != AlertStatus.active) return false;

    switch (type) {
      case AlertType.priceAbove:
        return currentPrice >= targetValue && previousPrice < targetValue;
      case AlertType.priceBelow:
        return currentPrice <= targetValue && previousPrice > targetValue;
      case AlertType.percentChange:
        final change = ((currentPrice - previousPrice) / previousPrice) * 100;
        return change.abs() >= targetValue.abs();
      case AlertType.volumeSpike:
        // Volume spike requires additional data
        return false;
      case AlertType.indicatorSignal:
        // Indicator signal requires technical analysis
        return false;
    }
  }

  /// Marks the alert as triggered
  void trigger(double price) {
    status = AlertStatus.triggered;
    triggeredAt = DateTime.now();
    triggeredPrice = price;
    updatedAt = DateTime.now();
  }

  /// Checks if the alert has expired
  bool isExpired(DateTime now) {
    // Alerts expire after 30 days of being triggered or 90 days of being active
    final maxAge = status == AlertStatus.triggered
        ? const Duration(days: 30)
        : const Duration(days: 90);
    return now.difference(createdAt) > maxAge;
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'type': type.name,
      'targetValue': targetValue,
      'note': note,
      'status': status.name,
      'triggeredAt': triggeredAt?.toIso8601String(),
      'triggeredPrice': triggeredPrice,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
    };
  }

  factory PriceAlert.fromJson(Map<String, dynamic> json) {
    return PriceAlert(
      id: json['id'],
      symbol: json['symbol'],
      type: AlertType.values.byName(json['type']),
      targetValue: json['targetValue'],
      note: json['note'],
      status: AlertStatus.values.byName(json['status']),
      triggeredAt: json['triggeredAt'] != null
          ? DateTime.parse(json['triggeredAt'])
          : null,
      triggeredPrice: json['triggeredPrice'],
      createdAt: DateTime.parse(json['createdAt']),
      updatedAt: DateTime.parse(json['updatedAt']),
    );
  }
}

/// Manages all price alerts for stocks
class AlertManager {
  final Map<String, PriceAlert> _alerts = {};
  final _alertController = StreamController<PriceAlert>.broadcast();
  Timer? _cleanupTimer;

  /// Stream of triggered alerts
  Stream<PriceAlert> get alertStream => _alertController.stream;

  AlertManager() {
    // Periodic cleanup of expired alerts
    _cleanupTimer = Timer.periodic(
      const Duration(hours: 1),
      (_) => _cleanupExpiredAlerts(),
    );
  }

  /// Creates a new price alert
  PriceAlert createPriceAlert({
    required String symbol,
    required AlertType type,
    required double targetValue,
    String? note,
  }) {
    final alert = PriceAlert(
      id: _generateId(),
      symbol: symbol,
      type: type,
      targetValue: targetValue,
      note: note,
    );
    _alerts[alert.id] = alert;
    return alert;
  }

  /// Creates a percentage change alert
  PriceAlert createPercentChangeAlert({
    required String symbol,
    required double percentChange,
  }) {
    return createPriceAlert(
      symbol: symbol,
      type: AlertType.percentChange,
      targetValue: percentChange.abs(),
      note: 'Alert when price changes by ${percentChange.abs()}%',
    );
  }

  /// Creates a threshold alert (above or below a price)
  PriceAlert createThresholdAlert({
    required String symbol,
    required double price,
    required bool isAbove,
  }) {
    return createPriceAlert(
      symbol: symbol,
      type: isAbove ? AlertType.priceAbove : AlertType.priceBelow,
      targetValue: price,
      note: isAbove
          ? 'Alert when price goes above $price'
          : 'Alert when price goes below $price',
    );
  }

  /// Gets all alerts
  List<PriceAlert> getAllAlerts() => List.unmodifiable(_alerts.values);

  /// Gets active alerts only
  List<PriceAlert> getActiveAlerts() =>
      _alerts.values.where((a) => a.status == AlertStatus.active).toList();

  /// Gets triggered alerts
  List<PriceAlert> getTriggeredAlerts() =>
      _alerts.values.where((a) => a.status == AlertStatus.triggered).toList();

  /// Gets alerts for a specific symbol
  List<PriceAlert> getAlertsForSymbol(String symbol) =>
      _alerts.values.where((a) => a.symbol == symbol).toList();

  /// Gets a single alert by ID
  PriceAlert? getAlert(String id) => _alerts[id];

  /// Updates an alert
  void updateAlert(String id, {AlertStatus? status}) {
    final alert = _alerts[id];
    if (alert != null && status != null) {
      alert.status = status;
      alert.updatedAt = DateTime.now();
    }
  }

  /// Deletes an alert
  void deleteAlert(String id) {
    _alerts.remove(id);
  }

  /// Checks all alerts against current prices and triggers if needed
  void checkAlerts(
    Map<String, double> currentPrices,
    Map<String, double> previousPrices,
  ) {
    for (final alert in getActiveAlerts()) {
      final currentPrice = currentPrices[alert.symbol];
      final previousPrice = previousPrices[alert.symbol];

      if (currentPrice != null && previousPrice != null) {
        if (alert.shouldTrigger(currentPrice, previousPrice)) {
          alert.trigger(currentPrice);
          _alertController.add(alert);
        }
      }
    }
  }

  /// Clears all triggered alerts older than the specified duration
  int clearOldTriggeredAlerts(Duration maxAge) {
    final now = DateTime.now();
    final toRemove = _alerts.values
        .where(
          (a) =>
              a.status == AlertStatus.triggered &&
              a.triggeredAt != null &&
              now.difference(a.triggeredAt!) > maxAge,
        )
        .map((a) => a.id)
        .toList();

    for (final id in toRemove) {
      _alerts.remove(id);
    }

    return toRemove.length;
  }

  /// Exports all alerts to JSON
  List<Map<String, dynamic>> exportToJson() {
    return _alerts.values.map((a) => a.toJson()).toList();
  }

  /// Imports alerts from JSON
  void importFromJson(List<Map<String, dynamic>> jsonList) {
    for (final json in jsonList) {
      final alert = PriceAlert.fromJson(json);
      _alerts[alert.id] = alert;
    }
  }

  /// Cleanup expired alerts
  void _cleanupExpiredAlerts() {
    final now = DateTime.now();
    final expiredIds = _alerts.values
        .where((a) => a.isExpired(now))
        .map((a) => a.id)
        .toList();

    for (final id in expiredIds) {
      _alerts.remove(id);
    }
  }

  String _generateId() {
    return '${DateTime.now().millisecondsSinceEpoch}_${_alerts.length}';
  }

  /// Disposes the manager and cleans up resources
  void dispose() {
    _cleanupTimer?.cancel();
    _alertController.close();
  }
}

/// Statistics about alerts
class AlertStatistics {
  final int totalAlerts;
  final int activeAlerts;
  final int triggeredAlerts;
  final Map<String, int> alertsBySymbol;
  final Map<AlertType, int> alertsByType;

  AlertStatistics({
    required this.totalAlerts,
    required this.activeAlerts,
    required this.triggeredAlerts,
    required this.alertsBySymbol,
    required this.alertsByType,
  });

  factory AlertStatistics.fromAlerts(List<PriceAlert> alerts) {
    final bySymbol = <String, int>{};
    final byType = <AlertType, int>{};

    for (final alert in alerts) {
      bySymbol[alert.symbol] = (bySymbol[alert.symbol] ?? 0) + 1;
      byType[alert.type] = (byType[alert.type] ?? 0) + 1;
    }

    return AlertStatistics(
      totalAlerts: alerts.length,
      activeAlerts: alerts.where((a) => a.status == AlertStatus.active).length,
      triggeredAlerts: alerts
          .where((a) => a.status == AlertStatus.triggered)
          .length,
      alertsBySymbol: bySymbol,
      alertsByType: byType,
    );
  }
}
