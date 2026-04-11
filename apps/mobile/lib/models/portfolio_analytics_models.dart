import 'package:freezed_annotation/freezed_annotation.dart';

part 'portfolio_analytics_models.freezed.dart';
part 'portfolio_analytics_models.g.dart';

@freezed
class PortfolioInsight with _$PortfolioInsight {
  const factory PortfolioInsight({
    required double totalInvestment,
    required double totalValue,
    required double totalGainLoss,
    required double totalReturnPct,
    required int numHoldings,
    required List<SectorAllocation> sectorAllocation,
    required DateTime generatedAt,
  }) = _PortfolioInsight;

  factory PortfolioInsight.fromJson(Map<String, dynamic> json) =>
      _$PortfolioInsightFromJson(json);
}

@freezed
class SectorAllocation with _$SectorAllocation {
  const factory SectorAllocation({
    required String name,
    required double value,
    required double percentage,
  }) = _SectorAllocation;

  factory SectorAllocation.fromJson(Map<String, dynamic> json) =>
      _$SectorAllocationFromJson(json);
}

@freezed
class PortfolioSnapshotModel with _$PortfolioSnapshotModel {
  const factory PortfolioSnapshotModel({
    required String id,
    required double totalValue,
    required double totalInvestment,
    required double totalGainLoss,
    required DateTime snapshotDate,
  }) = _PortfolioSnapshotModel;

  factory PortfolioSnapshotModel.fromJson(Map<String, dynamic> json) =>
      _$PortfolioSnapshotModelFromJson(json);
}
