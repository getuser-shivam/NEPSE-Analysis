import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../../models/portfolio_analytics_models.dart';
import '../../theme/app_theme.dart';

class PerformanceLineChart extends StatelessWidget {
  final List<PortfolioSnapshotModel> snapshots;

  const PerformanceLineChart({super.key, required this.snapshots});

  @override
  Widget build(BuildContext context) {
    if (snapshots.isEmpty) {
      return const Center(child: Text('Not enough data points'));
    }

    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: false),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        minX: 0,
        maxX: (snapshots.length - 1).toDouble(),
        minY: _getMinY(),
        maxY: _getMaxY() * 1.05,
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (_) => AppColors.surface,
            tooltipRoundedRadius: 12,
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                final snapshot = snapshots[spot.x.toInt()];
                return LineTooltipItem(
                  '${snapshot.snapshotDate.toString().split(' ')[0]}\n',
                  const TextStyle(color: AppColors.textSecondary, fontSize: 10),
                  children: [
                    TextSpan(
                      text: 'NPR ${spot.y.toStringAsFixed(0)}',
                      style: const TextStyle(color: AppColors.accent, fontWeight: FontWeight.bold),
                    ),
                  ],
                );
              }).toList();
            },
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: snapshots.asMap().entries.map((e) {
              return FlSpot(e.key.toDouble(), e.value.totalValue);
            }).toList(),
            isCurved: true,
            color: AppColors.accent,
            barWidth: 3,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AppColors.accent.withValues(alpha: 0.2),
                  AppColors.accent.withValues(alpha: 0.0),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  double _getMinY() {
    return snapshots.map((s) => s.totalValue).reduce((a, b) => a < b ? a : b) * 0.95;
  }

  double _getMaxY() {
    return snapshots.map((s) => s.totalValue).reduce((a, b) => a > b ? a : b);
  }
}
