import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/portfolio_analytics_models.dart';
import '../services/portfolio_analytics_service.dart';
import '../theme/app_theme.dart';
import '../widgets/analytics/performance_line_chart.dart';
import '../widgets/common/glass_container.dart';
import '../widgets/common/status_pill.dart';

final portfolioAnalyticsServiceProvider = Provider((ref) => PortfolioAnalyticsService());

final portfolioInsightProvider = FutureProvider<PortfolioInsight>((ref) async {
  return ref.watch(portfolioAnalyticsServiceProvider).getInsights();
});

final portfolioHistoryProvider = FutureProvider<List<PortfolioSnapshotModel>>((ref) async {
  return ref.watch(portfolioAnalyticsServiceProvider).getHistory();
});

class PortfolioPerformanceScreen extends ConsumerWidget {
  const PortfolioPerformanceScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final insightsAsync = ref.watch(portfolioInsightProvider);
    final historyAsync = ref.watch(portfolioHistoryProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Investment Analytics')),
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: insightsAsync.when(
          data: (insight) => _buildContent(context, insight, historyAsync),
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (err, _) => Center(child: Text('Error: $err')),
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context, PortfolioInsight insight, AsyncValue<List<PortfolioSnapshotModel>> historyAsync) {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        _buildWealthSummary(insight),
        const SizedBox(height: 24),
        const Text('Sector Diversification', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        _buildSectorAllocation(insight.sectorAllocation),
        const SizedBox(height: 24),
        const Text('Growth Trend', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        _buildHistoryPreview(historyAsync),
      ],
    );
  }

  Widget _buildWealthSummary(PortfolioInsight insight) {
    return GlassContainer(
      child: Column(
        children: [
          const Text('Total Net Worth', style: TextStyle(fontSize: 14, color: AppColors.textSecondary)),
          const SizedBox(height: 8),
          Text(
            'NPR ${insight.totalValue.toStringAsFixed(2)}',
            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: AppColors.primary),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildMetric('Investment', 'NPR ${insight.totalInvestment.toStringAsFixed(0)}'),
              _buildMetric('Total Gain', 'NPR ${insight.totalGainLoss.toStringAsFixed(0)}', 
                color: insight.totalGainLoss >= 0 ? AppColors.positive : AppColors.negative),
              _buildMetric('Return', '${insight.totalReturnPct.toStringAsFixed(2)}%',
                color: insight.totalReturnPct >= 0 ? AppColors.positive : AppColors.negative),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value, {Color? color}) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
        const SizedBox(height: 4),
        Text(value, style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: color)),
      ],
    );
  }

  Widget _buildSectorAllocation(List<SectorAllocation> allocation) {
    return GlassContainer(
      padding: EdgeInsets.zero,
      child: Column(
        children: allocation.map((item) {
          return ListTile(
            title: Text(item.name),
            subtitle: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: item.percentage / 100,
                backgroundColor: AppColors.accent.withValues(alpha: 0.1),
                color: AppColors.accent,
                minHeight: 6,
              ),
            ),
            trailing: Text('${item.percentage.toStringAsFixed(1)}%', 
              style: const TextStyle(fontWeight: FontWeight.bold)),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildHistoryPreview(AsyncValue<List<PortfolioSnapshotModel>> historyAsync) {
    return historyAsync.when(
      data: (history) => GlassContainer(
        padding: const EdgeInsets.only(top: 24, left: 8, right: 8),
        child: SizedBox(
          height: 200,
          child: PerformanceLineChart(snapshots: history),
        ),
      ),
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (err, _) => Text('History error: $err'),
    );
  }
}
