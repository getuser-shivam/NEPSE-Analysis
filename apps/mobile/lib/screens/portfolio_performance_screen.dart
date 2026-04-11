import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';
import '../providers/portfolio_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/analytics/performance_line_chart.dart';
import '../widgets/common/glass_container.dart';

class PortfolioPerformanceScreen extends ConsumerWidget {
  const PortfolioPerformanceScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final portfolioAsync = ref.watch(portfolioProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Investment Analytics'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      extendBodyBehindAppBar: true,
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: SafeArea(
          child: portfolioAsync.when(
            loading: () => const Center(child: CircularProgressIndicator(color: AppColors.accent)),
            error: (err, _) => Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, color: AppColors.negative, size: 64),
                  const SizedBox(height: 16),
                  Text('Portfolio Sync Failed', style: const TextStyle(color: AppColors.negative, fontSize: 18)),
                  const SizedBox(height: 8),
                  Text(err.toString(), style: const TextStyle(color: AppColors.textSecondary, fontSize: 12)),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () => ref.refresh(portfolioProvider),
                    style: ElevatedButton.styleFrom(backgroundColor: AppColors.accent),
                    child: const Text('Retry', style: TextStyle(color: Colors.black)),
                  )
                ],
              )
            ),
            data: (portfolio) => _buildContent(context, portfolio),
          ),
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context, PortfolioOverview portfolio) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth > 800) {
          // --- WEB GRID LAYOUT ---
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 3,
                  child: Column(
                    children: [
                      _buildHistoryPreview(),
                      const SizedBox(height: 24),
                      Expanded(child: _buildHoldingsTable(portfolio.holdings, constraints)),
                    ],
                  )
                ),
                const SizedBox(width: 24),
                Expanded(
                  flex: 2,
                  child: Column(
                    children: [
                      _buildWealthSummary(portfolio.summary),
                      const SizedBox(height: 24),
                      _buildSectorAllocation(portfolio.holdings),
                    ],
                  )
                ),
              ],
            ),
          );
        }
        
        // --- MOBILE VERTICAL SCROLL LAYOUT ---
        return ListView(
          padding: const EdgeInsets.all(16.0),
          children: [
            _buildWealthSummary(portfolio.summary),
            const SizedBox(height: 24),
            const Text(
              'Growth Trend',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            _buildHistoryPreview(),
            const SizedBox(height: 24),
            const Text(
              'Sector Diversification',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            _buildSectorAllocation(portfolio.holdings),
            const SizedBox(height: 24),
            _buildHoldingsTable(portfolio.holdings, constraints),
          ],
        );
      }
    );
  }

  Widget _buildWealthSummary(PortfolioSummary summary) {
    return GlassContainer(
      child: Column(
        children: [
          const Text(
            'Total Net Worth',
            style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 8),
          Text(
            'NPR ${summary.currentValue.toStringAsFixed(2)}',
            style: const TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.w900,
              letterSpacing: -0.5,
              color: AppColors.primary,
            ),
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildMetric(
                'Investment',
                'NPR ${summary.totalInvestment.toStringAsFixed(0)}',
              ),
              _buildMetric(
                'Total Gain',
                'NPR ${summary.totalGainLoss.toStringAsFixed(0)}',
                color: summary.totalGainLoss >= 0
                    ? AppColors.positive
                    : AppColors.negative,
              ),
              _buildMetric(
                'Return',
                '${summary.totalReturnPct.toStringAsFixed(2)}%',
                color: summary.totalReturnPct >= 0
                    ? AppColors.positive
                    : AppColors.negative,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value, {Color? color}) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color ?? Colors.white,
          ),
        ),
      ],
    );
  }

  Widget _buildSectorAllocation(List<PortfolioHolding> holdings) {
    // Generate allocation logic based on dummy sector data as holding lacks explicit "sector" currently
    final Map<String, double> sectors = {};
    double total = 0;
    
    for (var h in holdings) {
       sectors[h.symbol] = (sectors[h.symbol] ?? 0) + (h.currentValue ?? h.buyPrice * h.shares);
       total += (h.currentValue ?? h.buyPrice * h.shares);
    }
    
    if (total == 0) return const GlassContainer(child: Center(child: Text("No holdings available")));

    return GlassContainer(
      padding: EdgeInsets.zero,
      child: Column(
        children: sectors.entries.map((entry) {
          final pct = (entry.value / total) * 100;
          return ListTile(
            title: Text(entry.key, style: const TextStyle(fontWeight: FontWeight.bold)),
            subtitle: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: pct / 100,
                backgroundColor: AppColors.accent.withValues(alpha: 0.1),
                color: AppColors.accent,
                minHeight: 6,
              ),
            ),
            trailing: Text(
              '${pct.toStringAsFixed(1)}%',
              style: const TextStyle(fontWeight: FontWeight.bold, color: AppColors.textSecondary),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildHistoryPreview() {
     // Performance Line Chart consumes old mock "Snapshot Models".
     // We will gracefully implement a generic "Chart Pending API Historicals" block to circumvent UI crashes on Web.
     return GlassContainer(
        padding: const EdgeInsets.only(top: 24, left: 8, right: 8),
        child: SizedBox(
          height: 350,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.query_stats, color: AppColors.accent.withValues(alpha: 0.5), size: 64),
              const SizedBox(height: 16),
              const Text('Historical Growth Projection', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              const Text('Awaiting backend historical snapshot APIs for rendering.', style: TextStyle(color: AppColors.textSecondary)),
            ]
          )
        ),
    );
  }
  
  Widget _buildHoldingsTable(List<PortfolioHolding> holdings, BoxConstraints constraints) {
    return GlassContainer(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text('Active Positions', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SingleChildScrollView(
                child: DataTable(
                  headingTextStyle: const TextStyle(color: AppColors.textSecondary, fontWeight: FontWeight.w600),
                  dataTextStyle: const TextStyle(color: Colors.white, fontSize: 14),
                  columnSpacing: constraints.maxWidth > 800 ? 56 : 24,
                  columns: const [
                    DataColumn(label: Text('Asset')),
                    DataColumn(label: Text('Shares')),
                    DataColumn(label: Text('Value')),
                    DataColumn(label: Text('P&L')),
                  ],
                  rows: holdings.map((h) {
                    final isPositive = (h.gainLoss ?? 0) >= 0;
                    return DataRow(cells: [
                      DataCell(Text(h.symbol, style: const TextStyle(fontWeight: FontWeight.bold))),
                      DataCell(Text('${h.shares.toStringAsFixed(0)} Units')),
                      DataCell(Text((h.currentValue ?? (h.buyPrice * h.shares)).toStringAsFixed(2))),
                      DataCell(Text('${isPositive ? '+' : ''}${(h.gainLoss ?? 0).toStringAsFixed(2)}', style: TextStyle(color: isPositive ? AppColors.positive : AppColors.negative))),
                    ]);
                  }).toList(),
                ),
              ),
            ),
          )
        ],
      )
    );
  }
}
