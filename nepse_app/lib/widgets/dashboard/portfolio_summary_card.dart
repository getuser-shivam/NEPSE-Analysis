import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:nepse_client/nepse_client.dart';

import '../../theme/app_theme.dart';
import '../common/app_panel.dart';

class PortfolioSummaryCard extends StatelessWidget {
  const PortfolioSummaryCard({
    required this.snapshot,
    super.key,
  });

  final DashboardSnapshot snapshot;

  @override
  Widget build(BuildContext context) {
    final summary = snapshot.portfolio.summary;
    final currency = NumberFormat.currency(symbol: 'Rs ', decimalDigits: 2);
    final gainPositive = summary.totalGainLoss >= 0;

    return AppPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Portfolio summary',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
          const SizedBox(height: 6),
          Text(
            '${summary.numStocks} holdings tracked with current price snapshots and watchlist context.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
          const SizedBox(height: 18),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _StatTile(
                label: 'Investment',
                value: currency.format(summary.totalInvestment),
              ),
              _StatTile(
                label: 'Value',
                value: currency.format(summary.totalValue),
              ),
              _StatTile(
                label: 'Gain / Loss',
                value: currency.format(summary.totalGainLoss),
                valueColor: gainPositive ? AppColors.positive : AppColors.negative,
              ),
              _StatTile(
                label: 'Return',
                value: '${summary.totalReturnPct.toStringAsFixed(2)}%',
                valueColor: gainPositive ? AppColors.positive : AppColors.negative,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatTile extends StatelessWidget {
  const _StatTile({
    required this.label,
    required this.value,
    this.valueColor,
  });

  final String label;
  final String value;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Container(
      width: 150,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: textTheme.labelMedium?.copyWith(color: AppColors.textMuted),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: tabularFigures(
              textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: valueColor ?? AppColors.textPrimary,
                  ) ??
                  const TextStyle(),
            ),
          ),
        ],
      ),
    );
  }
}
