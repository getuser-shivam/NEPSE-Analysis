import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:nepse_client/nepse_client.dart';

import '../../theme/app_theme.dart';
import '../common/glass_container.dart';
import '../common/status_pill.dart';

class SymbolsOverviewCard extends StatelessWidget {
  const SymbolsOverviewCard({required this.snapshot, super.key});

  final DashboardSnapshot snapshot;

  @override
  Widget build(BuildContext context) {
    final currency = NumberFormat.currency(symbol: 'Rs ', decimalDigits: 2);
    final symbols = snapshot.symbols;

    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Tracked symbols',
            style: Theme.of(
              context,
            ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            'Blended portfolio and watchlist view, shaped like the dashboard snapshots from the backend.',
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(color: AppColors.textSecondary),
          ),
          const SizedBox(height: 18),
          for (final symbol in symbols) ...[
            _SymbolRow(symbol: symbol, currency: currency),
            if (symbol != symbols.last) const SizedBox(height: 12),
          ],
        ],
      ),
    );
  }
}

class _SymbolRow extends StatelessWidget {
  const _SymbolRow({required this.symbol, required this.currency});

  final DashboardSymbolSnapshot symbol;
  final NumberFormat currency;

  @override
  Widget build(BuildContext context) {
    final lastPrice = symbol.recentPrices.isEmpty
        ? null
        : symbol.recentPrices.first;
    final holding = symbol.holding;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                symbol.symbol,
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              if (symbol.inPortfolio)
                const StatusPill(label: 'Portfolio', color: AppColors.accent),
              if (symbol.inPortfolio && symbol.inWatchlist)
                const SizedBox(width: 8),
              if (symbol.inWatchlist)
                const StatusPill(
                  label: 'Watchlist',
                  color: AppColors.highlight,
                ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _InlineMetric(
                label: 'Shares',
                value: holding == null
                    ? 'Not held'
                    : holding.shares.toStringAsFixed(2),
              ),
              _InlineMetric(
                label: 'Current',
                value: holding?.currentPrice == null
                    ? (lastPrice == null
                          ? 'No price'
                          : currency.format(lastPrice.close))
                    : currency.format(holding!.currentPrice!),
              ),
              _InlineMetric(
                label: 'Last close',
                value: lastPrice == null
                    ? 'No data'
                    : currency.format(lastPrice.close),
              ),
              _InlineMetric(
                label: 'Snapshots',
                value: '${symbol.recentPrices.length}',
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _InlineMetric extends StatelessWidget {
  const _InlineMetric({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return SizedBox(
      width: 128,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: textTheme.labelSmall?.copyWith(color: AppColors.textMuted),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: tabularFigures(
              textTheme.labelLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ) ??
                  const TextStyle(),
            ),
          ),
        ],
      ),
    );
  }
}
