import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:nepse_client/nepse_client.dart';

import '../../theme/app_theme.dart';
import '../common/app_panel.dart';
import '../common/status_pill.dart';

class DashboardHeaderCard extends StatelessWidget {
  const DashboardHeaderCard({
    required this.snapshot,
    required this.baseUrl,
    super.key,
  });

  final DashboardSnapshot snapshot;
  final String baseUrl;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final generatedAt = DateFormat('MMM d, HH:mm').format(snapshot.generatedAt.toLocal());

    return AppPanel(
      padding: EdgeInsets.zero,
      child: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF1A4456),
              Color(0xFF0D2230),
              Color(0xFF153042),
            ],
          ),
          borderRadius: BorderRadius.all(Radius.circular(24)),
        ),
        child: Padding(
          padding: const EdgeInsets.all(22),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: AppColors.highlight.withValues(alpha: 0.18),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Icon(
                      Icons.candlestick_chart_rounded,
                      color: AppColors.highlight,
                    ),
                  ),
                  const Spacer(),
                  const StatusPill(
                    label: 'LIVE API',
                    color: AppColors.positive,
                  ),
                ],
              ),
              const SizedBox(height: 18),
              Text(
                'NEPSE mobile desk',
                style: textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Portfolio, watchlist, and latest snapshots from your Node + Prisma backend in one view.',
                style: textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                  height: 1.45,
                ),
              ),
              const SizedBox(height: 18),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  _MetaChip(
                    label: 'Updated',
                    value: generatedAt,
                  ),
                  _MetaChip(
                    label: 'Period',
                    value: snapshot.settings.defaultPeriod,
                  ),
                  _MetaChip(
                    label: 'Backend',
                    value: baseUrl,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MetaChip extends StatelessWidget {
  const _MetaChip({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: textTheme.labelSmall?.copyWith(color: AppColors.textMuted),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: textTheme.labelLarge?.copyWith(
              color: AppColors.textPrimary,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}
