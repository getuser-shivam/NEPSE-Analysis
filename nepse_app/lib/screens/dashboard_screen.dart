import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/app_providers.dart';
import '../services/api_settings_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/app_panel.dart';
import '../widgets/dashboard/dashboard_header_card.dart';
import '../widgets/dashboard/portfolio_summary_card.dart';
import '../widgets/dashboard/symbols_overview_card.dart';

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final snapshotAsync = ref.watch(dashboardSnapshotProvider);
    final baseUrl = ref.watch(apiBaseUrlProvider).valueOrNull ?? defaultApiBaseUrl;

    return DecoratedBox(
      decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
      child: SafeArea(
        child: RefreshIndicator(
          onRefresh: () => ref.refresh(dashboardSnapshotProvider.future),
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
            children: [
              Text(
                'NEPSE dashboard',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                'Wireless-ready mobile view for the Node + Prisma stack.',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppColors.textSecondary,
                    ),
              ),
              const SizedBox(height: 18),
              snapshotAsync.when(
                data: (snapshot) => Column(
                  children: [
                    DashboardHeaderCard(snapshot: snapshot, baseUrl: baseUrl),
                    const SizedBox(height: 16),
                    PortfolioSummaryCard(snapshot: snapshot),
                    const SizedBox(height: 16),
                    SymbolsOverviewCard(snapshot: snapshot),
                  ],
                ),
                loading: () => const _LoadingState(),
                error: (error, stackTrace) => _ErrorState(
                  message: error.toString(),
                  onRetry: () => ref.invalidate(dashboardSnapshotProvider),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LoadingState extends StatelessWidget {
  const _LoadingState();

  @override
  Widget build(BuildContext context) {
    return const AppPanel(
      child: Column(
        children: [
          SizedBox(height: 8),
          CircularProgressIndicator(),
          SizedBox(height: 18),
          Text('Loading dashboard snapshot...'),
        ],
      ),
    );
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({
    required this.message,
    required this.onRetry,
  });

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return AppPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Dashboard unavailable',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
          const SizedBox(height: 8),
          Text(
            message,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppColors.textSecondary,
                ),
          ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: onRetry,
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}
