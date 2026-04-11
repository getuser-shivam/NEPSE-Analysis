import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/distribution_models.dart';
import '../services/distribution_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/glass_container.dart';
import '../widgets/common/status_pill.dart';

final distributionServiceProvider = Provider((ref) => DistributionService());

final analyticsProvider = FutureProvider<Map<String, dynamic>>((ref) async {
  return ref.watch(distributionServiceProvider).getAnalytics();
});

class AdminAnalyticsScreen extends ConsumerWidget {
  const AdminAnalyticsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final analyticsAsync = ref.watch(analyticsProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Market Insights')),
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: analyticsAsync.when(
          data: (data) => _buildContent(context, data, ref),
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (err, _) => Center(child: Text('Error: $err')),
        ),
      ),
    );
  }

  Widget _buildContent(
    BuildContext context,
    Map<String, dynamic> data,
    WidgetRef ref,
  ) {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        _buildSummaryCards(data),
        const SizedBox(height: 24),
        const Text(
          'Product Engagement',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        _buildInteractionList(data['topInteractions'] ?? []),
        const SizedBox(height: 24),
        const Text(
          'Inventory Health',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        const GlassContainer(
          child: Column(
            children: [
              ListTile(
                leading: Icon(
                  Icons.inventory_2_outlined,
                  color: AppColors.accent,
                ),
                title: Text('Stock Forecasting Active'),
                subtitle: Text('Powered by real-time distribution engine.'),
                trailing: StatusPill(
                  label: 'Optimal',
                  color: AppColors.positive,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryCards(Map<String, dynamic> data) {
    return Row(
      children: [
        Expanded(
          child: GlassContainer(
            child: Column(
              children: [
                const Text(
                  'Total Revenue',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'NPR ${data['totalRevenue']}',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: GlassContainer(
            child: Column(
              children: [
                const Text(
                  'Total Orders',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${data['totalOrders']}',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildInteractionList(List<dynamic> interactions) {
    return GlassContainer(
      padding: EdgeInsets.zero,
      child: Column(
        children: interactions.map<Widget>((item) {
          return ListTile(
            title: Text(
              'Product: ${item['productId'].toString().substring(0, 8)}...',
            ),
            subtitle: Text('Action: ${item['action']}'),
            trailing: CircleAvatar(
              backgroundColor: AppColors.accent.withValues(alpha: 0.1),
              child: Text(
                item['_count']['id'].toString(),
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: AppColors.accent,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}
