import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/watchlist_models.dart';
import '../services/watchlist_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/glass_container.dart';
import '../widgets/common/status_pill.dart';

final watchlistServiceProvider = Provider((ref) => WatchlistService());

final watchlistsProvider = FutureProvider<List<Watchlist>>((ref) async {
  return ref.watch(watchlistServiceProvider).getWatchlists();
});

class WatchlistScreen extends ConsumerWidget {
  const WatchlistScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final watchlistsAsync = ref.watch(watchlistsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Watchlists'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _showCreateWatchlistDialog(context, ref),
          ),
        ],
      ),
      body: DecoratedBox(
        decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
        child: watchlistsAsync.when(
          data: (watchlists) => _buildWatchlistList(context, watchlists, ref),
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (err, _) =>
              Center(child: Text('Error loading watchlists: $err')),
        ),
      ),
    );
  }

  Widget _buildWatchlistList(
    BuildContext context,
    List<Watchlist> watchlists,
    WidgetRef ref,
  ) {
    if (watchlists.isEmpty) {
      return const Center(
        child: Text(
          'No watchlists yet. Create one to start monitoring stocks.',
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: watchlists.length,
      itemBuilder: (context, index) {
        final watchlist = watchlists[index];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 12),
              child: Text(
                watchlist.name,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            ...watchlist.items.map(
              (item) => _buildWatchlistItem(context, item, ref),
            ),
            const SizedBox(height: 24),
          ],
        );
      },
    );
  }

  Widget _buildWatchlistItem(
    BuildContext context,
    WatchlistItem item,
    WidgetRef ref,
  ) {
    final stock = item.stock;
    if (stock == null) return const SizedBox.shrink();

    final isPositive = (stock.changePercent ?? 0) >= 0;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: GlassContainer(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    stock.symbol,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    stock.name,
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  'NPR ${stock.lastPrice?.toStringAsFixed(2) ?? '0.00'}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                StatusPill(
                  label:
                      '${isPositive ? '+' : ''}${stock.changePercent?.toStringAsFixed(2)}%',
                  color: isPositive ? AppColors.positive : AppColors.negative,
                ),
              ],
            ),
            const SizedBox(width: 8),
            IconButton(
              icon: const Icon(Icons.notifications_none_outlined, size: 20),
              onPressed: () => _showSetAlertSheet(context, item, ref),
            ),
          ],
        ),
      ),
    );
  }

  void _showCreateWatchlistDialog(BuildContext context, WidgetRef ref) {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('New Watchlist'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(labelText: 'Watchlist Name'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              await ref
                  .read(watchlistServiceProvider)
                  .createWatchlist(controller.text);
              ref.invalidate(watchlistsProvider);
              if (context.mounted) Navigator.pop(context);
            },
            child: const Text('Create'),
          ),
        ],
      ),
    );
  }

  void _showSetAlertSheet(
    BuildContext context,
    WatchlistItem item,
    WidgetRef ref,
  ) {
    // Implementation for Set Alert bottom sheet
  }
}
