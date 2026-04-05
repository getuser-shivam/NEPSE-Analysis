import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/app_providers.dart';
import '../services/api_settings_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/app_panel.dart';
import '../widgets/common/status_pill.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  late final TextEditingController _baseUrlController;

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController();
    Future.microtask(() async {
      final url = await ref.read(apiBaseUrlProvider.future);
      if (mounted && _baseUrlController.text.isEmpty) {
        _baseUrlController.text = url;
      }
    });
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final baseUrlAsync = ref.watch(apiBaseUrlProvider);
    final healthAsync = ref.watch(apiHealthProvider);

    final loadedBaseUrl = baseUrlAsync.valueOrNull ?? defaultApiBaseUrl;
    if (_baseUrlController.text.isEmpty && loadedBaseUrl.isNotEmpty) {
      _baseUrlController.text = loadedBaseUrl;
    }

    return DecoratedBox(
      decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
      child: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
          children: [
            Text(
              'Connection settings',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'Point the mobile app at the LAN-accessible Node backend running on your computer.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppColors.textSecondary,
                  ),
            ),
            const SizedBox(height: 18),
            AppPanel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        'API base URL',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                      ),
                      const Spacer(),
                      healthAsync.when(
                        data: (_) => const StatusPill(
                          label: 'Reachable',
                          color: AppColors.positive,
                        ),
                        loading: () => const StatusPill(
                          label: 'Checking',
                          color: AppColors.highlight,
                        ),
                        error: (_, _) => const StatusPill(
                          label: 'Offline',
                          color: AppColors.negative,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _baseUrlController,
                    keyboardType: TextInputType.url,
                    decoration: const InputDecoration(
                      labelText: 'Backend URL',
                      hintText: 'http://192.168.1.79:4000',
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Default build target uses your current PC address so the phone can connect over Wi-Fi.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppColors.textMuted,
                          height: 1.5,
                        ),
                  ),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () async {
                            final messenger = ScaffoldMessenger.of(context);
                            await ref
                                .read(apiBaseUrlProvider.notifier)
                                .saveBaseUrl(_baseUrlController.text);

                            if (!mounted) {
                              return;
                            }

                            messenger.showSnackBar(
                              const SnackBar(
                                content: Text('API base URL saved'),
                              ),
                            );
                          },
                          child: const Text('Save URL'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () {
                            _baseUrlController.text = defaultApiBaseUrl;
                          },
                          child: const Text('Use Default'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            AppPanel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Current target',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    loadedBaseUrl,
                    style: tabularFigures(
                      Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w700,
                          ) ??
                          const TextStyle(),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'If your computer IP changes, update this field or run the app with a new --dart-define NEPSE_API_URL=... value.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppColors.textSecondary,
                          height: 1.5,
                        ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
