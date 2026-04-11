import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/ai_config.dart';
import '../models/ai_provider.dart';
import '../models/ai_service_status.dart';
import '../providers/app_providers.dart';
import '../services/api_settings_service.dart';
import '../theme/app_theme.dart';
import '../widgets/common/glass_container.dart';
import '../widgets/common/status_pill.dart';
import '../models/user_auth.dart';
import 'mfa_setup_screen.dart';
import 'login_screen.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  late final TextEditingController _baseUrlController;
  late final TextEditingController _aiUrlController;
  late final TextEditingController _aiModelController;
  late final TextEditingController _aiApiKeyController;

  AiProvider _selectedAiProvider = AiProvider.groqChat;
  bool _aiConfigLoaded = false;

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController();
    _aiUrlController = TextEditingController();
    _aiModelController = TextEditingController();
    _aiApiKeyController = TextEditingController();

    Future.microtask(() async {
      final url = await ref.read(apiBaseUrlProvider.future);
      final aiConfig = await ref.read(aiConfigProvider.future);

      if (!mounted) {
        return;
      }

      _baseUrlController.text = url;
      _applyAiConfig(aiConfig);
    });
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _aiUrlController.dispose();
    _aiModelController.dispose();
    _aiApiKeyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final baseUrlAsync = ref.watch(apiBaseUrlProvider);
    final healthAsync = ref.watch(apiHealthProvider);
    final aiConfigAsync = ref.watch(aiConfigProvider);
    final aiVerificationAsync = ref.watch(aiVerificationProvider);
    final authState = ref.watch(authProvider);

    final loadedBaseUrl = baseUrlAsync.valueOrNull ?? defaultApiBaseUrl;
    if (_baseUrlController.text.isEmpty && loadedBaseUrl.isNotEmpty) {
      _baseUrlController.text = loadedBaseUrl;
    }

    final loadedAiConfig = aiConfigAsync.valueOrNull;
    if (!_aiConfigLoaded && loadedAiConfig != null) {
      _applyAiConfig(loadedAiConfig);
    }

    return DecoratedBox(
      decoration: const BoxDecoration(gradient: AppTheme.backgroundGradient),
      child: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
          children: [
            Text(
              'Connection settings',
              style: Theme.of(
                context,
              ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(
              'Configure the LAN backend plus Groq or Pollinations AI credentials for NEPSE summaries.',
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.textSecondary),
            ),
            const SizedBox(height: 18),
            _buildUserPanel(context, authState),
            const SizedBox(height: 16),
            _buildApiPanel(context, healthAsync),
            const SizedBox(height: 16),
            _buildAiPanel(context, aiVerificationAsync),
            const SizedBox(height: 16),
            _buildCurrentTargetPanel(context, loadedBaseUrl, loadedAiConfig),
          ],
        ),
      ),
    );
  }

  Widget _buildUserPanel(BuildContext context, AuthState authState) {
    final user = authState.user;
    if (user == null) {
      return GlassContainer(
        child: Column(
          children: [
            const Text('You are not logged in.'),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) => const LoginScreen()));
              },
              child: const Text('Go to Login'),
            ),
          ],
        ),
      );
    }

    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                backgroundColor: AppColors.accent.withValues(alpha: 0.1),
                child: const Icon(Icons.person, color: AppColors.accent),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      user.name ?? 'Stock Analyst',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
                    ),
                    Text(
                      user.email,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(color: AppColors.textSecondary),
                    ),
                  ],
                ),
              ),
              StatusPill(
                label: user.mfaEnabled ? 'MFA: Enabled' : 'MFA: Off',
                color: user.mfaEnabled ? AppColors.positive : AppColors.highlight,
              ),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const MfaSetupScreen()));
                  },
                  icon: const Icon(Icons.security),
                  label: const Text('Security'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () async {
                    await ref.read(authProvider.notifier).logout();
                  },
                  icon: const Icon(Icons.logout),
                  label: const Text('Logout'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.withValues(alpha: 0.1),
                    foregroundColor: Colors.red,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildApiPanel(
    BuildContext context,
    AsyncValue<Map<String, dynamic>> healthAsync,
  ) {
    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'API base URL',
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
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
                      const SnackBar(content: Text('API base URL saved')),
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
    );
  }

  Widget _buildAiPanel(
    BuildContext context,
    AsyncValue<AiServiceStatus> aiVerificationAsync,
  ) {
    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'AI provider configuration',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Learned from the iFutures provider flow: choose a provider, save the key securely, then verify before using AI features.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              _buildAiStatusPill(aiVerificationAsync),
            ],
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<AiProvider>(
            key: ValueKey(_selectedAiProvider),
            initialValue: _selectedAiProvider,
            decoration: const InputDecoration(labelText: 'AI Provider'),
            items: AiProvider.values
                .map(
                  (provider) => DropdownMenuItem<AiProvider>(
                    value: provider,
                    child: Text(provider.label),
                  ),
                )
                .toList(growable: false),
            onChanged: (value) {
              if (value == null) {
                return;
              }

              setState(() {
                _selectedAiProvider = value;
                _aiUrlController.text = value.defaultUrl;
                _aiModelController.text = value.defaultModel;
              });
            },
          ),
          const SizedBox(height: 12),
          Text(
            _selectedAiProvider.helperText,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppColors.textMuted,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _aiApiKeyController,
            obscureText: true,
            decoration: InputDecoration(
              labelText: _selectedAiProvider.requiresApiKey
                  ? '${_selectedAiProvider.label} API Key'
                  : '${_selectedAiProvider.label} API Key (Optional)',
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _aiUrlController,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(labelText: 'AI URL'),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _aiModelController,
            decoration: const InputDecoration(labelText: 'AI Model'),
          ),
          const SizedBox(height: 16),
          if (aiVerificationAsync.valueOrNull != null)
            _buildAiVerificationNote(context, aiVerificationAsync.valueOrNull!),
          if (aiVerificationAsync.valueOrNull != null)
            const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final messenger = ScaffoldMessenger.of(context);
                    await ref
                        .read(aiConfigProvider.notifier)
                        .saveConfig(_currentAiConfig());

                    if (!mounted) {
                      return;
                    }

                    messenger.showSnackBar(
                      const SnackBar(
                        content: Text('AI settings saved locally'),
                      ),
                    );
                  },
                  icon: const Icon(Icons.save_outlined),
                  label: const Text('Save AI'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () async {
                    final messenger = ScaffoldMessenger.of(context);
                    await ref
                        .read(aiConfigProvider.notifier)
                        .saveConfig(_currentAiConfig());
                    await ref.read(aiVerificationProvider.notifier).verify();

                    if (!mounted) {
                      return;
                    }

                    final status = ref.read(aiVerificationProvider).valueOrNull;
                    messenger.showSnackBar(
                      SnackBar(
                        content: Text(
                          status?.message ??
                              'AI provider verification finished.',
                        ),
                      ),
                    );
                  },
                  icon: const Icon(Icons.psychology_alt_outlined),
                  label: const Text('Verify AI'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCurrentTargetPanel(
    BuildContext context,
    String loadedBaseUrl,
    AiConfig? loadedAiConfig,
  ) {
    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Current targets',
            style: Theme.of(
              context,
            ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
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
          const SizedBox(height: 14),
          if (loadedAiConfig != null) ...[
            Text(
              '${loadedAiConfig.provider.label} | ${loadedAiConfig.model}',
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Text(
              loadedAiConfig.url,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppColors.textSecondary,
                height: 1.5,
              ),
            ),
          ],
          const SizedBox(height: 10),
          Text(
            'If your computer IP changes, update the backend URL. If you switch AI providers, save and verify again so the dashboard brief uses the new credentials.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppColors.textSecondary,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  void _applyAiConfig(AiConfig config) {
    setState(() {
      _selectedAiProvider = config.provider;
      _aiUrlController.text = config.url;
      _aiModelController.text = config.model;
      _aiApiKeyController.text = config.apiKey;
      _aiConfigLoaded = true;
    });
  }

  AiConfig _currentAiConfig() {
    return AiConfig(
      provider: _selectedAiProvider,
      url: _aiUrlController.text.trim(),
      model: _aiModelController.text.trim(),
      apiKey: _aiApiKeyController.text,
    );
  }

  Widget _buildAiVerificationNote(
    BuildContext context,
    AiServiceStatus status,
  ) {
    final localizations = MaterialLocalizations.of(context);
    final checkedAt = status.checkedAt == null
        ? 'Not checked yet'
        : 'Checked ${localizations.formatTimeOfDay(TimeOfDay.fromDateTime(status.checkedAt!.toLocal()))}';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            status.providerLabel,
            style: Theme.of(
              context,
            ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            status.message ?? 'No verification details yet.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppColors.textSecondary,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            checkedAt,
            style: Theme.of(
              context,
            ).textTheme.labelMedium?.copyWith(color: AppColors.textMuted),
          ),
        ],
      ),
    );
  }

  Widget _buildAiStatusPill(AsyncValue<AiServiceStatus> status) {
    return status.when(
      data: (data) => StatusPill(
        label: switch (data.state) {
          AiServiceState.notConfigured => 'AI: Pending',
          AiServiceState.checking => 'AI: Checking',
          AiServiceState.active => 'AI: Active',
          AiServiceState.attentionRequired => 'AI: Attention',
        },
        color: switch (data.state) {
          AiServiceState.notConfigured => AppColors.highlight,
          AiServiceState.checking => AppColors.highlight,
          AiServiceState.active => AppColors.positive,
          AiServiceState.attentionRequired => AppColors.negative,
        },
      ),
      loading: () =>
          const StatusPill(label: 'AI: Checking', color: AppColors.highlight),
      error: (error, stackTrace) =>
          const StatusPill(label: 'AI: Error', color: AppColors.negative),
    );
  }
}
