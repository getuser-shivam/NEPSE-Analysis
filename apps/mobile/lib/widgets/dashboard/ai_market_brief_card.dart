import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:nepse_client/nepse_client.dart';

import '../../models/ai_brief_focus.dart';
import '../../models/ai_market_brief.dart';
import '../../models/ai_provider.dart';
import '../../models/ai_service_status.dart';
import '../../providers/app_providers.dart';
import '../../theme/app_theme.dart';
import '../common/glass_container.dart';
import '../common/status_pill.dart';

class AiMarketBriefCard extends ConsumerWidget {
  const AiMarketBriefCard({required this.snapshot, super.key});

  final DashboardSnapshot snapshot;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final configAsync = ref.watch(aiConfigProvider);
    final verificationAsync = ref.watch(aiVerificationProvider);
    final focus = ref.watch(aiBriefFocusProvider);
    final briefAsync = ref.watch(aiMarketBriefProvider);

    return configAsync.when(
      data: (config) {
        final ready = !config.provider.requiresApiKey || config.hasApiKey;
        final verification = verificationAsync.valueOrNull;
        final providerColor = switch (config.provider) {
          AiProvider.groqChat => AppColors.accent,
          AiProvider.pollinationsText => AppColors.highlight,
        };

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
                          'AI market brief',
                          style: Theme.of(context).textTheme.titleLarge
                              ?.copyWith(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Generate a concise NEPSE read using Groq or Pollinations from the live dashboard snapshot.',
                          style: Theme.of(context).textTheme.bodyMedium
                              ?.copyWith(color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      StatusPill(
                        label: config.provider.label,
                        color: providerColor,
                      ),
                      const SizedBox(height: 8),
                      StatusPill(
                        label: _statusLabel(verification, ready: ready),
                        color: _statusColor(verification, ready: ready),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
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
                        const Icon(
                          Icons.tune_rounded,
                          size: 18,
                          color: AppColors.highlight,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Analysis focus',
                          style: Theme.of(context).textTheme.titleSmall
                              ?.copyWith(fontWeight: FontWeight.w700),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 10,
                      runSpacing: 10,
                      children: [
                        for (final option in AiBriefFocus.values)
                          _FocusChip(
                            focus: option,
                            selected: focus == option,
                            onTap: () {
                              ref.read(aiBriefFocusProvider.notifier).state =
                                  option;
                            },
                          ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text(
                      focus.helperText,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppColors.textSecondary,
                        height: 1.45,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              _ProviderMetaRow(
                model: config.model,
                url: config.url,
                verification: verification,
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () {
                        if (!ready) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                'Add your ${config.provider.label} API key in Settings first.',
                              ),
                            ),
                          );
                          return;
                        }

                        ref
                            .read(aiMarketBriefProvider.notifier)
                            .generate(snapshot, focus: focus);
                      },
                      icon: const Icon(Icons.auto_awesome_rounded),
                      label: const Text('Generate Brief'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton.icon(
                    onPressed: () {
                      ref.read(aiMarketBriefProvider.notifier).clear();
                    },
                    icon: const Icon(Icons.layers_clear_outlined),
                    label: const Text('Clear'),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              briefAsync.when(
                data: (brief) {
                  if (brief == null) {
                    return Text(
                      'No AI brief yet. Generate one after configuring your provider.',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    );
                  }

                  return _BriefBody(brief: brief);
                },
                loading: () => const _LoadingBrief(),
                error: (error, stackTrace) => Text(
                  error.toString(),
                  style: Theme.of(
                    context,
                  ).textTheme.bodyMedium?.copyWith(color: AppColors.negative),
                ),
              ),
            ],
          ),
        );
      },
      loading: () => const GlassContainer(
        child: Column(
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Loading AI configuration...'),
          ],
        ),
      ),
      error: (error, stackTrace) => GlassContainer(
        child: Text(
          error.toString(),
          style: Theme.of(
            context,
          ).textTheme.bodyMedium?.copyWith(color: AppColors.negative),
        ),
      ),
    );
  }
}

class _LoadingBrief extends StatelessWidget {
  const _LoadingBrief();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: const Row(
        children: [
          SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2.2),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Text('AI is reading the dashboard and drafting a brief...'),
          ),
        ],
      ),
    );
  }
}

class _ProviderMetaRow extends StatelessWidget {
  const _ProviderMetaRow({
    required this.model,
    required this.url,
    required this.verification,
  });

  final String model;
  final String url;
  final AiServiceStatus? verification;

  @override
  Widget build(BuildContext context) {
    final checkedAt = verification?.checkedAt == null
        ? 'Not checked yet'
        : DateFormat('MMM d, HH:mm').format(verification!.checkedAt!.toLocal());

    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: [
        _MetaBadge(
          label: 'Model',
          value: model.trim().isEmpty ? 'Default' : model.trim(),
        ),
        _MetaBadge(label: 'Checked', value: checkedAt),
        _MetaBadge(label: 'Endpoint', value: _compactUrl(url)),
      ],
    );
  }

  String _compactUrl(String value) {
    final parsed = Uri.tryParse(value);
    if (parsed == null || parsed.host.isEmpty) {
      return value;
    }

    return parsed.host;
  }
}

class _BriefBody extends StatelessWidget {
  const _BriefBody({required this.brief});

  final AiMarketBrief brief;

  @override
  Widget build(BuildContext context) {
    final generatedAt = DateFormat(
      'MMM d, HH:mm',
    ).format(brief.generatedAt.toLocal());

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            StatusPill(
              label: switch (brief.actionBias) {
                AiActionBias.bullish => 'Bullish Bias',
                AiActionBias.bearish => 'Bearish Bias',
                AiActionBias.neutral => 'Neutral Bias',
              },
              color: switch (brief.actionBias) {
                AiActionBias.bullish => AppColors.positive,
                AiActionBias.bearish => AppColors.negative,
                AiActionBias.neutral => AppColors.highlight,
              },
            ),
            const Spacer(),
            Text(
              '${brief.providerLabel} | $generatedAt',
              style: Theme.of(
                context,
              ).textTheme.labelMedium?.copyWith(color: AppColors.textMuted),
            ),
          ],
        ),
        const SizedBox(height: 16),
        _BriefSection(title: 'Market view', body: brief.marketView),
        const SizedBox(height: 12),
        _BriefSection(title: 'Portfolio take', body: brief.portfolioTake),
        const SizedBox(height: 12),
        _BriefSection(title: 'Risk note', body: brief.riskNote),
        const SizedBox(height: 12),
        _BriefSection(title: 'Next step', body: brief.nextStep),
        if (brief.watchlistFocus.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text(
            'Watchlist focus',
            style: Theme.of(
              context,
            ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 10),
          for (final item in brief.watchlistFocus)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 6),
                    child: Icon(
                      Icons.brightness_1_rounded,
                      size: 8,
                      color: AppColors.highlight,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      item,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: AppColors.textSecondary,
                        height: 1.45,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ],
    );
  }
}

class _BriefSection extends StatelessWidget {
  const _BriefSection({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
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
            title,
            style: Theme.of(
              context,
            ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(
            body,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppColors.textSecondary,
              height: 1.45,
            ),
          ),
        ],
      ),
    );
  }
}

class _FocusChip extends StatelessWidget {
  const _FocusChip({
    required this.focus,
    required this.selected,
    required this.onTap,
  });

  final AiBriefFocus focus;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: selected
              ? AppColors.highlight.withValues(alpha: 0.18)
              : AppColors.surface.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: selected ? AppColors.highlight : AppColors.border,
          ),
        ),
        child: Text(
          focus.label,
          style: Theme.of(context).textTheme.labelLarge?.copyWith(
            fontWeight: FontWeight.w700,
            color: selected ? AppColors.textPrimary : AppColors.textSecondary,
          ),
        ),
      ),
    );
  }
}

class _MetaBadge extends StatelessWidget {
  const _MetaBadge({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: Theme.of(
              context,
            ).textTheme.labelSmall?.copyWith(color: AppColors.textMuted),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

String _statusLabel(AiServiceStatus? status, {required bool ready}) {
  if (!ready) {
    return 'Key Missing';
  }
  if (status == null) {
    return 'Saved';
  }

  return switch (status.state) {
    AiServiceState.notConfigured => 'Saved',
    AiServiceState.checking => 'Checking',
    AiServiceState.active => 'Active',
    AiServiceState.attentionRequired => 'Attention',
  };
}

Color _statusColor(AiServiceStatus? status, {required bool ready}) {
  if (!ready) {
    return AppColors.negative;
  }
  if (status == null) {
    return AppColors.highlight;
  }

  return switch (status.state) {
    AiServiceState.notConfigured => AppColors.highlight,
    AiServiceState.checking => AppColors.highlight,
    AiServiceState.active => AppColors.positive,
    AiServiceState.attentionRequired => AppColors.negative,
  };
}
