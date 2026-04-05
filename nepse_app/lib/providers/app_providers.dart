import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';

import '../services/api_settings_service.dart';

final apiSettingsServiceProvider = Provider<ApiSettingsService>((ref) {
  return ApiSettingsService();
});

class ApiBaseUrlNotifier extends AsyncNotifier<String> {
  @override
  Future<String> build() async {
    return ref.read(apiSettingsServiceProvider).loadBaseUrl();
  }

  Future<void> saveBaseUrl(String value) async {
    final normalized = _normalizeBaseUrl(value);
    await ref.read(apiSettingsServiceProvider).saveBaseUrl(normalized);
    state = AsyncData(normalized);
    ref.invalidate(nepseApiClientProvider);
    ref.invalidate(dashboardSnapshotProvider);
    ref.invalidate(apiHealthProvider);
  }

  String _normalizeBaseUrl(String value) {
    final trimmed = value.trim().replaceAll(RegExp(r'/+$'), '');
    if (trimmed.isEmpty) {
      return defaultApiBaseUrl;
    }

    final uri = Uri.tryParse(trimmed);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
      return defaultApiBaseUrl;
    }

    return trimmed;
  }
}

final apiBaseUrlProvider =
    AsyncNotifierProvider<ApiBaseUrlNotifier, String>(ApiBaseUrlNotifier.new);

final nepseApiClientProvider = Provider<NepseApiClient>((ref) {
  final baseUrl = ref.watch(apiBaseUrlProvider).valueOrNull ?? defaultApiBaseUrl;
  final client = NepseApiClient(baseUrl: baseUrl);
  ref.onDispose(client.close);
  return client;
});

final dashboardServiceProvider = Provider<NepseDashboardService>((ref) {
  return NepseDashboardService(ref.watch(nepseApiClientProvider));
});

final dashboardSnapshotProvider = FutureProvider<DashboardSnapshot>((ref) async {
  return ref.watch(dashboardServiceProvider).loadSnapshot(limit: 6);
});

final apiHealthProvider = FutureProvider<Map<String, dynamic>>((ref) async {
  return ref.watch(nepseApiClientProvider).getHealth();
});
