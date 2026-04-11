import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';

import '../models/ai_brief_focus.dart';
import '../models/ai_config.dart';
import '../models/ai_market_brief.dart';
import '../models/ai_provider.dart';
import '../models/ai_service_status.dart';
import '../services/api_settings_service.dart';
import '../services/ai_brief_service.dart';
import '../services/ai_settings_service.dart';
import '../models/product.dart';
import '../services/product_service.dart';
import '../models/user_auth.dart';
import '../services/auth_service.dart';

final apiSettingsServiceProvider = Provider<ApiSettingsService>((ref) {
  return ApiSettingsService();
});

final aiSettingsServiceProvider = Provider<AiSettingsService>((ref) {
  return AiSettingsService();
});

final aiBriefServiceProvider = Provider<AiBriefService>((ref) {
  return const AiBriefService();
});

final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService();
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

final apiBaseUrlProvider = AsyncNotifierProvider<ApiBaseUrlNotifier, String>(
  ApiBaseUrlNotifier.new,
);

final nepseApiClientProvider = Provider<NepseApiClient>((ref) {
  final baseUrl =
      ref.watch(apiBaseUrlProvider).valueOrNull ?? defaultApiBaseUrl;
  final client = NepseApiClient(baseUrl: baseUrl);
  ref.onDispose(client.close);
  return client;
});

final dashboardServiceProvider = Provider<NepseDashboardService>((ref) {
  return NepseDashboardService(ref.watch(nepseApiClientProvider));
});

final dashboardSnapshotProvider = FutureProvider<DashboardSnapshot>((
  ref,
) async {
  return ref.watch(dashboardServiceProvider).loadSnapshot(limit: 6);
});

final apiHealthProvider = FutureProvider<Map<String, dynamic>>((ref) async {
  return ref.watch(nepseApiClientProvider).getHealth();
});

final productsProvider = FutureProvider<List<Product>>((ref) async {
  return ref.watch(productServiceProvider).getProducts();
});

class AiConfigNotifier extends AsyncNotifier<AiConfig> {
  @override
  Future<AiConfig> build() async {
    return ref.read(aiSettingsServiceProvider).loadConfig();
  }

  Future<void> saveConfig(AiConfig config) async {
    await ref.read(aiSettingsServiceProvider).saveConfig(config);
    state = AsyncData(config);
    ref.invalidate(aiVerificationProvider);
    ref.invalidate(aiMarketBriefProvider);
  }
}

final aiConfigProvider = AsyncNotifierProvider<AiConfigNotifier, AiConfig>(
  AiConfigNotifier.new,
);

class AiVerificationNotifier extends AsyncNotifier<AiServiceStatus> {
  @override
  Future<AiServiceStatus> build() async {
    final config = await ref.watch(aiConfigProvider.future);
    return _statusFromConfig(config);
  }

  Future<void> verify() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      final config = await ref.read(aiConfigProvider.future);
      return ref.read(aiBriefServiceProvider).verifyConnection(config);
    });
  }

  AiServiceStatus _statusFromConfig(AiConfig config) {
    if (config.provider.requiresApiKey && !config.hasApiKey) {
      return AiServiceStatus.notConfigured(
        providerLabel: config.provider.label,
        message: 'API key required for ${config.provider.label}.',
      );
    }

    return AiServiceStatus.notConfigured(
      providerLabel: config.provider.label,
      message: 'Saved locally. Run Verify to check the provider.',
    );
  }
}

final aiVerificationProvider =
    AsyncNotifierProvider<AiVerificationNotifier, AiServiceStatus>(
      AiVerificationNotifier.new,
    );

final aiBriefFocusProvider = StateProvider<AiBriefFocus>(
  (ref) => AiBriefFocus.balanced,
);

class AiMarketBriefNotifier extends AsyncNotifier<AiMarketBrief?> {
  @override
  Future<AiMarketBrief?> build() async {
    return null;
  }

  Future<void> generate(
    DashboardSnapshot snapshot, {
    required AiBriefFocus focus,
  }) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      final config = await ref.read(aiConfigProvider.future);
      return ref
          .read(aiBriefServiceProvider)
          .generateBrief(config: config, snapshot: snapshot, focus: focus);
    });
  }

  void clear() {
    state = const AsyncData(null);
  }
}

final aiMarketBriefProvider =
    AsyncNotifierProvider<AiMarketBriefNotifier, AiMarketBrief?>(
      AiMarketBriefNotifier.new,
    );

class AuthNotifier extends StateNotifier<AuthState> {
  final AuthService _authService;

  AuthNotifier(this._authService) : super(const AuthState());

  Future<void> login(String email, String password, {String? mfaCode}) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final result = await _authService.login(
        email: email,
        password: password,
        mfaCode: mfaCode,
      );

      if (result['mfaRequired'] == true) {
        state = state.copyWith(
          isLoading: false,
          mfaRequired: true,
          tempUserId: result['userId'],
        );
      } else {
        state = state.copyWith(
          isLoading: false,
          user: result['user'],
          mfaRequired: false,
        );
      }
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }

  Future<void> logout() async {
    await _authService.logout();
    state = const AuthState();
  }
}

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.watch(authServiceProvider));
});
