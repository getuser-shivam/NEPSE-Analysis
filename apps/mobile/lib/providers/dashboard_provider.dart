import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';
import 'dart:convert';
import 'api_client_provider.dart';

// Provides the Dashboard Snapshot Payload using the injected globally authenticated API Client
final dashboardProvider = FutureProvider.autoDispose<DashboardSnapshot>((ref) async {
  final apiClient = ref.watch(apiClientProvider);
  return await apiClient.getDashboard();
});
