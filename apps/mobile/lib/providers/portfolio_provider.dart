import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:nepse_client/nepse_client.dart';
import 'api_client_provider.dart';

// Provides the authenticated secure Portfolio Payload from the Node backend
final portfolioProvider = FutureProvider.autoDispose<PortfolioOverview>((ref) async {
  final apiClient = ref.watch(apiClientProvider);
  return await apiClient.getPortfolio();
});
