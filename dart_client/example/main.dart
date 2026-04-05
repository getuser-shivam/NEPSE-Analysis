import 'dart:io';

import 'package:nepse_client/nepse_client.dart';

Future<void> main() async {
  final api = NepseApiClient(
    baseUrl: Platform.environment['NEPSE_API_URL'] ?? 'http://localhost:4000',
  );
  final dashboardService = NepseDashboardService(api);

  try {
    final health = await api.getHealth();
    print('Health: $health');

    final dashboard = await dashboardService.loadSnapshot(limit: 3);
    print('Default period: ${dashboard.settings.defaultPeriod}');
    print('Tracked holdings: ${dashboard.portfolio.summary.numStocks}');
    print('Dashboard symbols: ${dashboard.symbols.map((item) => item.symbol).join(', ')}');
  } finally {
    api.close();
  }
}
