import '../models/dashboard_snapshot.dart';
import '../nepse_api_client.dart';

class NepseDashboardService {
  const NepseDashboardService(this._apiClient);

  final NepseApiClient _apiClient;

  Future<DashboardSnapshot> loadSnapshot({int limit = 5}) {
    return _apiClient.getDashboard(limit: limit);
  }
}
