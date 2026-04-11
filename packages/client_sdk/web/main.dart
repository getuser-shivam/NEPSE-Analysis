import 'dart:html';
import 'dart:async';
import 'dart:js' as js;
import 'package:nepse_client/nepse_client.dart';

void main() async {
  print('NEPSE Web Dashboard Initializing...');
  
  // Use production URL if hosted, fallback to localhost for dev
  final client = NepseApiClient(baseUrl: 'http://localhost:3000');
  final dashboardService = NepseDashboardService(client);

  // Initial load
  await refreshDashboard(dashboardService);

  // Auto-refresh every 60 seconds
  Timer.periodic(const Duration(seconds: 60), (_) => refreshDashboard(dashboardService));
}

Future<void> refreshDashboard(NepseDashboardService service) async {
  try {
    final snapshot = await service.loadSnapshot();
    updateMarketGrid(snapshot.symbols);
    renderPriceChart(snapshot.symbols);
    print('Dashboard Refreshed: ${snapshot.generatedAt}');
  } catch (e) {
    print('Failed to refresh dashboard: $e');
  }
}

void updateMarketGrid(List<DashboardSymbolSnapshot> symbols) {
  final tbody = querySelector('#gainers-table tbody');
  if (tbody == null) return;

  tbody.children.clear();
  
  for (final symbolSnapshot in symbols.take(10)) {
    if (symbolSnapshot.recentPrices.isEmpty) continue;
    
    final lastPrice = symbolSnapshot.recentPrices.first;
    final change = lastPrice.close - lastPrice.open;
    final percentChange = (change / lastPrice.open) * 100;

    final tr = TableRowElement();
    tr.children.add(TableCellElement()..text = symbolSnapshot.symbol);
    tr.children.add(TableCellElement()
      ..text = lastPrice.close.toStringAsFixed(2)
      ..style.fontWeight = 'bold');
    
    final changeCell = TableCellElement()
      ..text = '${change >= 0 ? "+" : ""}${change.toStringAsFixed(2)}';
    
    final pChangeCell = TableCellElement()
      ..text = '${percentChange.toStringAsFixed(2)}%';
    
    if (change >= 0) {
      changeCell.classes.add('status-positive');
      pChangeCell.classes.add('status-positive');
    } else {
      changeCell.classes.add('status-negative');
      pChangeCell.classes.add('status-negative');
    }
    
    tr.children.add(changeCell);
    tr.children.add(pChangeCell);
    tbody.children.add(tr);
  }
}

void renderPriceChart(List<DashboardSymbolSnapshot> symbols) {
  if (symbols.isEmpty || symbols.first.recentPrices.isEmpty) return;

  final firstSymbol = symbols.first;
  final dataPoints = firstSymbol.recentPrices.map((p) => p.close).toList();
  final categories = firstSymbol.recentPrices.map((p) => p.tradeDate.toString().split(' ')[0]).toList();

  final options = js.JsObject.jsify({
    'chart': {'type': 'area', 'height': 350, 'toolbar': {'show': false}},
    'theme': {'mode': 'dark'},
    'colors': ['#00E5FF'],
    'dataLabels': {'enabled': false},
    'stroke': {'curve': 'smooth', 'width': 3},
    'series': [{
      'name': firstSymbol.symbol,
      'data': dataPoints
    }],
    'xaxis': {'categories': categories},
    'fill': {
      'type': 'gradient',
      'gradient': {
        'shadeIntensity': 1,
        'opacityFrom': 0.7,
        'opacityTo': 0.1,
      }
    }
  });

  js.context.callMethod('eval', ["if (window.chart) { window.chart.destroy(); }"]);
  js.context.callMethod('eval', ["window.chart = new ApexCharts(document.querySelector('#chart'), $options); window.chart.render();"]);
}
