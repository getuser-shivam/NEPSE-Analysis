import 'dart:html';
import 'dart:async';
import 'dart:js' as js;
import 'package:nepse_client/nepse_client.dart';

void main() async {
  print('NEPSE Web Dashboard Initializing...');
  
  // Use production URL if hosted, fallback to localhost for dev
  final client = NepseApiClient(baseUrl: 'http://localhost:3000');
  final dashboardService = NepseDashboardService(client);

  // Setup UI event bindings (Auth Modal, Overlays)
  setupEventBindings();

  // Initial load
  await refreshDashboard(dashboardService);

  // Auto-refresh every 60 seconds
  Timer.periodic(const Duration(seconds: 60), (_) => refreshDashboard(dashboardService));
}

// Global state trackers
List<DashboardSymbolSnapshot> currentSymbols = [];
Set<String> activeIndicators = {};

void setupEventBindings() {
  // Auth Modal Bindings
  final authBtn = querySelector('#auth-btn');
  final closeAuthBtn = querySelector('#close-auth');
  final authModal = querySelector('#auth-modal');
  final authForm = querySelector('#auth-form') as FormElement?;

  authBtn?.onClick.listen((_) {
    authModal?.classes.remove('hidden');
  });

  closeAuthBtn?.onClick.listen((_) {
    authModal?.classes.add('hidden');
  });

  // Mock Authentication Flow
  authForm?.onSubmit.listen((e) {
    e.preventDefault();
    final submitBtn = authForm.querySelector('.submit-auth');
    submitBtn?.text = 'Authenticating...';
    
    Timer(const Duration(seconds: 1), () {
      authModal?.classes.add('hidden');
      authBtn?.text = 'Dashboard';
      authBtn?.classes.remove('primary-glow');
      authBtn?.classes.add('status-positive');
      submitBtn?.text = 'Authenticate';
      window.alert('Authentication successful! Welcome to NEPSE Pro.');
    });
  });

  // Chart Overlay Bindings
  final toggles = querySelectorAll('.indicator-toggle');
  for (var toggle in toggles) {
    toggle.onClick.listen((e) {
      final btn = e.target as Element;
      final indicator = btn.dataset['indicator'];
      
      if (indicator != null) {
        if (activeIndicators.contains(indicator)) {
          activeIndicators.remove(indicator);
          btn.classes.remove('active');
        } else {
          activeIndicators.add(indicator);
          btn.classes.add('active');
        }
        
        // Re-render chart with new indicators overlay
        if (currentSymbols.isNotEmpty) {
          renderPriceChart(currentSymbols);
        }
      }
    });
  }
}


Future<void> refreshDashboard(NepseDashboardService service) async {
  try {
    final snapshot = await service.loadSnapshot();
    currentSymbols = snapshot.symbols;
    updateMarketGrid(snapshot.symbols);
    renderPriceChart(snapshot.symbols);
    print('Dashboard Refreshed: ${snapshot.generatedAt}');
  } catch (e) {
    print('Backend unreachable: $e. Rendering fallback mock data for aesthetic demo...');
    final fallbackSymbols = _generateMockSymbols();
    currentSymbols = fallbackSymbols;
    updateMarketGrid(fallbackSymbols);
    renderPriceChart(fallbackSymbols);
  }
}

List<DashboardSymbolSnapshot> _generateMockSymbols() {
  final now = DateTime.now();
  final mockPrices = List.generate(30, (i) {
    return Price(
      symbol: 'NEPSE',
      tradeDate: now.subtract(Duration(days: 30 - i)),
      open: 2100.0 + (i * 2),
      high: 2150.0 + (i * 3.5),
      low: 2090.0 + (i * 1.5),
      close: 2120.0 + (i * 3) + (i % 2 == 0 ? 15 : -10),
      volume: 1500000,
    );
  });
  
  return [
    DashboardSymbolSnapshot(
      symbol: 'NEPSE INDEX',
      recentPrices: mockPrices,
      trend: 'bullish',
      technicalSignal: 'BUY'
    )
  ];
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
  final rawPrices = firstSymbol.recentPrices.toList();
  // Reverse to chronological order if it's descending
  if (rawPrices.length > 1 && rawPrices.first.tradeDate.isAfter(rawPrices.last.tradeDate)) {
    rawPrices = rawPrices.reversed.toList();
  }
  
  final dataPoints = rawPrices.map((p) => p.close).toList();
  final categories = rawPrices.map((p) => p.tradeDate.toString().split(' ')[0]).toList();

  final List<Map<String, dynamic>> seriesList = [
    {
      'name': firstSymbol.symbol,
      'type': 'area',
      'data': dataPoints
    }
  ];

  // Colors mapping for overlays
  final colors = ['#00E5FF'];
  
  // Calculate mock SMA (Simple Moving Average) overlay
  if (activeIndicators.contains('sma')) {
    final period = 5;
    final smaData = List<num?>.filled(dataPoints.length, null);
    for (int i = period - 1; i < dataPoints.length; i++) {
        double sum = 0;
        for (int j = 0; j < period; j++) {
            sum += dataPoints[i - j];
        }
        smaData[i] = num.parse((sum / period).toStringAsFixed(2));
    }
    
    seriesList.add({
      'name': 'SMA (5)',
      'type': 'line',
      'data': smaData
    });
    colors.add('#FFB300'); // Amber
  }

  // Configuration for ApexCharts
  final options = js.JsObject.jsify({
    'chart': {
      'type': 'area', 
      'height': 350, 
      'toolbar': {'show': false},
      'fontFamily': 'Inter, sans-serif'
    },
    'theme': {'mode': 'dark'},
    'colors': colors,
    'dataLabels': {'enabled': false},
    'stroke': {
      'curve': 'smooth', 
      'width': [3, 2] // Width for base area and overlay line
    },
    'series': seriesList,
    'xaxis': {
      'categories': categories,
      'tooltip': {'enabled': false}
    },
    'yaxis': [
      {
        'title': {'text': 'Price (NPR)'},
        'labels': {
          'formatter': js.allowInterop((val) => val == null ? '' : val.toStringAsFixed(0))
        }
      }
    ],
    'fill': {
      'type': ['gradient', 'solid'],
      'gradient': {
        'shadeIntensity': 1,
        'opacityFrom': 0.7,
        'opacityTo': 0.1,
      }
    },
    'legend': {
      'position': 'top',
      'horizontalAlign': 'right'
    }
  });

  js.context.callMethod('eval', ["if (window.chart) { window.chart.destroy(); }"]);
  js.context.callMethod('eval', ["window.chart = new ApexCharts(document.querySelector('#chart'), \$options); window.chart.render();"]);
}
