/// NEPSE Client - A Dart library for the NEPSE Analysis API
///
/// This library provides type-safe access to the NEPSE Analysis
/// Node.js backend API with models for dashboard data, portfolio
/// information, and market data.
///
/// Features:
/// - Dashboard snapshot retrieval
/// - Portfolio data models
/// - Price and market data
/// - Watchlist management
library nepse_client;

// Models
export 'src/models/app_settings.dart';
export 'src/models/dashboard_snapshot.dart';
export 'src/models/dashboard_symbol_snapshot.dart';
export 'src/models/portfolio_holding.dart';
export 'src/models/portfolio_overview.dart';
export 'src/models/portfolio_summary.dart';
export 'src/models/price_snapshot.dart';
export 'src/models/watchlist_item.dart';

// API Client
export 'src/nepse_api_client.dart';

// Services
export 'src/services/nepse_dashboard_service.dart';
