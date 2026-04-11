PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS app_settings (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  auto_save_interval INTEGER NOT NULL DEFAULT 300,
  max_data_age_days INTEGER NOT NULL DEFAULT 7,
  backup_enabled INTEGER NOT NULL DEFAULT 1,
  chart_style TEXT NOT NULL DEFAULT 'seaborn-v0_8',
  default_period TEXT NOT NULL DEFAULT '1y',
  max_watchlist_size INTEGER NOT NULL DEFAULT 50,
  log_level TEXT NOT NULL DEFAULT 'INFO',
  refresh_interval INTEGER NOT NULL DEFAULT 300,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO app_settings (
  id,
  name,
  auto_save_interval,
  max_data_age_days,
  backup_enabled,
  chart_style,
  default_period,
  max_watchlist_size,
  log_level,
  refresh_interval
) VALUES (
  'default-settings',
  'default',
  300,
  7,
  1,
  'seaborn-v0_8',
  '1y',
  50,
  'INFO',
  300
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL UNIQUE,
  shares REAL NOT NULL,
  buy_price REAL NOT NULL,
  current_price REAL,
  notes TEXT,
  last_updated TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist_items (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL UNIQUE,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS price_snapshots (
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'manual',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(symbol, trade_date, source)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_symbol
  ON portfolio_holdings(symbol);

CREATE INDEX IF NOT EXISTS idx_watchlist_items_symbol
  ON watchlist_items(symbol);

CREATE INDEX IF NOT EXISTS idx_price_snapshots_symbol_trade_date
  ON price_snapshots(symbol, trade_date DESC);
