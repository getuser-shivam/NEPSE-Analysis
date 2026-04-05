import { prisma } from '../lib/prisma.js';
import { ensureDefaultSettings } from '../lib/appSettings.js';
import { buildPortfolioSummary } from './portfolioSummary.js';

export async function loadDashboardSnapshot({ limit = 5 } = {}) {
  const [settings, holdings, watchlist] = await Promise.all([
    ensureDefaultSettings(),
    prisma.portfolioHolding.findMany({
      orderBy: { symbol: 'asc' }
    }),
    prisma.watchlistItem.findMany({
      orderBy: { symbol: 'asc' }
    })
  ]);

  const holdingBySymbol = new Map(
    holdings.map((holding) => [holding.symbol, holding])
  );
  const watchlistSymbols = new Set(watchlist.map((item) => item.symbol));
  const orderedSymbols = [
    ...holdings.map((holding) => holding.symbol),
    ...watchlist
      .map((item) => item.symbol)
      .filter((symbol) => !holdingBySymbol.has(symbol))
  ];

  const symbols = await Promise.all(
    orderedSymbols.map(async (symbol) => {
      const recentPrices = await prisma.priceSnapshot.findMany({
        where: { symbol },
        orderBy: { tradeDate: 'desc' },
        take: limit
      });

      return {
        symbol,
        inPortfolio: holdingBySymbol.has(symbol),
        inWatchlist: watchlistSymbols.has(symbol),
        holding: holdingBySymbol.get(symbol) ?? null,
        recentPrices
      };
    })
  );

  return {
    generatedAt: new Date().toISOString(),
    settings,
    portfolio: {
      holdings,
      summary: buildPortfolioSummary(holdings)
    },
    watchlist,
    symbols
  };
}
