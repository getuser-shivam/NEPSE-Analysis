export function buildPortfolioSummary(holdings) {
  const totals = holdings.reduce(
    (summary, holding) => {
      const currentPrice = holding.currentPrice ?? holding.buyPrice;
      const investment = holding.shares * holding.buyPrice;
      const value = holding.shares * currentPrice;

      return {
        totalInvestment: summary.totalInvestment + investment,
        totalValue: summary.totalValue + value
      };
    },
    {
      totalInvestment: 0,
      totalValue: 0
    }
  );

  const totalGainLoss = totals.totalValue - totals.totalInvestment;
  const totalReturnPct =
    totals.totalInvestment > 0
      ? (totalGainLoss / totals.totalInvestment) * 100
      : 0;

  return {
    totalInvestment: totals.totalInvestment,
    totalValue: totals.totalValue,
    totalGainLoss,
    totalReturnPct,
    numStocks: holdings.length
  };
}
