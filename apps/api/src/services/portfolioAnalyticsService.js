import { prisma } from '../database/client.js';

export class PortfolioAnalyticsService {
  /**
   * Generates a deep analytics summary for a user's portfolio
   */
  static async getPortfolioInsight(userId) {
    const holdings = await prisma.portfolioHolding.findMany({
      where: { portfolio: { userId } },
      include: { stock: true },
    });

    if (holdings.length === 0) return this.emptySummary();

    let totalInvestment = 0;
    let totalValue = 0;
    const sectorMap = {};

    holdings.forEach((holding) => {
      const currentPrice = holding.stock.lastPrice ?? holding.buyPrice;
      const investment = holding.shares * holding.buyPrice;
      const value = holding.shares * currentPrice;

      totalInvestment += investment;
      totalValue += value;

      // Aggregating by sector
      const sector = holding.stock.sector || 'OTHERS';
      sectorMap[sector] = (sectorMap[sector] || 0) + value;
    });

    const totalGainLoss = totalValue - totalInvestment;
    const totalReturnPct = totalInvestment > 0 ? (totalGainLoss / totalInvestment) * 100 : 0;

    // Formatting sector distribution
    const sectorAllocation = Object.entries(sectorMap).map(([name, value]) => ({
      name,
      value,
      percentage: (value / totalValue) * 100,
    }));

    return {
      totalInvestment,
      totalValue,
      totalGainLoss,
      totalReturnPct,
      numHoldings: holdings.length,
      sectorAllocation,
      generatedAt: new Date(),
    };
  }

  /**
   * Captures a historical snapshot of the portfolio
   */
  static async captureSnapshot(userId) {
    const insight = await this.getPortfolioInsight(userId);
    
    // Check if snapshot for today exists
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    return await prisma.portfolioSnapshot.upsert({
      where: {
        userId_snapshotDate: {
          userId,
          snapshotDate: today,
        },
      },
      update: {
        totalValue: insight.totalValue,
        totalInvestment: insight.totalInvestment,
        totalGainLoss: insight.totalGainLoss,
      },
      create: {
        userId,
        snapshotDate: today,
        totalValue: insight.totalValue,
        totalInvestment: insight.totalInvestment,
        totalGainLoss: insight.totalGainLoss,
      },
    });
  }

  /**
   * Retrieves historical performance data
   */
  static async getHistoricalPerformance(userId, days = 30) {
    const sinceDate = new Date();
    sinceDate.setDate(sinceDate.getDate() - days);

    return await prisma.portfolioSnapshot.findMany({
      where: {
        userId,
        snapshotDate: { gte: sinceDate },
      },
      orderBy: { snapshotDate: 'asc' },
    });
  }

  static emptySummary() {
    return {
      totalInvestment: 0,
      totalValue: 0,
      totalGainLoss: 0,
      totalReturnPct: 0,
      numHoldings: 0,
      sectorAllocation: [],
    };
  }
}
