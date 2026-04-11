import { PortfolioAnalyticsService } from '../services/portfolioAnalyticsService.js';
import { prisma } from '../database/client.js';

/**
 * Get portfolio insights (Summary and Sector Allocation)
 */
export const getInsights = async (request, response, next) => {
  try {
    const userId = request.user.id;
    const insights = await PortfolioAnalyticsService.getPortfolioInsight(userId);
    response.json(insights);
  } catch (error) {
    next(error);
  }
};

/**
 * Get performance history for charts
 */
export const getHistory = async (request, response, next) => {
  try {
    const userId = request.user.id;
    const { days } = request.query;
    const history = await PortfolioAnalyticsService.getHistoricalPerformance(userId, parseInt(days) || 30);
    response.json(history);
  } catch (error) {
    next(error);
  }
};

/**
 * Manual trigger for snapshot (usually automated)
 */
export const recordSnapshot = async (request, response, next) => {
  try {
    const userId = request.user.id;
    const snapshot = await PortfolioAnalyticsService.captureSnapshot(userId);
    response.status(201).json(snapshot);
  } catch (error) {
    next(error);
  }
};
