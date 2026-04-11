/**
 * Recommendations Routes
 * 
 * API endpoints for personalized investment recommendations.
 * 
 * Routes:
 * - GET /          - Get recommendations for user
 * - GET /:symbol   - Get recommendation for specific stock
 */

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';

const router = Router();
const prisma = new PrismaClient();

const querySchema = z.object({
  riskProfile: z.enum(['conservative', 'moderate', 'aggressive']).optional(),
  sector: z.string().optional(),
  limit: z.string().transform(Number).default('10'),
});

/**
 * @swagger
 * /api/recommendations:
 *   get:
 *     summary: Get investment recommendations
 *     description: Get personalized investment recommendations based on technical analysis and AI
 *     tags: [Recommendations]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: riskProfile
 *         schema:
 *           type: string
 *           enum: [conservative, moderate, aggressive]
 *         description: User risk profile
 *       - in: query
 *         name: sector
 *         schema:
 *           type: string
 *         description: Filter by sector
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 10
 *         description: Number of recommendations
 *     responses:
 *       200:
 *         description: Recommendations retrieved successfully
 *       401:
 *         description: Unauthorized
 */
router.get('/', async (request, response, next) => {
  try {
    const { riskProfile, sector, limit } = querySchema.parse(request.query);
    
    // Get user's watchlist or popular stocks
    const where = sector ? { sector } : {};
    
    const stocks = await prisma.stock.findMany({
      where,
      take: limit,
      orderBy: { symbol: 'asc' },
      include: {
        priceHistory: {
          orderBy: { tradeDate: 'desc' },
          take: 30,
        },
        indicators: {
          orderBy: { date: 'desc' },
          take: 1,
        },
      },
    });

    // Generate recommendations
    const recommendations = stocks.map(stock => {
      const latestPrice = stock.priceHistory[0]?.close || 0;
      const indicator = stock.indicators[0];
      
      // Simple recommendation logic
      let action = 'HOLD';
      let confidence = 0.5;
      
      if (indicator) {
        if (indicator.rsi14 && indicator.rsi14 < 30) {
          action = 'BUY';
          confidence = 0.7;
        } else if (indicator.rsi14 && indicator.rsi14 > 70) {
          action = 'SELL';
          confidence = 0.7;
        }
        
        if (indicator.macdHistogram && indicator.macdHistogram > 0) {
          confidence += 0.1;
        }
      }
      
      return {
        symbol: stock.symbol,
        stockName: stock.name,
        action,
        confidence,
        currentPrice: latestPrice,
        targetPrice: action === 'BUY' ? latestPrice * 1.08 : latestPrice * 0.95,
        stopLoss: action === 'BUY' ? latestPrice * 0.95 : latestPrice * 1.05,
        timeHorizon: riskProfile === 'conservative' ? '3-6 months' : '1-3 months',
        rationale: `Based on technical indicators (RSI: ${indicator?.rsi14?.toFixed(1) || 'N/A'})`,
        riskLevel: riskProfile?.toUpperCase() || 'MODERATE',
        technicalFactors: [
          `RSI: ${indicator?.rsi14?.toFixed(1) || 'N/A'}`,
          `MACD: ${indicator?.macdHistogram?.toFixed(2) || 'N/A'}`,
        ].filter(Boolean),
        generatedAt: new Date().toISOString(),
      };
    });

    // Sort by confidence
    recommendations.sort((a, b) => b.confidence - a.confidence);

    response.json({
      recommendations,
      riskProfile: riskProfile || 'moderate',
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/recommendations/{symbol}:
 *   get:
 *     summary: Get recommendation for specific stock
 *     description: Get personalized recommendation for a single stock
 *     tags: [Recommendations]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: symbol
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Recommendation retrieved
 *       404:
 *         description: Stock not found
 */
router.get('/:symbol', async (request, response, next) => {
  try {
    const { symbol } = request.params;
    
    const stock = await prisma.stock.findUnique({
      where: { symbol: symbol.toUpperCase() },
      include: {
        priceHistory: {
          orderBy: { tradeDate: 'desc' },
          take: 30,
        },
        indicators: {
          orderBy: { date: 'desc' },
          take: 1,
        },
      },
    });

    if (!stock) {
      return response.status(404).json({
        error: 'Stock not found',
        symbol: symbol.toUpperCase(),
      });
    }

    const latestPrice = stock.priceHistory[0]?.close || 0;
    const indicator = stock.indicators[0];
    
    // Generate recommendation
    let action = 'HOLD';
    let confidence = 0.5;
    
    if (indicator) {
      if (indicator.rsi14 < 30) {
        action = 'BUY';
        confidence = 0.75;
      } else if (indicator.rsi14 > 70) {
        action = 'SELL';
        confidence = 0.75;
      }
    }

    response.json({
      symbol: stock.symbol,
      stockName: stock.name,
      action,
      confidence,
      currentPrice: latestPrice,
      targetPrice: action === 'BUY' ? latestPrice * 1.08 : latestPrice * 0.95,
      stopLoss: action === 'BUY' ? latestPrice * 0.95 : latestPrice * 1.05,
      timeHorizon: '1-3 months',
      rationale: `Technical analysis shows RSI at ${indicator?.rsi14?.toFixed(1) || 'N/A'}`,
      riskLevel: 'MEDIUM',
      technicalFactors: [
        `RSI: ${indicator?.rsi14?.toFixed(1) || 'N/A'}`,
        `MACD: ${indicator?.macdHistogram?.toFixed(2) || 'N/A'}`,
      ],
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

export default router;
