/**
 * Stocks Routes
 * 
 * API endpoints for stock data retrieval and management.
 * 
 * Routes:
 * - GET /          - List all stocks
 * - GET /:symbol   - Get specific stock details
 * - GET /:symbol/prices - Get stock price history
 */

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';

const router = Router();
const prisma = new PrismaClient();

// Validation schemas
const querySchema = z.object({
  sector: z.string().optional(),
  search: z.string().optional(),
  page: z.string().transform(Number).default('1'),
  limit: z.string().transform(Number).default('50'),
});

const paramsSchema = z.object({
  symbol: z.string().min(1).max(20),
});

/**
 * @swagger
 * /api/stocks:
 *   get:
 *     summary: List all stocks
 *     description: Get a paginated list of all available stocks with optional filtering
 *     tags: [Stocks]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: sector
 *         schema:
 *           type: string
 *         description: Filter by sector (BANKING, HYDRO, etc.)
 *       - in: query
 *         name: search
 *         schema:
 *           type: string
 *         description: Search by symbol or name
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           default: 1
 *         description: Page number
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 50
 *         description: Items per page
 *     responses:
 *       200:
 *         description: List of stocks retrieved successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Server error
 */
router.get('/', async (request, response, next) => {
  try {
    const { sector, search, page, limit } = querySchema.parse(request.query);
    
    const where = {};
    if (sector) {
      where.sector = sector;
    }
    if (search) {
      where.OR = [
        { symbol: { contains: search, mode: 'insensitive' } },
        { name: { contains: search, mode: 'insensitive' } },
      ];
    }

    const skip = (page - 1) * limit;
    
    const [stocks, total] = await Promise.all([
      prisma.stock.findMany({
        where,
        skip,
        take: limit,
        orderBy: { symbol: 'asc' },
        include: {
          priceHistory: {
            orderBy: { tradeDate: 'desc' },
            take: 1,
          },
        },
      }),
      prisma.stock.count({ where }),
    ]);

    response.json({
      data: stocks,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/stocks/{symbol}:
 *   get:
 *     summary: Get stock details
 *     description: Get detailed information about a specific stock
 *     tags: [Stocks]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: symbol
 *         required: true
 *         schema:
 *           type: string
 *         description: Stock symbol
 *     responses:
 *       200:
 *         description: Stock details retrieved successfully
 *       404:
 *         description: Stock not found
 *       401:
 *         description: Unauthorized
 */
router.get('/:symbol', async (request, response, next) => {
  try {
    const { symbol } = paramsSchema.parse(request.params);
    
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

    response.json(stock);
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/stocks/{symbol}/prices:
 *   get:
 *     summary: Get stock price history
 *     description: Get historical OHLCV data for a stock
 *     tags: [Stocks]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: symbol
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: from
 *         schema:
 *           type: string
 *           format: date
 *         description: Start date (YYYY-MM-DD)
 *       - in: query
 *         name: to
 *         schema:
 *           type: string
 *           format: date
 *         description: End date (YYYY-MM-DD)
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [1d, 1w, 1m, 3m, 6m, 1y, all]
 *           default: 1y
 *     responses:
 *       200:
 *         description: Price history retrieved successfully
 *       404:
 *         description: Stock not found
 */
router.get('/:symbol/prices', async (request, response, next) => {
  try {
    const { symbol } = paramsSchema.parse(request.params);
    const { from, to, period = '1y' } = request.query;

    const stock = await prisma.stock.findUnique({
      where: { symbol: symbol.toUpperCase() },
    });

    if (!stock) {
      return response.status(404).json({
        error: 'Stock not found',
        symbol: symbol.toUpperCase(),
      });
    }

    // Calculate date range based on period
    const toDate = to ? new Date(to) : new Date();
    const fromDate = from ? new Date(from) : calculatePeriodStart(period, toDate);

    const prices = await prisma.stockPrice.findMany({
      where: {
        stockId: stock.id,
        tradeDate: {
          gte: fromDate,
          lte: toDate,
        },
      },
      orderBy: { tradeDate: 'asc' },
    });

    response.json({
      symbol: stock.symbol,
      data: prices,
      period,
      from: fromDate.toISOString().split('T')[0],
      to: toDate.toISOString().split('T')[0],
      count: prices.length,
    });
  } catch (error) {
    next(error);
  }
});

// Helper function to calculate period start date
function calculatePeriodStart(period, endDate) {
  const date = new Date(endDate);
  switch (period) {
    case '1d':
      date.setDate(date.getDate() - 1);
      break;
    case '1w':
      date.setDate(date.getDate() - 7);
      break;
    case '1m':
      date.setMonth(date.getMonth() - 1);
      break;
    case '3m':
      date.setMonth(date.getMonth() - 3);
      break;
    case '6m':
      date.setMonth(date.getMonth() - 6);
      break;
    case '1y':
      date.setFullYear(date.getFullYear() - 1);
      break;
    case 'all':
      date.setFullYear(2000);
      break;
    default:
      date.setFullYear(date.getFullYear() - 1);
  }
  return date;
}

export default router;
