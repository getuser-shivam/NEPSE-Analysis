/**
 * Analysis Routes
 * 
 * API endpoints for technical analysis and AI-powered market analysis.
 * 
 * Routes:
 * - POST /          - Perform comprehensive analysis
 * - POST /technical - Calculate technical indicators only
 * - POST /ai        - Get AI-powered analysis only
 */

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';
import groqService from '../services/groq.service.js';
import pollensService from '../services/pollens.service.js';
import { aiConfig } from '../config/ai.config.js';

const router = Router();
const prisma = new PrismaClient();

// Validation schemas
const analysisSchema = z.object({
  symbol: z.string().min(1).max(20),
  period: z.enum(['1d', '1w', '1m', '3m', '6m', '1y']).default('3m'),
  indicators: z.array(z.enum(['rsi', 'macd', 'bollinger', 'stochastic', 'williamsR', 'atr'])).optional(),
  enableAI: z.boolean().default(true),
});

const technicalSchema = z.object({
  symbol: z.string().min(1).max(20),
  period: z.enum(['1d', '1w', '1m', '3m', '6m', '1y']).default('3m'),
  indicators: z.array(z.string()).optional(),
});

/**
 * @swagger
 * /api/analysis:
 *   post:
 *     summary: Perform comprehensive market analysis
 *     description: Calculate technical indicators and get AI-powered analysis for a stock
 *     tags: [Analysis]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               symbol:
 *                 type: string
 *                 example: NABIL
 *               period:
 *                 type: string
 *                 enum: [1d, 1w, 1m, 3m, 6m, 1y]
 *                 default: 3m
 *               indicators:
 *                 type: array
 *                 items:
 *                   type: string
 *                   enum: [rsi, macd, bollinger, stochastic, williamsR, atr]
 *               enableAI:
 *                 type: boolean
 *                 default: true
 *     responses:
 *       200:
 *         description: Analysis completed successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 symbol:
 *                   type: string
 *                 currentPrice:
 *                   type: number
 *                 technicalIndicators:
 *                   type: object
 *                 aiAnalysis:
 *                   type: object
 *                 recommendation:
 *                   type: string
 *                 confidence:
 *                   type: number
 *       400:
 *         description: Invalid request parameters
 *       404:
 *         description: Stock not found
 *       500:
 *         description: Analysis error
 */
router.post('/', async (request, response, next) => {
  try {
    const { symbol, period, indicators, enableAI } = analysisSchema.parse(request.body);

    // Get stock from database
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
    const { fromDate, toDate } = calculateDateRange(period);

    // Get price history
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

    if (prices.length < 30) {
      return response.status(400).json({
        error: 'Insufficient price data',
        required: 'Minimum 30 data points',
        available: prices.length,
      });
    }

    // Extract price data
    const closes = prices.map(p => p.close);
    const highs = prices.map(p => p.high);
    const lows = prices.map(p => p.low);

    // Calculate technical indicators
    const technicalIndicators = calculateTechnicalIndicators(
      closes,
      highs,
      lows,
      indicators
    );

    // Get AI analysis if enabled
    let aiAnalysis = null;
    if (enableAI) {
      aiAnalysis = await getAIAnalysis(stock.symbol, {
        closes, highs, lows,
        currentPrice: closes[closes.length - 1],
        technicalIndicators,
      });
    }

    // Generate recommendation
    const recommendation = generateRecommendation(technicalIndicators, aiAnalysis);
    const confidence = calculateConfidence(technicalIndicators, aiAnalysis);

    response.json({
      symbol: stock.symbol,
      currentPrice: closes.last,
      technicalIndicators,
      aiAnalysis,
      recommendation,
      confidence,
      period,
      dataPoints: prices.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/analysis/technical:
 *   post:
 *     summary: Calculate technical indicators only
 *     description: Get technical analysis without AI insights
 *     tags: [Analysis]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               symbol:
 *                 type: string
 *               period:
 *                 type: string
 *                 enum: [1d, 1w, 1m, 3m, 6m, 1y]
 *               indicators:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       200:
 *         description: Technical analysis completed
 */
router.post('/technical', async (request, response, next) => {
  try {
    const { symbol, period, indicators } = technicalSchema.parse(request.body);

    const stock = await prisma.stock.findUnique({
      where: { symbol: symbol.toUpperCase() },
    });

    if (!stock) {
      return response.status(404).json({
        error: 'Stock not found',
        symbol: symbol.toUpperCase(),
      });
    }

    const { fromDate, toDate } = calculateDateRange(period);

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

    if (prices.length < 30) {
      return response.status(400).json({
        error: 'Insufficient price data',
        required: 30,
        available: prices.length,
      });
    }

    const closes = prices.map(p => p.close);
    const highs = prices.map(p => p.high);
    const lows = prices.map(p => p.low);

    const technicalIndicators = calculateTechnicalIndicators(
      closes,
      highs,
      lows,
      indicators
    );

    response.json({
      symbol: stock.symbol,
      technicalIndicators,
      dataPoints: prices.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

// Helper functions
function calculateDateRange(period) {
  const toDate = new Date();
  const fromDate = new Date();

  switch (period) {
    case '1d':
      fromDate.setDate(toDate.getDate() - 1);
      break;
    case '1w':
      fromDate.setDate(toDate.getDate() - 7);
      break;
    case '1m':
      fromDate.setMonth(toDate.getMonth() - 1);
      break;
    case '3m':
      fromDate.setMonth(toDate.getMonth() - 3);
      break;
    case '6m':
      fromDate.setMonth(toDate.getMonth() - 6);
      break;
    case '1y':
      fromDate.setFullYear(toDate.getFullYear() - 1);
      break;
  }

  return { fromDate, toDate };
}

function calculateTechnicalIndicators(closes, highs, lows, requestedIndicators) {
  const indicators = {};

  // Always calculate basic indicators
  indicators.rsi14 = calculateRSI(closes, 14);
  indicators.rsi7 = calculateRSI(closes, 7);

  // Calculate requested indicators
  if (!requestedIndicators || requestedIndicators.includes('macd')) {
    const macd = calculateMACD(closes);
    indicators.macdLine = macd.line.last;
    indicators.macdSignal = macd.signal.last;
    indicators.macdHistogram = macd.histogram.last;
  }

  if (!requestedIndicators || requestedIndicators.includes('bollinger')) {
    const bb = calculateBollingerBands(closes);
    indicators.bbUpper = bb.upper.last;
    indicators.bbMiddle = bb.middle.last;
    indicators.bbLower = bb.lower.last;
  }

  if (!requestedIndicators || requestedIndicators.includes('stochastic')) {
    const stoch = calculateStochastic(highs, lows, closes);
    indicators.stochK = stoch.k.last;
    indicators.stochD = stoch.d.last;
  }

  indicators.sma20 = calculateSMA(closes, 20).last;
  indicators.sma50 = calculateSMA(closes, 50).last;
  indicators.ema12 = calculateEMA(closes, 12).last;
  indicators.ema26 = calculateEMA(closes, 26).last;

  // Generate signal
  indicators.signal = generateSignal(indicators);

  return indicators;
}

// Simplified indicator calculations (in production, import from calculations module)
function calculateRSI(prices, period) {
  if (prices.length < period + 1) return 50;
  
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateMACD(prices) {
  // Simplified MACD calculation
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  const line = [];
  
  for (let i = 0; i < Math.min(ema12.length, ema26.length); i++) {
    line.push(ema12[i] - ema26[i]);
  }
  
  const signal = calculateEMA(line, 9);
  const histogram = [];
  
  for (let i = 0; i < signal.length; i++) {
    histogram.push(line[line.length - signal.length + i] - signal[i]);
  }
  
  return { line, signal, histogram };
}

function calculateSMA(prices, period) {
  const result = [];
  for (let i = period - 1; i < prices.length; i++) {
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += prices[i - j];
    }
    result.push(sum / period);
  }
  return result;
}

function calculateEMA(prices, period) {
  const multiplier = 2 / (period + 1);
  const result = [];
  
  let ema = prices.slice(0, period).reduce((a, b) => a + b) / period;
  result.push(ema);
  
  for (let i = period; i < prices.length; i++) {
    ema = (prices[i] - ema) * multiplier + ema;
    result.push(ema);
  }
  
  return result;
}

function calculateBollingerBands(prices, period = 20) {
  const sma = calculateSMA(prices, period);
  const upper = [];
  const lower = [];
  
  for (let i = period - 1; i < prices.length; i++) {
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += Math.pow(prices[i - j] - sma[i - period + 1], 2);
    }
    const stdDev = Math.sqrt(sum / period);
    
    upper.push(sma[i - period + 1] + (stdDev * 2));
    lower.push(sma[i - period + 1] - (stdDev * 2));
  }
  
  return { upper, middle: sma, lower };
}

function calculateStochastic(highs, lows, closes) {
  const k = [];
  const period = 14;
  
  for (let i = period - 1; i < closes.length; i++) {
    const highestHigh = Math.max(...highs.slice(i - period + 1, i + 1));
    const lowestLow = Math.min(...lows.slice(i - period + 1, i + 1));
    
    if (highestHigh === lowestLow) {
      k.push(50);
    } else {
      k.push(((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100);
    }
  }
  
  const d = calculateSMA(k, 3);
  
  return { k, d };
}

function generateSignal(indicators) {
  let buyScore = 0;
  let sellScore = 0;
  
  if (indicators.rsi14 < 30) buyScore += 2;
  if (indicators.rsi14 > 70) sellScore += 2;
  if (indicators.macdHistogram > 0) buyScore += 1;
  if (indicators.macdHistogram < 0) sellScore += 1;
  if (indicators.stochK < 20) buyScore += 1;
  if (indicators.stochK > 80) sellScore += 1;
  
  if (buyScore >= 4) return 'strong_buy';
  if (buyScore >= 2) return 'buy';
  if (sellScore >= 4) return 'strong_sell';
  if (sellScore >= 2) return 'sell';
  return 'hold';
}

function analyzeSentiment(indicators) {
  if (indicators.rsi14 > 60 && indicators.macdHistogram > 0) return 'bullish';
  if (indicators.rsi14 < 40 && indicators.macdHistogram < 0) return 'bearish';
  return 'neutral';
}

function generateRecommendation(indicators, aiAnalysis) {
  let recommendation = indicators.signal;
  
  if (aiAnalysis && aiAnalysis.confidence > 0.6) {
    if (aiAnalysis.sentiment === 'bullish' && recommendation.includes('sell')) {
      recommendation = 'hold';
    } else if (aiAnalysis.sentiment === 'bearish' && recommendation.includes('buy')) {
      recommendation = 'hold';
    }
  }
  
  return recommendation;
}

function calculateConfidence(indicators, aiAnalysis) {
  let confidence = 0.5;
  
  if (indicators.rsi14 < 30 || indicators.rsi14 > 70) confidence += 0.2;
  
  if (aiAnalysis) {
    confidence = (confidence + (aiAnalysis.confidence || 0.5)) / 2;
  }
  
  return Math.min(confidence, 1.0);
}

// ============================================
// AI ORCHESTRATION
// ============================================

/**
 * Get AI analysis using Groq (primary) with Pollinations.ai fallback
 */
async function getAIAnalysis(symbol, data) {
  const { closes, highs, lows, currentPrice, technicalIndicators } = data;

  // Prepare market data summary for the AI
  const marketData = {
    currentPrice,
    priceChange: closes.length > 1
      ? ((closes[closes.length - 1] - closes[closes.length - 2]) / closes[closes.length - 2] * 100).toFixed(2) + '%'
      : '0%',
    high52w: Math.max(...highs.slice(-252)),
    low52w: Math.min(...lows.slice(-252)),
    avgVolume: 'N/A',
  };

  // Try Groq first (fastest inference)
  if (aiConfig.features.enableGroq) {
    try {
      const groqResult = await groqService.analyzeMarket({
        symbol,
        marketData,
        indicators: technicalIndicators,
      });
      if (!groqResult.error) {
        groqResult.provider = 'groq';
        return groqResult;
      }
    } catch (err) {
      console.warn('Groq AI unavailable, falling back to Pollens:', err.message);
    }
  }

  // Fallback to Pollinations.ai
  if (aiConfig.features.enablePollens) {
    try {
      const pollensResult = await pollensService.predictPrice({
        symbol,
        priceHistory: closes,
        daysAhead: 5,
      });
      if (!pollensResult.error) {
        pollensResult.provider = 'pollinations';
        return {
          sentiment: pollensResult.trendDirection === 'bullish' ? 'bullish'
            : pollensResult.trendDirection === 'bearish' ? 'bearish' : 'neutral',
          confidence: pollensResult.confidence,
          prediction: pollensResult.summary || 'AI prediction generated',
          factors: [`Trend: ${pollensResult.trendDirection}`, `Risk: ${pollensResult.riskLevel}`],
          predictions: pollensResult.predictions,
          provider: 'pollinations',
        };
      }
    } catch (err) {
      console.warn('Pollens AI unavailable:', err.message);
    }
  }

  // Last resort: static analysis from indicators
  return {
    sentiment: analyzeSentiment(technicalIndicators),
    confidence: 0.5,
    prediction: 'AI services unavailable — using indicator-based analysis',
    factors: getIndicatorFactors(technicalIndicators),
    provider: 'static',
  };
}

/**
 * Extract human-readable factors from technical indicators
 */
function getIndicatorFactors(indicators) {
  const factors = [];
  if (indicators.rsi14 < 30) factors.push('RSI oversold — potential reversal');
  else if (indicators.rsi14 > 70) factors.push('RSI overbought — potential pullback');
  else factors.push(`RSI neutral at ${indicators.rsi14?.toFixed(1)}`);

  if (indicators.macdHistogram > 0) factors.push('Positive MACD momentum');
  else if (indicators.macdHistogram < 0) factors.push('Negative MACD momentum');

  if (indicators.signal) factors.push(`Signal: ${indicators.signal}`);

  return factors;
}

// ============================================
// AI-SPECIFIC ENDPOINTS
// ============================================

/**
 * POST /api/analysis/ai — Direct AI analysis endpoint
 */
router.post('/ai', async (request, response, next) => {
  try {
    const schema = z.object({
      symbol: z.string().min(1).max(20),
      priceHistory: z.array(z.number()).min(10).optional(),
      candleData: z.array(z.object({
        open: z.number(), high: z.number(),
        low: z.number(), close: z.number(),
        volume: z.number().optional(),
      })).optional(),
      provider: z.enum(['groq', 'pollens', 'auto']).default('auto'),
    });

    const { symbol, priceHistory, candleData, provider } = schema.parse(request.body);

    let result;

    if (provider === 'groq' || (provider === 'auto' && aiConfig.features.enableGroq)) {
      result = await groqService.analyzeMarket({
        symbol,
        marketData: { priceHistory: priceHistory?.slice(-20) },
        indicators: {},
      });
      result.provider = 'groq';
    }

    if ((!result || result.error) && (provider === 'pollens' || provider === 'auto')) {
      if (priceHistory) {
        result = await pollensService.predictPrice({ symbol, priceHistory });
      } else if (candleData) {
        result = await pollensService.analyzeTrend({ symbol, candleData });
      }
      if (result) result.provider = 'pollinations';
    }

    if (!result) {
      return response.status(503).json({ error: 'No AI providers available' });
    }

    response.json({
      symbol: symbol.toUpperCase(),
      analysis: result,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/analysis/ai/predict — Price prediction endpoint
 */
router.post('/ai/predict', async (request, response, next) => {
  try {
    const schema = z.object({
      symbol: z.string().min(1).max(20),
      priceHistory: z.array(z.number()).min(10),
      daysAhead: z.number().min(1).max(30).default(5),
    });

    const { symbol, priceHistory, daysAhead } = schema.parse(request.body);

    const prediction = await pollensService.predictPrice({
      symbol,
      priceHistory,
      daysAhead,
    });

    response.json({
      symbol: symbol.toUpperCase(),
      prediction,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/analysis/ai/brief — Market brief generation
 */
router.post('/ai/brief', async (request, response, next) => {
  try {
    const schema = z.object({
      stocks: z.array(z.object({
        symbol: z.string(),
        price: z.number().optional(),
        change: z.number().optional(),
      })).optional(),
      marketIndex: z.number().optional(),
    });

    const { stocks, marketIndex } = schema.parse(request.body);

    const brief = await pollensService.generateMarketBrief({ stocks, marketIndex });

    response.json({
      brief,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/analysis/ai/status — Check AI service status
 */
router.get('/ai/status', (request, response) => {
  response.json({
    groq: groqService.getStatus(),
    pollens: pollensService.getStatus(),
    features: aiConfig.features,
    rateLimit: aiConfig.rateLimit,
  });
});

export default router;
