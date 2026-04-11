/**
 * Pollens AI Service (Pollinations.ai)
 * 
 * Integration with Pollinations.ai free API for predictive modeling and trend analysis.
 * Uses the OpenAI-compatible REST endpoint — no API key required.
 * 
 * @see https://text.pollinations.ai
 */

import { aiConfig } from '../config/ai.config.js';

class PollensService {
  constructor() {
    this.baseUrl = 'https://text.pollinations.ai';
    this.isInitialized = false;
    this.requestCount = 0;
    this.lastRequestTime = null;
    this.maxRetries = 3;
    this.retryDelay = 1000;
  }

  /**
   * Initialize the Pollens API client
   */
  initialize() {
    if (this.isInitialized) return true;

    try {
      this.baseUrl = aiConfig.pollens?.baseUrl || 'https://text.pollinations.ai';
      this.isInitialized = true;
      console.log('Pollens AI (Pollinations.ai): Service initialized — no API key required');
      return true;
    } catch (error) {
      console.error('Pollens AI: Failed to initialize', error.message);
      return false;
    }
  }

  /**
   * Check rate limits
   */
  checkRateLimit() {
    const now = Date.now();
    const oneMinute = 60 * 1000;

    if (this.lastRequestTime && (now - this.lastRequestTime) < oneMinute) {
      if (this.requestCount >= aiConfig.rateLimit.maxRequestsPerMinute) {
        throw new Error('Rate limit exceeded: Maximum requests per minute reached');
      }
    } else {
      this.requestCount = 0;
    }

    this.requestCount++;
    this.lastRequestTime = now;
    return true;
  }

  /**
   * Make a POST request to the Pollinations.ai OpenAI-compatible endpoint
   * 
   * @param {Array} messages - Chat messages array [{role, content}]
   * @param {Object} options - Additional options
   * @returns {Promise<string>} AI response text
   */
  async chat(messages, options = {}) {
    this.initialize();
    this.checkRateLimit();

    const payload = {
      messages,
      model: options.model || 'openai',
      seed: options.seed || Math.floor(Math.random() * 100000),
      jsonMode: options.jsonMode || false,
    };

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await fetch(this.baseUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(aiConfig.pollens?.timeout || 30000),
        });

        if (!response.ok) {
          throw new Error(`Pollinations API error: ${response.status} ${response.statusText}`);
        }

        const text = await response.text();
        
        // Try to parse as JSON if jsonMode was requested
        if (options.jsonMode) {
          try {
            return JSON.parse(text);
          } catch {
            // If JSON parsing fails, try to extract JSON from the text
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (jsonMatch) return JSON.parse(jsonMatch[0]);
          }
        }

        return text;
      } catch (error) {
        if (attempt < this.maxRetries && this.isRetryableError(error)) {
          console.log(`Pollens AI: Retrying request (${attempt + 1}/${this.maxRetries})...`);
          await this.delay(this.retryDelay * (attempt + 1));
          continue;
        }
        throw error;
      }
    }
  }

  /**
   * Predict stock price trends using Pollinations.ai
   * 
   * @param {Object} params - Prediction parameters
   * @param {string} params.symbol - Stock symbol
   * @param {number[]} params.priceHistory - Array of historical prices
   * @param {number} params.daysAhead - Number of days to predict (default: 5)
   * @returns {Promise<Object>} Prediction result
   */
  async predictPrice({ symbol, priceHistory, daysAhead = 5 }) {
    if (!priceHistory || priceHistory.length < 10) {
      throw new Error('Insufficient price history (minimum 10 data points)');
    }

    // Use last 60 prices max for context window efficiency
    const recentPrices = priceHistory.slice(-60);
    const avgPrice = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
    const lastPrice = recentPrices[recentPrices.length - 1];
    const priceChange = ((lastPrice - recentPrices[0]) / recentPrices[0] * 100).toFixed(2);

    try {
      const response = await this.chat([
        {
          role: 'system',
          content: `You are a quantitative financial analyst specializing in NEPSE (Nepal Stock Exchange).
Analyze price data and provide predictions. Always respond in valid JSON format with these fields:
{
  "symbol": "string",
  "predictions": [{"day": number, "price": number, "confidence": number}],
  "trend_direction": "bullish|bearish|neutral",
  "confidence": number (0-1),
  "summary": "string",
  "risk_level": "low|medium|high"
}`
        },
        {
          role: 'user',
          content: `Analyze ${symbol} stock:
- Recent prices (last ${recentPrices.length} days): ${recentPrices.slice(-10).join(', ')}
- Current price: ${lastPrice}
- Average price: ${avgPrice.toFixed(2)}
- Period change: ${priceChange}%
- Forecast horizon: ${daysAhead} days

Provide price predictions and trend analysis in JSON.`
        }
      ], { jsonMode: true });

      return this.parsePredictionResponse(response, symbol);
    } catch (error) {
      return this.handleError(error, 'predictPrice');
    }
  }

  /**
   * Analyze trend patterns using Pollinations.ai
   * 
   * @param {Object} params - Analysis parameters
   * @param {string} params.symbol - Stock symbol
   * @param {Object[]} params.candleData - Array of OHLCV candle data
   * @returns {Promise<Object>} Trend analysis result
   */
  async analyzeTrend({ symbol, candleData }) {
    if (!candleData || candleData.length < 5) {
      throw new Error('Insufficient candle data (minimum 5 data points)');
    }

    const recentCandles = candleData.slice(-20);

    try {
      const response = await this.chat([
        {
          role: 'system',
          content: `You are a technical analysis expert for NEPSE stocks.
Analyze OHLCV candle data and identify patterns. Respond in valid JSON:
{
  "symbol": "string",
  "trend_direction": "bullish|bearish|neutral",
  "trend_strength": number (0-100),
  "patterns": ["string"],
  "support_levels": [number],
  "resistance_levels": [number],
  "confidence": number (0-1),
  "summary": "string"
}`
        },
        {
          role: 'user',
          content: `Analyze ${symbol} candle data:
${recentCandles.map(c => `O:${c.open} H:${c.high} L:${c.low} C:${c.close} V:${c.volume}`).join('\n')}

Identify trend, patterns, and key support/resistance levels.`
        }
      ], { jsonMode: true });

      return this.parseTrendResponse(response, symbol);
    } catch (error) {
      return this.handleError(error, 'analyzeTrend');
    }
  }

  /**
   * Detect anomalies in price data
   */
  async detectAnomalies({ symbol, priceHistory }) {
    if (!priceHistory || priceHistory.length < 10) {
      throw new Error('Insufficient price history');
    }

    try {
      const response = await this.chat([
        {
          role: 'system',
          content: `You are a financial anomaly detection specialist.
Analyze price data for unusual patterns, spikes, or irregularities. Respond in JSON:
{
  "anomalies": [{"index": number, "price": number, "type": "string", "severity": "low|medium|high"}],
  "confidence": number (0-1),
  "summary": "string"
}`
        },
        {
          role: 'user',
          content: `Detect anomalies in ${symbol} price data:
${priceHistory.slice(-30).join(', ')}`
        }
      ], { jsonMode: true });

      const parsed = typeof response === 'string' ? JSON.parse(response) : response;
      return {
        anomalies: parsed.anomalies || [],
        confidence: parsed.confidence || 0,
        summary: parsed.summary || 'No anomalies detected',
      };
    } catch (error) {
      return this.handleError(error, 'detectAnomalies');
    }
  }

  /**
   * Generate a market brief summary
   */
  async generateMarketBrief({ stocks, marketIndex }) {
    try {
      const response = await this.chat([
        {
          role: 'system',
          content: `You are a NEPSE market analyst. Provide a concise daily market brief in JSON:
{
  "headline": "string",
  "sentiment": "bullish|bearish|neutral",
  "top_movers": [{"symbol": "string", "reason": "string"}],
  "key_levels": {"support": number, "resistance": number},
  "outlook": "string",
  "risk_factors": ["string"]
}`
        },
        {
          role: 'user',
          content: `Generate market brief:
NEPSE Index: ${marketIndex || 'N/A'}
Active stocks: ${JSON.stringify(stocks?.slice(0, 10) || [])}`
        }
      ], { jsonMode: true });

      return typeof response === 'string' ? JSON.parse(response) : response;
    } catch (error) {
      return this.handleError(error, 'generateMarketBrief');
    }
  }

  // --- Helper methods ---

  isRetryableError(error) {
    if (error.name === 'AbortError') return true;
    if (!error.response) return true;
    const status = error.response?.status;
    return status === 429 || status >= 500;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  parsePredictionResponse(data, symbol) {
    const parsed = typeof data === 'string' ? JSON.parse(data) : data;
    return {
      symbol: parsed.symbol || symbol,
      predictions: parsed.predictions || [],
      trendDirection: parsed.trend_direction || 'neutral',
      confidence: Math.max(0, Math.min(1, parsed.confidence || 0.5)),
      summary: parsed.summary || '',
      riskLevel: parsed.risk_level || 'medium',
      forecastHorizon: parsed.predictions?.length || 5,
      modelVersion: 'pollinations-v1',
      timestamp: new Date().toISOString(),
    };
  }

  parseTrendResponse(data, symbol) {
    const parsed = typeof data === 'string' ? JSON.parse(data) : data;
    return {
      symbol: parsed.symbol || symbol,
      trendDirection: parsed.trend_direction || 'neutral',
      trendStrength: parsed.trend_strength || 0,
      patterns: parsed.patterns || [],
      supportLevels: parsed.support_levels || [],
      resistanceLevels: parsed.resistance_levels || [],
      confidence: Math.max(0, Math.min(1, parsed.confidence || 0.5)),
      summary: parsed.summary || '',
      timestamp: new Date().toISOString(),
    };
  }

  handleError(error, operation) {
    console.error(`Pollens AI Error [${operation}]:`, error.message);
    return {
      error: true,
      operation,
      message: error.message,
      fallback: true,
      predictions: [],
      trendDirection: 'neutral',
      confidence: 0,
      timestamp: new Date().toISOString(),
    };
  }

  getStatus() {
    return {
      initialized: this.isInitialized,
      provider: 'Pollinations.ai',
      apiKeyRequired: false,
      baseUrl: this.baseUrl,
      requestCount: this.requestCount,
      lastRequestTime: this.lastRequestTime,
    };
  }
}

export default new PollensService();
