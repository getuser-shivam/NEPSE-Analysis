/**
 * Groq AI Service
 * 
 * Integration with Groq AI API for market analysis and predictions.
 * Uses Groq SDK for API communication with proper error handling.
 */

import Groq from 'groq-sdk';
import { aiConfig } from '../config/ai.config.js';

class GroqService {
  constructor() {
    this.client = null;
    this.isInitialized = false;
    this.requestCount = 0;
    this.lastRequestTime = null;
  }

  /**
   * Initialize the Groq client
   */
  initialize() {
    if (this.isInitialized) return true;

    if (!aiConfig.groq.apiKey) {
      console.warn('Groq AI: API key not configured');
      return false;
    }

    try {
      this.client = new Groq({
        apiKey: aiConfig.groq.apiKey,
      });
      this.isInitialized = true;
      console.log('Groq AI: Service initialized successfully');
      return true;
    } catch (error) {
      console.error('Groq AI: Failed to initialize', error.message);
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
   * Analyze market data using Groq AI
   * 
   * @param {Object} params - Analysis parameters
   * @param {string} params.symbol - Stock symbol
   * @param {Object} params.marketData - Market data including prices, volumes
   * @param {Object} params.indicators - Technical indicators
   * @returns {Promise<Object>} AI analysis result
   */
  async analyzeMarket({ symbol, marketData, indicators }) {
    this.initialize();
    this.checkRateLimit();

    if (!this.isInitialized) {
      throw new Error('Groq AI service not initialized');
    }

    const prompt = this.buildAnalysisPrompt(symbol, marketData, indicators);

    try {
      const response = await this.client.chat.completions.create({
        model: aiConfig.groq.model,
        messages: [
          {
            role: 'system',
            content: `You are a financial analyst specializing in NEPSE (Nepal Stock Exchange) market analysis. 
Provide concise analysis with sentiment (bullish/bearish/neutral), confidence level (0-1), 
and key factors. Respond in JSON format with fields: sentiment, confidence, prediction, factors (array), recommendation.`,
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: aiConfig.groq.temperature,
        max_tokens: aiConfig.groq.maxTokens,
        response_format: { type: 'json_object' },
      });

      const content = response.choices[0]?.message?.content;
      if (!content) {
        throw new Error('Empty response from Groq AI');
      }

      return this.parseAnalysisResponse(content);
    } catch (error) {
      return this.handleError(error, 'analyzeMarket');
    }
  }

  /**
   * Generate trading recommendation
   * 
   * @param {Object} params - Recommendation parameters
   * @param {string} params.symbol - Stock symbol
   * @param {number} params.currentPrice - Current stock price
   * @param {Object} params.portfolio - Portfolio information
   * @param {string} params.riskProfile - Risk profile (conservative/moderate/aggressive)
   * @returns {Promise<string>} Recommendation text
   */
  async generateRecommendation({ symbol, currentPrice, portfolio, riskProfile }) {
    this.initialize();
    this.checkRateLimit();

    if (!this.isInitialized) {
      throw new Error('Groq AI service not initialized');
    }

    try {
      const response = await this.client.chat.completions.create({
        model: aiConfig.groq.model,
        messages: [
          {
            role: 'system',
            content: 'You are an investment advisor. Provide brief, actionable recommendations (max 100 words).',
          },
          {
            role: 'user',
            content: `Stock: ${symbol}, Price: ${currentPrice}, Risk Profile: ${riskProfile}, Portfolio: ${JSON.stringify(portfolio)}. 
Should I buy, sell, or hold? Provide brief reasoning.`,
          },
        ],
        temperature: 0.4,
        max_tokens: 200,
      });

      return response.choices[0]?.message?.content || 'No recommendation available';
    } catch (error) {
      return this.handleError(error, 'generateRecommendation');
    }
  }

  /**
   * Build analysis prompt
   */
  buildAnalysisPrompt(symbol, marketData, indicators) {
    return `Analyze ${symbol} stock based on the following data:

Market Data:
${JSON.stringify(marketData, null, 2)}

Technical Indicators:
${JSON.stringify(indicators, null, 2)}

Provide analysis in JSON format with:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: number between 0 and 1
- prediction: brief price movement prediction
- factors: array of key influencing factors
- recommendation: brief trading recommendation`;
  }

  /**
   * Parse AI analysis response
   */
  parseAnalysisResponse(content) {
    try {
      const result = JSON.parse(content);
      return {
        sentiment: result.sentiment || 'neutral',
        confidence: Math.max(0, Math.min(1, result.confidence || 0.5)),
        prediction: result.prediction || 'No prediction available',
        factors: Array.isArray(result.factors) ? result.factors : [],
        recommendation: result.recommendation || null,
        rawData: result,
      };
    } catch (error) {
      console.error('Failed to parse AI response:', error);
      return {
        sentiment: 'neutral',
        confidence: 0,
        prediction: 'Failed to parse AI response',
        factors: [],
        recommendation: null,
        error: error.message,
      };
    }
  }

  /**
   * Handle errors with proper logging and fallback
   */
  handleError(error, operation) {
    console.error(`Groq AI Error [${operation}]:`, error.message);

    // Log detailed error for debugging
    if (error.response) {
      console.error('API Response:', error.response.data);
    }

    // Return fallback response
    return {
      error: true,
      operation,
      message: error.message,
      fallback: true,
      sentiment: 'neutral',
      confidence: 0,
      prediction: 'AI analysis temporarily unavailable',
      factors: ['AI service error'],
    };
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      apiConfigured: !!aiConfig.groq.apiKey,
      requestCount: this.requestCount,
      lastRequestTime: this.lastRequestTime,
    };
  }
}

export default new GroqService();
