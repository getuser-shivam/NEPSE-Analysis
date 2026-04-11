/**
 * Pollens AI Service
 * 
 * Integration with Pollens AI API for predictive modeling and trend analysis.
 * Implements REST API client with proper error handling and retry logic.
 */

import axios from 'axios';
import { aiConfig } from '../config/ai.config.js';

class PollensService {
  constructor() {
    this.client = null;
    this.isInitialized = false;
    this.requestCount = 0;
    this.lastRequestTime = null;
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1 second
  }

  /**
   * Initialize the Pollens API client
   */
  initialize() {
    if (this.isInitialized) return true;

    if (!aiConfig.pollens.apiKey) {
      console.warn('Pollens AI: API key not configured');
      return false;
    }

    try {
      this.client = axios.create({
        baseURL: aiConfig.pollens.baseUrl,
        timeout: aiConfig.pollens.timeout,
        headers: {
          'Authorization': `Bearer ${aiConfig.pollens.apiKey}`,
          'Content-Type': 'application/json',
        },
      });

      // Add response interceptor for error handling
      this.client.interceptors.response.use(
        (response) => response,
        (error) => this.handleAxiosError(error)
      );

      this.isInitialized = true;
      console.log('Pollens AI: Service initialized successfully');
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
   * Predict stock price using Pollens AI
   * 
   * @param {Object} params - Prediction parameters
   * @param {string} params.symbol - Stock symbol
   * @param {number[]} params.priceHistory - Array of historical prices
   * @param {number} params.daysAhead - Number of days to predict (default: 5)
   * @returns {Promise<Object>} Prediction result
   */
  async predictPrice({ symbol, priceHistory, daysAhead = 5 }) {
    this.initialize();
    this.checkRateLimit();

    if (!this.isInitialized) {
      throw new Error('Pollens AI service not initialized');
    }

    if (!priceHistory || priceHistory.length < 30) {
      throw new Error('Insufficient price history data (minimum 30 data points required)');
    }

    const requestData = {
      symbol,
      price_history: priceHistory,
      forecast_horizon: daysAhead,
      model: 'financial_time_series_v2',
    };

    try {
      const response = await this.makeRequest('post', '/predict', requestData);
      return this.parsePredictionResponse(response.data);
    } catch (error) {
      return this.handleError(error, 'predictPrice');
    }
  }

  /**
   * Analyze trend patterns using Pollens AI
   * 
   * @param {Object} params - Analysis parameters
   * @param {string} params.symbol - Stock symbol
   * @param {Object[]} params.candleData - Array of OHLCV candle data
   * @returns {Promise<Object>} Trend analysis result
   */
  async analyzeTrend({ symbol, candleData }) {
    this.initialize();
    this.checkRateLimit();

    if (!this.isInitialized) {
      throw new Error('Pollens AI service not initialized');
    }

    if (!candleData || candleData.length < 20) {
      throw new Error('Insufficient candle data (minimum 20 data points required)');
    }

    const requestData = {
      symbol,
      data: candleData,
      analysis_type: 'trend_pattern',
    };

    try {
      const response = await this.makeRequest('post', '/analyze/trend', requestData);
      return this.parseTrendResponse(response.data);
    } catch (error) {
      return this.handleError(error, 'analyzeTrend');
    }
  }

  /**
   * Detect anomalies in price data
   * 
   * @param {Object} params - Anomaly detection parameters
   * @param {string} params.symbol - Stock symbol
   * @param {number[]} params.priceHistory - Array of historical prices
   * @returns {Promise<Object>} Anomaly detection result
   */
  async detectAnomalies({ symbol, priceHistory }) {
    this.initialize();
    this.checkRateLimit();

    if (!this.isInitialized) {
      throw new Error('Pollens AI service not initialized');
    }

    const requestData = {
      symbol,
      price_history: priceHistory,
      analysis_type: 'anomaly_detection',
    };

    try {
      const response = await this.makeRequest('post', '/analyze/anomalies', requestData);
      return {
        anomalies: response.data.anomalies || [],
        confidence: response.data.confidence || 0,
        summary: response.data.summary || 'No anomalies detected',
      };
    } catch (error) {
      return this.handleError(error, 'detectAnomalies');
    }
  }

  /**
   * Make API request with retry logic
   */
  async makeRequest(method, endpoint, data, retries = 0) {
    try {
      return await this.client.request({
        method,
        url: endpoint,
        data,
      });
    } catch (error) {
      if (retries < this.maxRetries && this.isRetryableError(error)) {
        console.log(`Retrying request (${retries + 1}/${this.maxRetries})...`);
        await this.delay(this.retryDelay * (retries + 1));
        return this.makeRequest(method, endpoint, data, retries + 1);
      }
      throw error;
    }
  }

  /**
   * Check if error is retryable
   */
  isRetryableError(error) {
    if (!error.response) return true; // Network errors are retryable
    const status = error.response.status;
    return status === 429 || status >= 500; // Rate limit or server errors
  }

  /**
   * Delay utility
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Parse prediction response
   */
  parsePredictionResponse(data) {
    return {
      symbol: data.symbol,
      predictions: data.predictions || [],
      trendDirection: data.trend_direction || 'neutral',
      confidence: Math.max(0, Math.min(1, data.confidence || 0.5)),
      forecastHorizon: data.forecast_horizon || 5,
      modelVersion: data.model_version || 'unknown',
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Parse trend analysis response
   */
  parseTrendResponse(data) {
    return {
      symbol: data.symbol,
      trendDirection: data.trend_direction || 'neutral',
      trendStrength: data.trend_strength || 0,
      patterns: data.patterns || [],
      supportLevels: data.support_levels || [],
      resistanceLevels: data.resistance_levels || [],
      confidence: Math.max(0, Math.min(1, data.confidence || 0.5)),
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Handle Axios errors
   */
  handleAxiosError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      console.error(`Pollens API Error ${status}:`, data);
      
      if (status === 401) {
        throw new Error('Invalid API key');
      } else if (status === 429) {
        throw new Error('Rate limit exceeded');
      } else if (status >= 500) {
        throw new Error('Pollens AI service temporarily unavailable');
      }
    } else if (error.request) {
      // Request made but no response received
      console.error('No response from Pollens API:', error.message);
      throw new Error('Network error: Unable to reach Pollens AI service');
    }
    
    return Promise.reject(error);
  }

  /**
   * Handle errors with proper logging and fallback
   */
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

  /**
   * Get service status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      apiConfigured: !!aiConfig.pollens.apiKey,
      baseUrl: aiConfig.pollens.baseUrl,
      requestCount: this.requestCount,
      lastRequestTime: this.lastRequestTime,
    };
  }
}

export default new PollensService();
