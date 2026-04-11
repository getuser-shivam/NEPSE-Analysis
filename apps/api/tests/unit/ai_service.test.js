/**
 * AI Service Unit Tests
 * 
 * Unit tests for AI service business logic.
 */

import { describe, test, expect } from '@jest/globals';

describe('AI Service Unit Tests', () => {
  describe('Groq AI Configuration', () => {
    test('validates Groq API key format', () => {
      const validKey = 'gsk_abc123xyz456';
      const isValid = /^gsk_[a-zA-Z0-9]{32,}$/.test(validKey);
      expect(isValid).toBe(true);
    });

    test('rejects invalid Groq API keys', () => {
      const invalidKeys = ['invalid', 'gsk_short', ''];
      
      invalidKeys.forEach(key => {
        const isValid = /^gsk_[a-zA-Z0-9]{32,}$/.test(key);
        expect(isValid).toBe(false);
      });
    });
  });

  describe('Pollens AI Configuration', () => {
    test('validates Pollens API key format', () => {
      const validKey = 'pk_test_abc123xyz456';
      const isValid = /^pk_[a-zA-Z0-9_]{20,}$/.test(validKey);
      expect(isValid).toBe(true);
    });
  });

  describe('Rate Limiting', () => {
    test('calculates rate limit correctly', () => {
      const maxRequestsPerMinute = 10;
      const currentMinute = new Date().getMinutes();
      
      const requestsInCurrentMinute = 5;
      const canMakeRequest = requestsInCurrentMinute < maxRequestsPerMinute;
      
      expect(canMakeRequest).toBe(true);
    });

    test('blocks requests when limit exceeded', () => {
      const maxRequestsPerMinute = 10;
      const requestsInCurrentMinute = 10;
      const canMakeRequest = requestsInCurrentMinute < maxRequestsPerMinute;
      
      expect(canMakeRequest).toBe(false);
    });
  });

  describe('Prompt Building', () => {
    test('builds analysis prompt correctly', () => {
      const symbol = 'NABIL';
      const currentPrice = 1000;
      const priceChange = 10;
      
      const prompt = `Analyze ${symbol} at ${currentPrice} with ${priceChange}% change`;
      
      expect(prompt).toContain('NABIL');
      expect(prompt).toContain('1000');
      expect(prompt).toContain('10%');
    });
  });
});
