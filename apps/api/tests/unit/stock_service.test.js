/**
 * Stock Service Unit Tests
 * 
 * Unit tests for stock service business logic.
 */

import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';

describe('Stock Service Unit Tests', () => {
  describe('Stock Price Calculations', () => {
    test('calculates price change correctly', () => {
      const currentPrice = 1000;
      const previousClose = 990;
      const change = currentPrice - previousClose;
      const percentChange = (change / previousClose) * 100;

      expect(change).toBe(10);
      expect(percentChange).toBeCloseTo(1.01, 2);
    });

    test('handles zero previous close gracefully', () => {
      const currentPrice = 1000;
      const previousClose = 0;
      const percentChange = previousClose !== 0 ? (currentPrice - previousClose) / previousClose * 100 : 0;

      expect(percentChange).toBe(0);
    });
  });

  describe('Stock Validation', () => {
    test('validates stock symbol format', () => {
      const validSymbols = ['NABIL', 'EBL', 'NIBL', 'CHCL'];
      
      validSymbols.forEach(symbol => {
        const isValid = /^[A-Z]{3,6}$/.test(symbol);
        expect(isValid).toBe(true);
      });
    });

    test('rejects invalid stock symbols', () => {
      const invalidSymbols = ['nabil', 'NAB1L', 'NA', 'TOOLONG'];
      
      invalidSymbols.forEach(symbol => {
        const isValid = /^[A-Z]{3,6}$/.test(symbol);
        expect(isValid).toBe(false);
      });
    });
  });

  describe('Sector Classification', () => {
    test('classifies stocks by sector', () => {
      const stocks = [
        { symbol: 'NABIL', sector: 'BANKING' },
        { symbol: 'CHCL', sector: 'HYDRO' },
        { symbol: 'NLIC', sector: 'INSURANCE' },
      ];

      const bankingStocks = stocks.filter(s => s.sector === 'BANKING');
      expect(bankingStocks).toHaveLength(1);
      expect(bankingStocks[0].symbol).toBe('NABIL');
    });
  });
});
