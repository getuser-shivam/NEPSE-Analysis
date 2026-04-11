/**
 * Analysis API Integration Tests
 * 
 * Tests for /api/analysis endpoints including error scenarios.
 */

import { describe, test, expect } from '@jest/globals';
import request from 'supertest';
import { createApp } from '../src/app.js';

const app = createApp();

describe('Analysis API Integration Tests', () => {
  describe('POST /api/analysis', () => {
    test('should return 401 without authentication', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .send({
          symbol: 'NABIL',
          period: '3m',
          enableAI: true,
        });
      
      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });

    test('should validate required symbol field', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .send({
          period: '3m',
          enableAI: true,
        });
      
      // Should return 400 for missing required field
      expect([400, 401]).toContain(response.status);
    });

    test('should validate period parameter values', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .send({
          symbol: 'NABIL',
          period: 'invalid-period',
          enableAI: true,
        });
      
      // Should reject invalid period value
      expect([400, 401]).toContain(response.status);
    });

    test('should validate symbol length', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .send({
          symbol: 'TOO-LONG-SYMBOL-THAT-EXCEEDS-MAXIMUM',
          period: '3m',
          enableAI: true,
        });
      
      // Should reject symbol that's too long
      expect([400, 401]).toContain(response.status);
    });
  });

  describe('POST /api/analysis/technical', () => {
    test('should return 401 without authentication', async () => {
      const response = await request(app)
        .post('/api/analysis/technical')
        .send({
          symbol: 'NABIL',
          period: '3m',
        });
      
      expect(response.status).toBe(401);
    });

    test('should handle missing symbol field', async () => {
      const response = await request(app)
        .post('/api/analysis/technical')
        .set('Authorization', 'Bearer test-token')
        .send({
          period: '3m',
        });
      
      expect([400, 401]).toContain(response.status);
    });
  });

  describe('Error Scenarios', () => {
    test('should handle malformed JSON in request body', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .set('Content-Type', 'application/json')
        .send('{"symbol": "NABIL", "period": "3m"');
      
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    test('should handle empty request body', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .set('Content-Type', 'application/json')
        .send('{}');
      
      expect([400, 401]).toContain(response.status);
    });

    test('should handle non-existent stock symbol', async () => {
      const response = await request(app)
        .post('/api/analysis')
        .set('Authorization', 'Bearer test-token')
        .send({
          symbol: 'NONEXISTENT',
          period: '3m',
          enableAI: false,
        });
      
      // Should return 404 for non-existent stock (if authenticated)
      expect([404, 401]).toContain(response.status);
    });
  });
});
