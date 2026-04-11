/**
 * Jest Test Setup
 * 
 * Global test configuration and utilities
 */

import { jest } from '@jest/globals';
import { PrismaClient } from '@prisma/client';

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || 'sqlserver://localhost:1433;database=NEPSEAnalysisDB_Test;user=sa;password=test_password;trustServerCertificate=true';
process.env.JWT_SECRET = 'test-secret-key';
process.env.GROQ_API_KEY = 'test-groq-key';
process.env.POLLENS_API_KEY = 'test-pollens-key';

// Mock Prisma for unit tests
const mockPrisma = {
  stock: {
    findMany: jest.fn(),
    findUnique: jest.fn(),
    create: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
  },
  stockPrice: {
    findMany: jest.fn(),
    create: jest.fn(),
  },
  user: {
    findUnique: jest.fn(),
    create: jest.fn(),
  },
  portfolio: {
    findMany: jest.fn(),
    findUnique: jest.fn(),
    create: jest.fn(),
  },
  $connect: jest.fn(),
  $disconnect: jest.fn(),
};

// Global test utilities
global.testUtils = {
  mockPrisma,
  
  // Create mock stock data
  createMockStock: (overrides = {}) => ({
    id: 'test-stock-id',
    symbol: 'NABIL',
    name: 'Nabil Bank Ltd.',
    sector: 'BANKING',
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  }),
  
  // Create mock price data
  createMockPrices: (count = 30) => {
    const prices = [];
    let basePrice = 1000;
    
    for (let i = 0; i < count; i++) {
      const change = (Math.random() - 0.5) * 20;
      basePrice += change;
      
      prices.push({
        id: `price-${i}`,
        stockId: 'test-stock-id',
        tradeDate: new Date(Date.now() - (count - i) * 24 * 60 * 60 * 1000),
        open: basePrice - 5,
        high: basePrice + 10,
        low: basePrice - 10,
        close: basePrice,
        volume: Math.floor(Math.random() * 100000),
      });
    }
    
    return prices;
  },
  
  // Create mock user
  createMockUser: (overrides = {}) => ({
    id: 'test-user-id',
    email: 'test@example.com',
    name: 'Test User',
    riskProfile: 'moderate',
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  }),
  
  // Create mock JWT token
  createMockToken: (userId = 'test-user-id') => {
    return `mock-jwt-token-${userId}`;
  },
};

// Setup before all tests
beforeAll(async () => {
  // Connect to test database if needed
  // await prisma.$connect();
});

// Cleanup after all tests
afterAll(async () => {
  // Disconnect from test database
  // await prisma.$disconnect();
});

// Reset mocks before each test
beforeEach(() => {
  jest.clearAllMocks();
});
