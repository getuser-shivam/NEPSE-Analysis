/**
 * Swagger/OpenAPI Configuration
 * 
 * Defines the API documentation specification for the NEPSE Analysis API.
 */

export const swaggerSpec = {
  openapi: '3.0.0',
  info: {
    title: 'NEPSE Analysis API',
    version: '1.0.0',
    description: 'RESTful API for Nepal Stock Exchange (NEPSE) market analysis, technical indicators, and AI-powered investment recommendations',
    contact: {
      name: 'API Support',
      email: 'support@nepse-analysis.com'
    },
    license: {
      name: 'MIT',
      url: 'https://opensource.org/licenses/MIT'
    }
  },
  servers: [
    {
      url: 'http://localhost:4000',
      description: 'Development server'
    },
    {
      url: 'https://api.nepse-analysis.com',
      description: 'Production server'
    }
  ],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
        description: 'Enter your JWT token'
      }
    },
    schemas: {
      Stock: {
        type: 'object',
        properties: {
          id: { type: 'string', example: 'cl1234567890' },
          symbol: { type: 'string', example: 'NABIL' },
          name: { type: 'string', example: 'Nabil Bank Ltd.' },
          fullName: { type: 'string', example: 'Nabil Bank Limited' },
          sector: { type: 'string', enum: ['BANKING', 'HYDRO', 'INSURANCE', 'MANUFACTURING', 'TRADING', 'INVESTMENT', 'HOTEL', 'DEVELOPMENT_BANK', 'FINANCE', 'LIFE_INSURANCE', 'MUTUAL_FUND', 'OTHERS'] },
          listedShares: { type: 'integer', example: 100000000 },
          paidUpCapital: { type: 'number', example: 10000000000 },
          marketCap: { type: 'number', example: 50000000000 },
          isActive: { type: 'boolean', example: true },
          isListed: { type: 'boolean', example: true },
          listedDate: { type: 'string', format: 'date-time' },
          createdAt: { type: 'string', format: 'date-time' },
          updatedAt: { type: 'string', format: 'date-time' }
        },
        required: ['id', 'symbol', 'name', 'sector']
      },
      StockPrice: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          stockId: { type: 'string' },
          tradeDate: { type: 'string', format: 'date' },
          open: { type: 'number', example: 1000.0 },
          high: { type: 'number', example: 1020.0 },
          low: { type: 'number', example: 995.0 },
          close: { type: 'number', example: 1015.0 },
          volume: { type: 'integer', example: 50000 },
          turnover: { type: 'number', example: 50750000 },
          change: { type: 'number', example: 15.0 },
          changePercent: { type: 'number', example: 1.5 }
        },
        required: ['stockId', 'tradeDate', 'open', 'high', 'low', 'close', 'volume']
      },
      TechnicalIndicators: {
        type: 'object',
        properties: {
          rsi14: { type: 'number', example: 65.5, description: 'RSI 14-period' },
          rsi7: { type: 'number', example: 70.2, description: 'RSI 7-period' },
          sma20: { type: 'number', example: 1000.0 },
          sma50: { type: 'number', example: 980.0 },
          ema12: { type: 'number', example: 1010.0 },
          ema26: { type: 'number', example: 990.0 },
          macdLine: { type: 'number', example: 2.5 },
          macdSignal: { type: 'number', example: 1.8 },
          macdHistogram: { type: 'number', example: 0.7 },
          bbUpper: { type: 'number', example: 1050.0 },
          bbMiddle: { type: 'number', example: 1000.0 },
          bbLower: { type: 'number', example: 950.0 },
          stochK: { type: 'number', example: 75.0 },
          stochD: { type: 'number', example: 70.0 },
          williamsR: { type: 'number', example: -25.0 },
          atr: { type: 'number', example: 15.0 },
          trendStrength: { type: 'number', example: 0.75 },
          volatility: { type: 'number', example: 2.5 },
          signal: { type: 'string', enum: ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell'] }
        }
      },
      AnalysisRequest: {
        type: 'object',
        properties: {
          symbol: { type: 'string', example: 'NABIL' },
          period: { type: 'string', enum: ['1d', '1w', '1m', '3m', '6m', '1y'], example: '3m' },
          indicators: { 
            type: 'array', 
            items: { type: 'string', enum: ['rsi', 'macd', 'bollinger', 'stochastic', 'williamsR', 'atr'] },
            example: ['rsi', 'macd', 'bollinger']
          }
        },
        required: ['symbol']
      },
      AnalysisResponse: {
        type: 'object',
        properties: {
          symbol: { type: 'string', example: 'NABIL' },
          currentPrice: { type: 'number', example: 1015.0 },
          technicalIndicators: { $ref: '#/components/schemas/TechnicalIndicators' },
          aiAnalysis: {
            type: 'object',
            properties: {
              sentiment: { type: 'string', enum: ['bullish', 'bearish', 'neutral'] },
              confidence: { type: 'number', example: 0.75 },
              prediction: { type: 'string' },
              factors: { type: 'array', items: { type: 'string' } }
            }
          },
          pricePrediction: {
            type: 'object',
            properties: {
              predictions: { type: 'array', items: { type: 'number' } },
              trendDirection: { type: 'string' },
              confidence: { type: 'number' }
            }
          },
          generatedAt: { type: 'string', format: 'date-time' }
        }
      },
      Recommendation: {
        type: 'object',
        properties: {
          symbol: { type: 'string', example: 'NABIL' },
          action: { type: 'string', enum: ['BUY', 'SELL', 'HOLD', 'WATCH'] },
          confidence: { type: 'number', example: 0.82 },
          targetPrice: { type: 'number', example: 1100.0 },
          stopLoss: { type: 'number', example: 950.0 },
          timeHorizon: { type: 'string', example: '1-3 months' },
          rationale: { type: 'string' },
          riskLevel: { type: 'string', enum: ['LOW', 'MEDIUM', 'HIGH'] },
          technicalFactors: { type: 'array', items: { type: 'string' } },
          aiInsights: { type: 'string' },
          generatedAt: { type: 'string', format: 'date-time' }
        }
      },
      Portfolio: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          name: { type: 'string', example: 'My NEPSE Portfolio' },
          description: { type: 'string' },
          isDefault: { type: 'boolean' },
          totalInvestment: { type: 'number', example: 100000 },
          currentValue: { type: 'number', example: 110000 },
          totalGainLoss: { type: 'number', example: 10000 },
          totalGainLossPercent: { type: 'number', example: 10 },
          holdings: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                symbol: { type: 'string' },
                shares: { type: 'number' },
                buyPrice: { type: 'number' },
                currentPrice: { type: 'number' },
                gainLoss: { type: 'number' },
                gainLossPercent: { type: 'number' }
              }
            }
          }
        }
      },
      Alert: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          symbol: { type: 'string', example: 'NABIL' },
          alertType: { type: 'string', enum: ['PRICE_ABOVE', 'PRICE_BELOW', 'PERCENT_CHANGE', 'VOLUME_SPIKE', 'INDICATOR_SIGNAL'] },
          targetValue: { type: 'number', example: 1100 },
          status: { type: 'string', enum: ['ACTIVE', 'TRIGGERED', 'DISABLED', 'EXPIRED'] },
          notifyEmail: { type: 'boolean' },
          notifyPush: { type: 'boolean' },
          createdAt: { type: 'string', format: 'date-time' }
        }
      },
      Error: {
        type: 'object',
        properties: {
          error: { type: 'string' },
          details: { type: 'array', items: { type: 'object' } },
          code: { type: 'string' }
        }
      }
    }
  },
  paths: {
    '/health': {
      get: {
        summary: 'Health check',
        description: 'Returns API health status',
        tags: ['Health'],
        responses: {
          '200': {
            description: 'API is healthy',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    status: { type: 'string', example: 'healthy' },
                    timestamp: { type: 'string', format: 'date-time' },
                    version: { type: 'string', example: '1.0.0' }
                  }
                }
              }
            }
          }
        }
      }
    },
    '/api/auth/login': {
      post: {
        summary: 'User login',
        description: 'Authenticate user and receive JWT token',
        tags: ['Authentication'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  email: { type: 'string', example: 'user@example.com' },
                  password: { type: 'string', example: 'password123' }
                },
                required: ['email', 'password']
              }
            }
          }
        },
        responses: {
          '200': {
            description: 'Login successful',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    token: { type: 'string' },
                    user: { $ref: '#/components/schemas/User' }
                  }
                }
              }
            }
          },
          '401': {
            description: 'Invalid credentials'
          }
        }
      }
    },
    '/api/stocks': {
      get: {
        summary: 'List all stocks',
        description: 'Get a list of all available stocks with optional filtering',
        tags: ['Stocks'],
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: 'sector',
            in: 'query',
            description: 'Filter by sector',
            schema: { type: 'string' }
          },
          {
            name: 'search',
            in: 'query',
            description: 'Search by symbol or name',
            schema: { type: 'string' }
          },
          {
            name: 'page',
            in: 'query',
            description: 'Page number',
            schema: { type: 'integer', default: 1 }
          },
          {
            name: 'limit',
            in: 'query',
            description: 'Items per page',
            schema: { type: 'integer', default: 50 }
          }
        ],
        responses: {
          '200': {
            description: 'List of stocks',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    data: { type: 'array', items: { $ref: '#/components/schemas/Stock' } },
                    pagination: {
                      type: 'object',
                      properties: {
                        page: { type: 'integer' },
                        limit: { type: 'integer' },
                        total: { type: 'integer' },
                        totalPages: { type: 'integer' }
                      }
                    }
                  }
                }
              }
            }
          },
          '401': {
            description: 'Unauthorized'
          }
        }
      }
    },
    '/api/stocks/{symbol}': {
      get: {
        summary: 'Get stock details',
        description: 'Get detailed information about a specific stock',
        tags: ['Stocks'],
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: 'symbol',
            in: 'path',
            required: true,
            description: 'Stock symbol',
            schema: { type: 'string' }
          }
        ],
        responses: {
          '200': {
            description: 'Stock details',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/Stock' }
              }
            }
          },
          '404': {
            description: 'Stock not found'
          }
        }
      }
    },
    '/api/stocks/{symbol}/prices': {
      get: {
        summary: 'Get stock price history',
        description: 'Get historical price data for a stock',
        tags: ['Stocks'],
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: 'symbol',
            in: 'path',
            required: true,
            schema: { type: 'string' }
          },
          {
            name: 'from',
            in: 'query',
            description: 'Start date (YYYY-MM-DD)',
            schema: { type: 'string', format: 'date' }
          },
          {
            name: 'to',
            in: 'query',
            description: 'End date (YYYY-MM-DD)',
            schema: { type: 'string', format: 'date' }
          },
          {
            name: 'period',
            in: 'query',
            description: 'Data period',
            schema: { type: 'string', enum: ['1d', '1w', '1m', '3m', '6m', '1y', 'all'], default: '1y' }
          }
        ],
        responses: {
          '200': {
            description: 'Price history',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    symbol: { type: 'string' },
                    data: { type: 'array', items: { $ref: '#/components/schemas/StockPrice' } }
                  }
                }
              }
            }
          }
        }
      }
    },
    '/api/analysis': {
      post: {
        summary: 'Perform technical analysis',
        description: 'Calculate technical indicators and get AI-powered analysis',
        tags: ['Analysis'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/AnalysisRequest' }
            }
          }
        },
        responses: {
          '200': {
            description: 'Analysis results',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/AnalysisResponse' }
              }
            }
          },
          '400': {
            description: 'Invalid request',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/Error' }
              }
            }
          }
        }
      }
    },
    '/api/recommendations': {
      get: {
        summary: 'Get investment recommendations',
        description: 'Get personalized investment recommendations based on technical analysis and AI',
        tags: ['Recommendations'],
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: 'riskProfile',
            in: 'query',
            description: 'Risk profile filter',
            schema: { type: 'string', enum: ['conservative', 'moderate', 'aggressive'] }
          },
          {
            name: 'sector',
            in: 'query',
            description: 'Sector filter',
            schema: { type: 'string' }
          },
          {
            name: 'limit',
            in: 'query',
            description: 'Number of recommendations',
            schema: { type: 'integer', default: 10 }
          }
        ],
        responses: {
          '200': {
            description: 'List of recommendations',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    recommendations: { type: 'array', items: { $ref: '#/components/schemas/Recommendation' } },
                    generatedAt: { type: 'string', format: 'date-time' }
                  }
                }
              }
            }
          }
        }
      }
    },
    '/api/portfolio': {
      get: {
        summary: 'Get user portfolio',
        description: 'Get user portfolio details and holdings',
        tags: ['Portfolio'],
        security: [{ bearerAuth: [] }],
        responses: {
          '200': {
            description: 'Portfolio details',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/Portfolio' }
              }
            }
          }
        }
      },
      post: {
        summary: 'Create portfolio',
        description: 'Create a new portfolio',
        tags: ['Portfolio'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  description: { type: 'string' }
                },
                required: ['name']
              }
            }
          }
        },
        responses: {
          '201': {
            description: 'Portfolio created'
          }
        }
      }
    },
    '/api/watchlist': {
      get: {
        summary: 'Get watchlist',
        description: 'Get user watchlist items',
        tags: ['Watchlist'],
        security: [{ bearerAuth: [] }],
        responses: {
          '200': {
            description: 'Watchlist items',
            content: {
              'application/json': {
                schema: {
                  type: 'array',
                  items: { $ref: '#/components/schemas/Stock' }
                }
              }
            }
          }
        }
      },
      post: {
        summary: 'Add to watchlist',
        description: 'Add a stock to watchlist',
        tags: ['Watchlist'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  symbol: { type: 'string' }
                },
                required: ['symbol']
              }
            }
          }
        },
        responses: {
          '201': {
            description: 'Added to watchlist'
          }
        }
      }
    },
    '/api/alerts': {
      get: {
        summary: 'Get price alerts',
        description: 'Get user price alerts',
        tags: ['Alerts'],
        security: [{ bearerAuth: [] }],
        responses: {
          '200': {
            description: 'List of alerts',
            content: {
              'application/json': {
                schema: {
                  type: 'array',
                  items: { $ref: '#/components/schemas/Alert' }
                }
              }
            }
          }
        }
      },
      post: {
        summary: 'Create alert',
        description: 'Create a new price alert',
        tags: ['Alerts'],
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/Alert' }
            }
          }
        },
        responses: {
          '201': {
            description: 'Alert created'
          }
        }
      }
    }
  }
};

export default swaggerSpec;
