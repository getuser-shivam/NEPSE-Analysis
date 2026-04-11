import express from 'express';
import cors from 'cors';
import swaggerUi from 'swagger-ui-express';
import { ZodError } from 'zod';
import { env } from './config.js';
import { swaggerSpec } from './config/swagger.js';
import { errorHandler } from './middleware/errorHandler.js';
import { authenticateToken } from './middleware/auth.js';
import healthRouter from './routes/health.js';
import settingsRouter from './routes/settings.js';
import portfolioRouter from './routes/portfolio.js';
import watchlistRouter from './routes/watchlist.js';
import pricesRouter from './routes/prices.js';
import dashboardRouter from './routes/dashboard.js';
import stocksRouter from './routes/stocks.js';
import analysisRouter from './routes/analysis.js';
import recommendationsRouter from './routes/recommendations.js';
import alertsRouter from './routes/alerts.js';
import authRouter from './routes/auth.js';
import productsRouter from './routes/products.js';
import distributionRouter from './routes/distribution.js';
import { authRateLimiter, apiRateLimiter } from './middleware/rateLimit.js';

function buildCorsOptions() {
  const origins = env.corsOrigin
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  if (origins.length === 0 || origins.includes('*')) {
    return { origin: true };
  }

  return { origin: origins };
}

export function createApp() {
  const app = express();

  app.use(cors(buildCorsOptions()));
  app.use(express.json());

  // API Documentation
  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
    explorer: true,
    customCss: '.swagger-ui .topbar { display: none }',
    customSiteTitle: 'NEPSE Analysis API Documentation'
  }));

  // API Metadata
  app.get('/api', (req, res) => {
    res.json({ name: 'NEPSE Analysis API', version: '2.0.0', status: 'Healthy' });
  });

  app.get('/', (_request, response) => {
    response.json({
      service: 'NEPSE Analysis API',
      version: '1.0.0',
      documentation: '/api-docs',
      endpoints: {
        public: ['/health', '/api/auth/login', '/api-docs'],
        protected: [
          '/api/stocks',
          '/api/stocks/:symbol',
          '/api/analysis',
          '/api/recommendations',
          '/api/portfolio',
          '/api/watchlist',
          '/api/alerts',
          '/api/dashboard',
          '/api/settings',
          '/api/products',
          '/api/dist'
        ]
      }
    });
  });

  // Public routes (no authentication required)
  app.use('/health', healthRouter);
  app.use('/api/auth', authRateLimiter, authRouter);

  // Protected routes (authentication required)
  app.use('/api/stocks', authenticateToken, stocksRouter);
  app.use('/api/analysis', authenticateToken, analysisRouter);
  app.use('/api/recommendations', authenticateToken, recommendationsRouter);
  app.use('/api/portfolio', authenticateToken, portfolioRouter);
  app.use('/api/watchlist', authenticateToken, watchlistRouter);
  app.use('/api/alerts', authenticateToken, alertsRouter);
  app.use('/api/prices', authenticateToken, pricesRouter);
  app.use('/api/dashboard', authenticateToken, dashboardRouter);
  app.use('/api/settings', authenticateToken, settingsRouter);
  app.use('/api/products', authenticateToken, productsRouter);
  app.use('/api/dist', distributionRouter);

  app.use((request, response) => {
    response.status(404).json({
      error: `Route not found: ${request.method} ${request.originalUrl}`
    });
  });

  app.use((error, _request, response, _next) => {
    if (error instanceof ZodError) {
      return response.status(400).json({
        error: 'Validation failed',
        details: error.issues.map((issue) => ({
          path: issue.path.join('.'),
          message: issue.message
        }))
      });
    }

    if (error?.code === 'P2025') {
      return response.status(404).json({
        error: 'Record not found'
      });
    }

    console.error(error);
    return response.status(500).json({
      error: 'Internal server error'
    });
  });

  return app;
}
