import express from 'express';
import cors from 'cors';
import { ZodError } from 'zod';
import { env } from './config.js';
import healthRouter from './routes/health.js';
import settingsRouter from './routes/settings.js';
import portfolioRouter from './routes/portfolio.js';
import watchlistRouter from './routes/watchlist.js';
import pricesRouter from './routes/prices.js';
import dashboardRouter from './routes/dashboard.js';

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

  app.get('/', (_request, response) => {
    response.json({
      service: 'NEPSE Analysis API',
      version: '0.1.0',
      endpoints: ['/health', '/api/dashboard', '/api/settings', '/api/portfolio', '/api/watchlist', '/api/prices/:symbol']
    });
  });

  app.use('/health', healthRouter);
  app.use('/api/dashboard', dashboardRouter);
  app.use('/api/settings', settingsRouter);
  app.use('/api/portfolio', portfolioRouter);
  app.use('/api/watchlist', watchlistRouter);
  app.use('/api/prices', pricesRouter);

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
