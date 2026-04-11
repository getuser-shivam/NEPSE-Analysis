import { z } from 'zod';

/**
 * Standardized Error Handling Middleware
 * Ensures all API errors return a consistent JSON structure.
 */
export const errorHandler = (err, req, res, next) => {
  console.error(`[Error] ${req.method} ${req.url}:`, err.message);

  // Zod Validation Errors
  if (err instanceof z.ZodError) {
    return res.status(400).json({
      error: 'Validation Error',
      message: 'Invalid request data',
      details: err.errors.map(e => ({ path: e.path.join('.'), message: e.message })),
      code: 'VALIDATION_ERROR'
    });
  }

  // Known Business Errors
  if (err.status) {
    return res.status(err.status).json({
      error: err.name || 'Error',
      message: err.message,
      code: err.code || 'BUSINESS_ERROR'
    });
  }

  // Prisma Errors
  if (err.code && err.code.startsWith('P')) {
    return res.status(400).json({
      error: 'Database Error',
      message: 'A persistence error occurred',
      code: `DB_${err.code}`
    });
  }

  // Default Fallback
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'production' 
      ? 'An unexpected error occurred' 
      : err.message,
    code: 'INTERNAL_ERROR'
  });
};
