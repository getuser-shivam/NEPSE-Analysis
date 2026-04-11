/**
 * Portfolio Routes
 */

import { Router } from 'express';
import * as portfolioController from '../controllers/portfolioController.js';
import { authenticateToken } from '../middleware/auth.js';

const router = Router();

/**
 * @swagger
 * /api/portfolio/insights:
 *   get:
 *     summary: Detailed analytics insight (allocation, total return)
 */
router.get('/insights', authenticateToken, portfolioController.getInsights);

/**
 * @swagger
 * /api/portfolio/history:
 *   get:
 *     summary: Portfolio value history for charting
 */
router.get('/history', authenticateToken, portfolioController.getHistory);

/**
 * @swagger
 * /api/portfolio/snapshot:
 *   post:
 *     summary: Manual trigger to capture current portfolio state
 */
router.post('/snapshot', authenticateToken, portfolioController.recordSnapshot);

export default router;
