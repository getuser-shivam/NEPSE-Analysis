/**
 * Distribution & Marketing Routes
 */

import { Router } from 'express';
import { DistributionService } from '../services/distributionService.js';
import { MarketingService } from '../services/marketingService.js';
import { authenticateToken, authorizeRoles } from '../middleware/auth.js';
import { z } from 'zod';

const router = Router();

// Validation
const orderSchema = z.object({
  items: z.array(z.object({
    productId: z.string(),
    quantity: z.number().positive(),
  })),
  shippingAddress: z.string(),
});

/**
 * Place an order
 */
router.post('/orders', authenticateToken, async (request, response, next) => {
  try {
    const { items, shippingAddress } = orderSchema.parse(request.body);
    const order = await DistributionService.createOrder(request.user.id, items, shippingAddress);
    
    // Log marketing activity
    await MarketingService.logInteraction(request.user.id, null, 'ORDER_PLACED', { orderId: order.id });
    
    response.status(201).json(order);
  } catch (error) {
    next(error);
  }
});

/**
 * Get inventory forecast (Admin only)
 */
router.get('/inventory/forecast/:productId', authenticateToken, authorizeRoles('admin'), async (request, response, next) => {
  try {
    const forecast = await DistributionService.getInventoryForecast(request.params.productId);
    response.json(forecast);
  } catch (error) {
    next(error);
  }
});

/**
 * Get marketing performance analytics (Admin only)
 */
router.get('/analytics', authenticateToken, authorizeRoles('admin'), async (request, response, next) => {
  try {
    const analytics = await MarketingService.getPerformanceAnalytics();
    response.json(analytics);
  } catch (error) {
    next(error);
  }
});

/**
 * Log customer interaction
 */
router.post('/activities', authenticateToken, async (request, response, next) => {
  try {
    const { productId, action, metadata } = request.body;
    await MarketingService.logInteraction(request.user.id, productId, action, metadata);
    response.status(204).send();
  } catch (error) {
    next(error);
  }
});

export default router;
