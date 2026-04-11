/**
 * Alerts Routes
 * 
 * API endpoints for price alert management.
 */

import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../database/client.js';
import { authenticateToken } from '../middleware/auth.js';

const router = Router();

// Validation schemas
const createAlertSchema = z.object({
  stockId: z.string(),
  alertType: z.enum(['PRICE_ABOVE', 'PRICE_BELOW', 'PERCENT_CHANGE', 'VOLUME_SPIKE', 'INDICATOR_SIGNAL']),
  targetValue: z.number(),
  referencePrice: z.number().optional(),
  notifyEmail: z.boolean().default(true),
  notifyPush: z.boolean().default(false),
  expiresAt: z.string().optional(),
});

const updateAlertSchema = z.object({
  status: z.enum(['ACTIVE', 'TRIGGERED', 'DISABLED', 'EXPIRED']).optional(),
  targetValue: z.number().optional(),
  notifyEmail: z.boolean().optional(),
  notifyPush: z.boolean().optional(),
});

/**
 * @swagger
 * /api/alerts:
 *   get:
 *     summary: Get user's alerts
 */
router.get('/', authenticateToken, async (request, response, next) => {
  try {
    const { status } = request.query;
    const userId = request.user.id;

    const where = { userId };
    if (status) {
      where.status = status;
    }

    const alerts = await prisma.priceAlert.findMany({
      where,
      include: {
        stock: {
          select: {
            symbol: true,
            name: true,
          },
        },
      },
      orderBy: { createdAt: 'desc' },
    });

    response.json(alerts);
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/alerts:
 *   post:
 *     summary: Create new alert
 */
router.post('/', authenticateToken, async (request, response, next) => {
  try {
    const userId = request.user.id;
    const data = createAlertSchema.parse(request.body);

    const alert = await prisma.priceAlert.create({
      data: {
        ...data,
        userId,
        status: 'ACTIVE',
        expiresAt: data.expiresAt ? new Date(data.expiresAt) : null,
      },
      include: {
        stock: {
          select: {
            symbol: true,
            name: true,
          },
        },
      },
    });

    response.status(201).json(alert);
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/alerts/{id}:
 *   get:
 *     summary: Get specific alert
 */
router.get('/:id', authenticateToken, async (request, response, next) => {
  try {
    const { id } = request.params;
    const userId = request.user.id;

    const alert = await prisma.priceAlert.findFirst({
      where: { id, userId },
      include: {
        stock: {
          select: { symbol: true, name: true },
        },
      },
    });

    if (!alert) return response.status(404).json({ error: 'Alert not found' });
    response.json(alert);
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/alerts/{id}:
 *   put:
 *     summary: Update alert
 */
router.put('/:id', authenticateToken, async (request, response, next) => {
  try {
    const { id } = request.params;
    const userId = request.user.id;
    const data = updateAlertSchema.parse(request.body);

    const alert = await prisma.priceAlert.updateMany({
      where: { id, userId },
      data,
    });

    if (alert.count === 0) return response.status(404).json({ error: 'Alert not found' });

    const updatedAlert = await prisma.priceAlert.findFirst({
      where: { id },
      include: {
        stock: { select: { symbol: true, name: true } },
      },
    });

    response.json(updatedAlert);
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/alerts/{id}:
 *   delete:
 *     summary: Delete alert
 */
router.delete('/:id', authenticateToken, async (request, response, next) => {
  try {
    const { id } = request.params;
    const userId = request.user.id;

    const alert = await prisma.priceAlert.deleteMany({
      where: { id, userId },
    });

    if (alert.count === 0) return response.status(404).json({ error: 'Alert not found' });
    response.status(204).send();
  } catch (error) {
    next(error);
  }
});

export default router;
