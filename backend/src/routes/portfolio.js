import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { notesSchema, symbolSchema } from '../lib/schemas.js';
import { asyncHandler } from '../utils/asyncHandler.js';
import { buildPortfolioSummary } from '../services/portfolioSummary.js';

const router = Router();

const holdingInputSchema = z.object({
  symbol: symbolSchema,
  shares: z.coerce.number().positive(),
  buyPrice: z.coerce.number().positive(),
  currentPrice: z.coerce.number().positive().nullable().optional(),
  notes: notesSchema.optional(),
  lastUpdated: z.coerce.date().optional()
});

const holdingUpdateSchema = z
  .object({
    shares: z.coerce.number().positive().optional(),
    buyPrice: z.coerce.number().positive().optional(),
    currentPrice: z.coerce.number().positive().nullable().optional(),
    notes: notesSchema.optional(),
    lastUpdated: z.coerce.date().optional()
  })
  .refine((value) => Object.keys(value).length > 0, {
    message: 'Provide at least one field to update'
  });

router.get(
  '/',
  asyncHandler(async (_request, response) => {
    const holdings = await prisma.portfolioHolding.findMany({
      orderBy: { symbol: 'asc' }
    });

    response.json({
      data: {
        holdings,
        summary: buildPortfolioSummary(holdings)
      }
    });
  })
);

router.post(
  '/',
  asyncHandler(async (request, response) => {
    const payload = holdingInputSchema.parse(request.body ?? {});
    const existing = await prisma.portfolioHolding.findUnique({
      where: { symbol: payload.symbol },
      select: { id: true }
    });

    const holding = await prisma.portfolioHolding.upsert({
      where: { symbol: payload.symbol },
      update: {
        shares: payload.shares,
        buyPrice: payload.buyPrice,
        currentPrice: payload.currentPrice ?? null,
        notes: payload.notes ?? null,
        lastUpdated: payload.lastUpdated ?? new Date()
      },
      create: {
        symbol: payload.symbol,
        shares: payload.shares,
        buyPrice: payload.buyPrice,
        currentPrice: payload.currentPrice ?? null,
        notes: payload.notes ?? null,
        lastUpdated: payload.lastUpdated ?? new Date()
      }
    });

    response.status(existing ? 200 : 201).json({ data: holding });
  })
);

router.patch(
  '/:symbol',
  asyncHandler(async (request, response) => {
    const symbol = symbolSchema.parse(request.params.symbol);
    const payload = holdingUpdateSchema.parse(request.body ?? {});

    const holding = await prisma.portfolioHolding.update({
      where: { symbol },
      data: {
        ...payload,
        currentPrice:
          Object.prototype.hasOwnProperty.call(payload, 'currentPrice') &&
          payload.currentPrice == null
            ? null
            : payload.currentPrice,
        notes:
          Object.prototype.hasOwnProperty.call(payload, 'notes') &&
          payload.notes == null
            ? null
            : payload.notes,
        lastUpdated: payload.lastUpdated ?? new Date()
      }
    });

    response.json({ data: holding });
  })
);

router.delete(
  '/:symbol',
  asyncHandler(async (request, response) => {
    const symbol = symbolSchema.parse(request.params.symbol);
    await prisma.portfolioHolding.delete({
      where: { symbol }
    });

    response.status(204).send();
  })
);

export default router;
