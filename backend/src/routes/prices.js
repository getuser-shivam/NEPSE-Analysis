import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { symbolSchema } from '../lib/schemas.js';
import { asyncHandler } from '../utils/asyncHandler.js';

const router = Router();

const listQuerySchema = z.object({
  limit: z.coerce.number().int().positive().max(365).default(30)
});

const snapshotInputSchema = z
  .object({
    symbol: symbolSchema,
    tradeDate: z.coerce.date(),
    open: z.coerce.number().positive(),
    high: z.coerce.number().positive(),
    low: z.coerce.number().positive(),
    close: z.coerce.number().positive(),
    volume: z.coerce.number().nonnegative().default(0),
    source: z.string().trim().min(1).max(50).default('manual')
  })
  .superRefine((value, context) => {
    if (value.high < value.low) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'High price must be greater than or equal to low price',
        path: ['high']
      });
    }

    if (value.open > value.high || value.close > value.high) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Open and close prices cannot be above high price',
        path: ['high']
      });
    }

    if (value.open < value.low || value.close < value.low) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Open and close prices cannot be below low price',
        path: ['low']
      });
    }
  });

router.get(
  '/:symbol',
  asyncHandler(async (request, response) => {
    const symbol = symbolSchema.parse(request.params.symbol);
    const { limit } = listQuerySchema.parse(request.query);

    const snapshots = await prisma.priceSnapshot.findMany({
      where: { symbol },
      orderBy: { tradeDate: 'desc' },
      take: limit
    });

    response.json({ data: snapshots });
  })
);

router.post(
  '/',
  asyncHandler(async (request, response) => {
    const payload = snapshotInputSchema.parse(request.body ?? {});

    const snapshot = await prisma.priceSnapshot.upsert({
      where: {
        symbol_tradeDate_source: {
          symbol: payload.symbol,
          tradeDate: payload.tradeDate,
          source: payload.source
        }
      },
      update: payload,
      create: payload
    });

    response.status(201).json({ data: snapshot });
  })
);

export default router;
