import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { ensureDefaultSettings } from '../lib/appSettings.js';
import { notesSchema, symbolSchema } from '../lib/schemas.js';
import { asyncHandler } from '../utils/asyncHandler.js';

const router = Router();

router.get(
  '/',
  asyncHandler(async (_request, response) => {
    const items = await prisma.watchlistItem.findMany({
      orderBy: { symbol: 'asc' }
    });

    response.json({ data: items });
  })
);

router.post(
  '/',
  asyncHandler(async (request, response) => {
    const payload = {
      symbol: symbolSchema.parse(request.body?.symbol),
      notes: notesSchema.parse(request.body?.notes)
    };

    const existing = await prisma.watchlistItem.findUnique({
      where: { symbol: payload.symbol },
      select: { id: true }
    });

    if (!existing) {
      const settings = await ensureDefaultSettings();
      const count = await prisma.watchlistItem.count();

      if (count >= settings.maxWatchlistSize) {
        return response.status(409).json({
          error: 'Watchlist limit reached',
          maxWatchlistSize: settings.maxWatchlistSize
        });
      }
    }

    const item = await prisma.watchlistItem.upsert({
      where: { symbol: payload.symbol },
      update: { notes: payload.notes },
      create: payload
    });

    response.status(existing ? 200 : 201).json({ data: item });
  })
);

router.delete(
  '/:symbol',
  asyncHandler(async (request, response) => {
    const symbol = symbolSchema.parse(request.params.symbol);
    await prisma.watchlistItem.delete({
      where: { symbol }
    });

    response.status(204).send();
  })
);

export default router;
