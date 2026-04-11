import { prisma } from '../database/client.js';
import { z } from 'zod';

const watchlistSchema = z.object({
  name: z.string().min(1),
  isDefault: z.boolean().optional(),
});

export const getWatchlists = async (request, response, next) => {
  try {
    const userId = request.user.id;
    const watchlists = await prisma.watchlist.findMany({
      where: { userId },
      include: {
        items: {
          include: { stock: true },
        },
      },
    });
    response.json(watchlists);
  } catch (error) {
    next(error);
  }
};

export const createWatchlist = async (request, response, next) => {
  try {
    const userId = request.user.id;
    const { name, isDefault } = watchlistSchema.parse(request.body);

    const watchlist = await prisma.watchlist.create({
      data: { name, isDefault: isDefault ?? false, userId },
    });

    response.status(201).json(watchlist);
  } catch (error) {
    next(error);
  }
};

export const deleteWatchlist = async (request, response, next) => {
  try {
    const { id } = request.params;
    const userId = request.user.id;

    await prisma.watchlist.deleteMany({
      where: { id, userId },
    });

    response.status(204).send();
  } catch (error) {
    next(error);
  }
};

export const addToWatchlist = async (request, response, next) => {
  try {
    const { watchlistId } = request.params;
    const { stockId, notes } = request.body;
    const userId = request.user.id;

    // Verify ownership
    const watchlist = await prisma.watchlist.findFirst({
      where: { id: watchlistId, userId },
    });

    if (!watchlist) return response.status(404).json({ error: 'Watchlist not found' });

    const item = await prisma.watchlistItem.create({
      data: { watchlistId, stockId, notes },
    });

    response.status(201).json(item);
  } catch (error) {
    next(error);
  }
};
