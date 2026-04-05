import { Router } from 'express';
import { z } from 'zod';
import { asyncHandler } from '../utils/asyncHandler.js';
import { loadDashboardSnapshot } from '../services/dashboardService.js';

const router = Router();

const dashboardQuerySchema = z.object({
  limit: z.coerce.number().int().positive().max(60).default(5)
});

router.get(
  '/',
  asyncHandler(async (request, response) => {
    const { limit } = dashboardQuerySchema.parse(request.query);
    const snapshot = await loadDashboardSnapshot({ limit });

    response.json({ data: snapshot });
  })
);

export default router;
