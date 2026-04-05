import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { asyncHandler } from '../utils/asyncHandler.js';

const router = Router();

router.get(
  '/',
  asyncHandler(async (_request, response) => {
    await prisma.appSetting.count();

    response.json({
      status: 'ok',
      database: 'connected',
      timestamp: new Date().toISOString()
    });
  })
);

export default router;
