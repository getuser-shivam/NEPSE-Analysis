import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { ensureDefaultSettings } from '../lib/appSettings.js';
import { asyncHandler } from '../utils/asyncHandler.js';

const router = Router();

const settingsUpdateSchema = z
  .object({
    autoSaveInterval: z.coerce.number().int().positive().max(86400).optional(),
    maxDataAgeDays: z.coerce.number().int().positive().max(3650).optional(),
    backupEnabled: z.coerce.boolean().optional(),
    chartStyle: z.string().trim().min(1).max(100).optional(),
    defaultPeriod: z.string().trim().min(1).max(20).optional(),
    maxWatchlistSize: z.coerce.number().int().positive().max(1000).optional(),
    logLevel: z.string().trim().min(1).max(20).optional(),
    refreshInterval: z.coerce.number().int().positive().max(86400).optional()
  })
  .refine((value) => Object.keys(value).length > 0, {
    message: 'Provide at least one setting to update'
  });

router.get(
  '/',
  asyncHandler(async (_request, response) => {
    const settings = await ensureDefaultSettings();
    response.json({ data: settings });
  })
);

router.patch(
  '/',
  asyncHandler(async (request, response) => {
    const payload = settingsUpdateSchema.parse(request.body ?? {});
    const settings = await ensureDefaultSettings();

    const updatedSettings = await prisma.appSetting.update({
      where: { id: settings.id },
      data: payload
    });

    response.json({ data: updatedSettings });
  })
);

export default router;
