import { z } from 'zod';

export const symbolSchema = z
  .string()
  .trim()
  .min(1, 'Symbol is required')
  .max(10, 'Symbol must be at most 10 characters')
  .transform((value) => value.toUpperCase())
  .refine((value) => /^[A-Z0-9]+$/.test(value), {
    message: 'Symbol must contain only letters and numbers'
  });

export const notesSchema = z
  .union([z.string().trim().max(500), z.null(), z.undefined()])
  .transform((value) => {
    if (value == null || value === '') {
      return null;
    }

    return value;
  });
