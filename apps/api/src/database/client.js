import '../config.js';
import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis;

/**
 * Standardized Prisma Client
 * Singleton instance across the application.
 */
export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: ['warn', 'error']
  });

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}
