import 'dotenv/config';

function parsePort(value) {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 4000;
}

export const env = Object.freeze({
  port: parsePort(process.env.PORT),
  corsOrigin: process.env.CORS_ORIGIN?.trim() || '*',
  databaseUrl: process.env.DATABASE_URL?.trim() || 'file:./dev.db'
});
