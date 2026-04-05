import { createServer } from 'node:http';
import { env } from './config.js';
import { createApp } from './app.js';
import { prisma } from './lib/prisma.js';

const app = createApp();
const server = createServer(app);

async function shutdown(signal) {
  console.log(`Received ${signal}. Closing server...`);

  server.close(async () => {
    await prisma.$disconnect();
    process.exit(0);
  });

  setTimeout(async () => {
    await prisma.$disconnect();
    process.exit(1);
  }, 5000).unref();
}

server.listen(env.port, () => {
  console.log(`NEPSE Analysis API listening on http://localhost:${env.port}`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    shutdown(signal).catch((error) => {
      console.error('Failed to shut down cleanly', error);
      process.exit(1);
    });
  });
}
