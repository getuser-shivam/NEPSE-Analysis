import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  await prisma.appSetting.upsert({
    where: { name: 'default' },
    update: {},
    create: {
      name: 'default',
      autoSaveInterval: 300,
      maxDataAgeDays: 7,
      backupEnabled: true,
      chartStyle: 'seaborn-v0_8',
      defaultPeriod: '1y',
      maxWatchlistSize: 50,
      logLevel: 'INFO',
      refreshInterval: 300
    }
  });
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (error) => {
    console.error(error);
    await prisma.$disconnect();
    process.exit(1);
  });
