import { prisma } from './prisma.js';

const defaultSettings = {
  name: 'default',
  autoSaveInterval: 300,
  maxDataAgeDays: 7,
  backupEnabled: true,
  chartStyle: 'seaborn-v0_8',
  defaultPeriod: '1y',
  maxWatchlistSize: 50,
  logLevel: 'INFO',
  refreshInterval: 300
};

export function ensureDefaultSettings() {
  return prisma.appSetting.upsert({
    where: { name: defaultSettings.name },
    update: {},
    create: defaultSettings
  });
}
