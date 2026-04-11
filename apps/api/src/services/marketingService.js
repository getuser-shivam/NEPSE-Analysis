import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class MarketingService {
  /**
   * Logs a user interaction for marketing analysis
   */
  static async logInteraction(userId, productId, action, metadata = null) {
    return await prisma.userActivity.create({
      data: {
        userId,
        productId,
        action,
        metadata: metadata ? JSON.stringify(metadata) : null,
      },
    });
  }

  /**
   * Gets a sales and interaction performance report
   */
  static async getPerformanceAnalytics() {
    const totalOrders = await prisma.order.count();
    
    // Top products by interaction
    const topInteractions = await prisma.userActivity.groupBy({
      by: ['productId', 'action'],
      _count: {
        id: true,
      },
      orderBy: {
        _count: {
          id: 'desc',
        },
      },
      take: 5,
    });

    // Recent revenue
    const revenue = await prisma.order.aggregate({
      where: {
        status: { not: 'CANCELLED' },
      },
      _sum: {
        totalAmount: true,
      },
    });

    return {
      totalRevenue: revenue._sum.totalAmount || 0,
      totalOrders,
      topInteractions,
      generatedAt: new Date(),
    };
  }
}
