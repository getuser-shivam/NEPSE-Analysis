import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class DistributionService {
  /**
   * Places an order and handles stock reservation
   */
  static async createOrder(userId, items, shippingAddress) {
    return await prisma.$transaction(async (tx) => {
      let totalAmount = 0;
      const orderItemsData = [];

      // 1. Validate stock and calculate total
      for (const item of items) {
        const product = await tx.product.findUnique({
          where: { id: item.productId },
        });

        if (!product || product.stock < item.quantity) {
          throw new Error(`Insufficient stock for product: ${product?.name || item.productId}`);
        }

        totalAmount += product.price * item.quantity;
        orderItemsData.push({
          productId: item.productId,
          quantity: item.quantity,
          price: product.price,
        });

        // 2. Reduce stock
        await tx.product.update({
          where: { id: item.productId },
          data: { stock: { decrement: item.quantity } },
        });

        // 3. Log transaction
        await tx.inventoryTransaction.create({
          data: {
            productId: item.productId,
            delta: -item.quantity,
            reason: 'SALE',
          },
        });
      }

      // 4. Create order
      const order = await tx.order.create({
        data: {
          userId,
          totalAmount,
          shippingAddress,
          items: {
            create: orderItemsData,
          },
          status: 'PENDING',
        },
        include: { items: true },
      });

      return order;
    });
  }

  /**
   * Updates shipment status and estimated arrival
   */
  static async updateShipmentStatus(orderId, status, trackingNumber = null) {
    return await prisma.shipment.upsert({
      where: { orderId },
      update: { 
        status, 
        trackingNumber: trackingNumber || undefined,
        shippedAt: status === 'IN_TRANSIT' ? new Date() : undefined,
        deliveredAt: status === 'DELIVERED' ? new Date() : undefined,
      },
      create: {
        orderId,
        status,
        trackingNumber,
      },
    });
  }

  /**
   * Simple inventory forecasting (SMA based)
   * Predicts days of stock remaining
   */
  static async getInventoryForecast(productId) {
    // Get sales transactions for last 30 days
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const transactions = await prisma.inventoryTransaction.findMany({
      where: {
        productId,
        reason: 'SALE',
        createdAt: { gte: thirtyDaysAgo },
      },
    });

    const totalSold = transactions.reduce((acc, t) => acc + Math.abs(t.delta), 0);
    const avgDailySales = totalSold / 30;

    const product = await prisma.product.findUnique({ where: { id: productId } });
    const daysRemaining = avgDailySales > 0 ? (product.stock / avgDailySales) : Infinity;

    return {
      currentStock: product.stock,
      avgDailySales: avgDailySales.toFixed(2),
      daysRemaining: daysRemaining === Infinity ? 'N/A' : daysRemaining.toFixed(0),
      belowThreshold: daysRemaining < 7, // Alert if less than 7 days left
    };
  }
}
