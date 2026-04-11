import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class AlertService {
  /**
   * Evaluates all active alerts for all users
   */
  static async evaluateAlerts() {
    const activeAlerts = await prisma.priceAlert.findMany({
      where: { status: 'ACTIVE' },
      include: { stock: true },
    });

    const results = {
      evaluated: activeAlerts.length,
      triggered: 0,
    };

    for (const alert of activeAlerts) {
      const isTriggered = this.checkCondition(alert, alert.stock.lastPrice);
      
      if (isTriggered) {
        await this.triggerAlert(alert);
        results.triggered++;
      }
    }

    return results;
  }

  /**
   * Condition logic for different alert types
   */
  static checkCondition(alert, currentPrice) {
    if (!currentPrice) return false;

    switch (alert.alertType) {
      case 'PRICE_ABOVE':
        return currentPrice >= alert.targetValue;
      case 'PRICE_BELOW':
        return currentPrice <= alert.targetValue;
      case 'PERCENT_CHANGE':
        if (!alert.referencePrice) return false;
        const change = ((currentPrice - alert.referencePrice) / alert.referencePrice) * 100;
        return Math.abs(change) >= alert.targetValue;
      default:
        return false;
    }
  }

  /**
   * Action to take when an alert is triggered
   */
  static async triggerAlert(alert) {
    await prisma.priceAlert.update({
      where: { id: alert.id },
      data: {
        status: 'TRIGGERED',
        triggeredAt: new Date(),
      },
    });

    // In a real system, send Push notification or Email here
    console.log(`[ALERT] Alert ${alert.id} triggered for ${alert.stock.symbol} at ${alert.stock.lastPrice}`);
  }
}
