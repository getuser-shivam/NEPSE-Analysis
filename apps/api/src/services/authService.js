import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { authenticator } from 'otplib';
import { PrismaClient } from '@prisma/client';
import { env } from '../config.js';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
const REFRESH_TOKEN_SECRET = process.env.REFRESH_TOKEN_SECRET || 'your-refresh-token-secret-change-in-production';

const ACCESS_TOKEN_EXPIRES_IN = '15m'; // Short-lived access token
const REFRESH_TOKEN_EXPIRES_IN = '7d'; // Long-lived refresh token

export class AuthService {
  /**
   * Generates a pair of Access and Refresh tokens
   */
  static async generateTokenPair(user) {
    const accessToken = jwt.sign(
      { userId: user.id, email: user.email, role: user.role || 'user' },
      JWT_SECRET,
      { expiresIn: ACCESS_TOKEN_EXPIRES_IN }
    );

    const refreshTokenString = jwt.sign(
      { userId: user.id },
      REFRESH_TOKEN_SECRET,
      { expiresIn: REFRESH_TOKEN_EXPIRES_IN }
    );

    // Store refresh token in database (hashed for security)
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7);

    await prisma.refreshToken.create({
      data: {
        token: refreshTokenString,
        userId: user.id,
        expiresAt: expiresAt,
      },
    });

    return { accessToken, refreshToken: refreshTokenString };
  }

  /**
   * Verifies and rotates a refresh token
   */
  static async rotateRefreshToken(oldTokenString) {
    try {
      const decoded = jwt.verify(oldTokenString, REFRESH_TOKEN_SECRET);
      
      const storedToken = await prisma.refreshToken.findUnique({
        where: { token: oldTokenString },
        include: { user: true },
      });

      if (!storedToken || storedToken.revokedAt || storedToken.expiresAt < new Date()) {
        throw new Error('Invalid or expired refresh token');
      }

      // Revoke old token
      await prisma.refreshToken.update({
        where: { id: storedToken.id },
        data: { revokedAt: new Date() },
      });

      // Generate new pair
      return await this.generateTokenPair(storedToken.user);
    } catch (error) {
      throw new Error('Refresh token rotation failed');
    }
  }

  /**
   * MFA: Setup TOTP for a user
   */
  static generateMfaSecret(email) {
    const secret = authenticator.generateSecret();
    const otpauth = authenticator.keyuri(email, 'NEPSE Analysis', secret);
    return { secret, otpauth };
  }

  /**
   * MFA: Verify TOTP token
   */
  static verifyMfaToken(token, secret) {
    return authenticator.verify({ token, secret });
  }

  /**
   * Account Protection: Check if account is locked
   */
  static isAccountLocked(user) {
    if (!user.lockedUntil) return false;
    return user.lockedUntil > new Date();
  }

  /**
   * Account Protection: Reset failed attempts
   */
  static async resetFailedAttempts(userId) {
    await prisma.user.update({
      where: { id: userId },
      data: { failedLoginAttempts: 0, lockedUntil: null },
    });
  }

  /**
   * Account Protection: Register failed attempt
   */
  static async registerFailedAttempt(userId, currentAttempts) {
    const newAttempts = currentAttempts + 1;
    const data = { failedLoginAttempts: newAttempts };

    if (newAttempts >= 5) {
      const lockoutTime = new Date();
      lockoutTime.setMinutes(lockoutTime.getMinutes() + 15); // 15 minute lock
      data.lockedUntil = lockoutTime;
    }

    await prisma.user.update({
      where: { id: userId },
      data,
    });

    return { attempts: newAttempts, isLocked: newAttempts >= 5 };
  }
}
