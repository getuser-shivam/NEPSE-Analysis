import { z } from 'zod';
import bcrypt from 'bcryptjs';
import { prisma } from '../database/client.js';
import { AuthService } from '../services/authService.js';
import { MailService } from '../services/mailService.js';
import crypto from 'crypto';

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  mfaCode: z.string().length(6).optional(),
});

export const login = async (request, response, next) => {
  try {
    const { email, password, mfaCode } = loginSchema.parse(request.body);

    const user = await prisma.user.findUnique({
      where: { email: email.toLowerCase() },
    });

    if (!user) {
      return response.status(401).json({ error: 'Invalid credentials' });
    }

    if (AuthService.isAccountLocked(user)) {
      return response.status(403).json({
        error: 'Account locked',
        message: 'Too many failed attempts. Try again later.',
        lockedUntil: user.lockedUntil,
      });
    }

    const isValidPassword = await bcrypt.compare(password, user.passwordHash);
    if (!isValidPassword) {
      const { isLocked } = await AuthService.registerFailedAttempt(user.id, user.failedLoginAttempts);
      return response.status(401).json({
        error: 'Invalid credentials',
        message: isLocked ? 'Account has been locked.' : 'Incorrect password',
      });
    }

    if (user.mfaEnabled) {
      if (!mfaCode) {
        return response.status(200).json({ mfaRequired: true, userId: user.id });
      }
      
      const isMfaValid = AuthService.verifyMfaToken(mfaCode, user.mfaSecret);
      if (!isMfaValid) {
        return response.status(401).json({ error: 'Invalid MFA code' });
      }
    }

    await AuthService.resetFailedAttempts(user.id);
    const { accessToken, refreshToken } = await AuthService.generateTokenPair(user);

    response.json({
      accessToken,
      refreshToken,
      user: { id: user.id, email: user.email, name: user.name, mfaEnabled: user.mfaEnabled },
    });
  } catch (error) {
    next(error);
  }
};

export const refreshToken = async (request, response, next) => {
  try {
    const { refreshToken } = request.body;
    if (!refreshToken) return response.status(400).json({ error: 'Refresh token required' });

    const tokens = await AuthService.rotateRefreshToken(refreshToken);
    response.json(tokens);
  } catch (error) {
    response.status(401).json({ error: 'Invalid refresh token' });
  }
};

export const setupMfa = async (request, response, next) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: request.user.id } });
    const { secret, otpauth } = AuthService.generateMfaSecret(user.email);
    
    await prisma.user.update({
      where: { id: user.id },
      data: { mfaSecret: secret },
    });

    response.json({ secret, otpauth });
  } catch (error) {
    next(error);
  }
};
