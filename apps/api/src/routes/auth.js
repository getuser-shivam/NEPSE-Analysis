/**
 * Modernized Authentication Routes
 */

import { Router } from 'express';
import * as authController from '../controllers/authController.js';
import { authenticateToken } from '../middleware/auth.js';

const router = Router();

/**
 * @swagger
 * /api/auth/login:
 *   post:
 *     summary: User login with MFA and Lockout protection
 */
router.post('/login', authController.login);

/**
 * @swagger
 * /api/auth/refresh:
 *   post:
 *     summary: Rotate refresh token
 */
router.post('/refresh', authController.refreshToken);

/**
 * @swagger
 * /api/auth/mfa/setup:
 *   post:
 *     summary: Setup MFA (TOTP)
 */
router.post('/mfa/setup', authenticateToken, authController.setupMfa);

export default router;
