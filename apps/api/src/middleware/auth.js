/**
 * Authentication Middleware
 * 
 * JWT-based authentication middleware for protecting API routes.
 */

import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

/**
 * Middleware to authenticate JWT tokens
 * 
 * Verifies the JWT token from the Authorization header
 * and attaches the decoded user information to the request object.
 * 
 * @param {Request} request - Express request object
 * @param {Response} response - Express response object
 * @param {Function} next - Express next middleware function
 */
export function authenticateToken(request, response, next) {
  const authHeader = request.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return response.status(401).json({
      error: 'Access token required',
      message: 'Please provide a valid authentication token',
    });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Attach user information to request
    request.user = {
      id: decoded.userId,
      email: decoded.email,
    };
    
    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return response.status(401).json({
        error: 'Token expired',
        message: 'Authentication token has expired',
      });
    }
    
    if (error.name === 'JsonWebTokenError') {
      return response.status(401).json({
        error: 'Invalid token',
        message: 'Authentication token is invalid',
      });
    }
    
    return response.status(401).json({
      error: 'Authentication failed',
      message: 'Could not authenticate user',
    });
  }
}

/**
 * Optional authentication middleware
 * 
 * Attaches user information if token is provided, but doesn't require it.
 * Useful for routes that work both with and without authentication.
 * 
 * @param {Request} request - Express request object
 * @param {Response} response - Express response object
 * @param {Function} next - Express next middleware function
 */
export function optionalAuth(request, response, next) {
  const authHeader = request.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (token) {
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      request.user = {
        id: decoded.userId,
        email: decoded.email,
      };
    } catch (error) {
      // Ignore errors for optional auth
      // User will not be attached if token is invalid
    }
  }

  next();
}

/**
 * Role-based authorization middleware
 * 
 * Checks if the authenticated user has the required role.
 * 
 * @param {string[]} allowedRoles - Array of allowed roles
 * @returns {Function} Express middleware function
 */
export function authorizeRoles(...allowedRoles) {
  return (request, response, next) => {
    if (!request.user) {
      return response.status(401).json({
        error: 'Authentication required',
        message: 'You must be logged in to access this resource',
      });
    }

    // For now, we'll implement a simple role check
    // In a real application, you would fetch user roles from the database
    const userRole = request.user.role || 'user';

    if (!allowedRoles.includes(userRole)) {
      return response.status(403).json({
        error: 'Access denied',
        message: 'You do not have permission to access this resource',
      });
    }

    next();
  };
}
