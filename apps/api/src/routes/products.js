/**
 * Products Routes
 * 
 * API endpoints for the Product Catalog.
 */

import { Router } from 'express';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';

const router = Router();
const prisma = new PrismaClient();

// Validation schemas
const querySchema = z.object({
  category: z.string().optional(),
  search: z.string().optional(),
  priceMin: z.string().transform(Number).optional(),
  priceMax: z.string().transform(Number).optional(),
  page: z.string().transform(Number).default('1'),
  limit: z.string().transform(Number).default('20'),
  sortBy: z.enum(['price', 'name', 'createdAt']).default('createdAt'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
});

/**
 * @swagger
 * /api/products:
 *   get:
 *     summary: List all products
 *     description: Get a paginated list of products with filtering and sorting
 *     tags: [Products]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: category
 *         schema:
 *           type: string
 *       - in: query
 *         name: search
 *         schema:
 *           type: string
 *       - in: query
 *         name: priceMin
 *         schema:
 *           type: number
 *       - in: query
 *         name: priceMax
 *         schema:
 *           type: number
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *     responses:
 *       200:
 *         description: List of products retrieved successfully
 */
router.get('/', async (request, response, next) => {
  try {
    const { category, search, priceMin, priceMax, page, limit, sortBy, sortOrder } = querySchema.parse(request.query);
    
    const where = { isActive: true };
    
    if (category) {
      where.category = category;
    }
    
    if (search) {
      where.OR = [
        { name: { contains: search } },
        { description: { contains: search } },
      ];
    }
    
    if (priceMin !== undefined || priceMax !== undefined) {
      where.price = {};
      if (priceMin !== undefined) where.price.gte = priceMin;
      if (priceMax !== undefined) where.price.lte = priceMax;
    }

    const skip = (page - 1) * limit;
    
    const [products, total] = await Promise.all([
      prisma.product.findMany({
        where,
        skip,
        take: limit,
        orderBy: { [sortBy]: sortOrder },
      }),
      prisma.product.count({ where }),
    ]);

    response.json({
      data: products,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @swagger
 * /api/products/{id}:
 *   get:
 *     summary: Get product details
 *     description: Get detailed information about a specific product
 *     tags: [Products]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Product details retrieved successfully
 *       404:
 *         description: Product not found
 */
router.get('/:id', async (request, response, next) => {
  try {
    const { id } = request.params;
    
    const product = await prisma.product.findUnique({
      where: { id },
    });

    if (!product) {
      return response.status(404).json({
        error: 'Product not found',
      });
    }

    response.json(product);
  } catch (error) {
    next(error);
  }
});

export default router;
