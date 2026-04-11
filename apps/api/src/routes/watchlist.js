/**
 * Watchlist Routes
 */

import { Router } from 'express';
import * as watchlistController from '../controllers/watchlistController.js';
import { authenticateToken } from '../middleware/auth.js';

const router = Router();

router.use(authenticateToken);

router.get('/', watchlistController.getWatchlists);
router.post('/', watchlistController.createWatchlist);
router.delete('/:id', watchlistController.deleteWatchlist);

router.post('/:watchlistId/items', watchlistController.addToWatchlist);

export default router;
