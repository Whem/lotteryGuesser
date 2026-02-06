import { Router, Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { optionalAuth, authenticate, AuthRequest } from '../middleware/auth';
import { AlgorithmService } from '../services/algorithm.service';

const router = Router();

// Get all algorithms
router.get(
  '/',
  asyncHandler(async (req: Request, res: Response) => {
    const { 
      page = 1, 
      limit = 20, 
      sortBy = 'globalScore', 
      sortOrder = 'desc',
      category,
      isPremium,
    } = req.query;

    const result = await AlgorithmService.getAllAlgorithms({
      page: Number(page),
      limit: Number(limit),
      sortBy: String(sortBy),
      sortOrder: sortOrder as 'asc' | 'desc',
      category: category as string | undefined,
      isPremium: isPremium !== undefined ? isPremium === 'true' : undefined,
    });

    res.json({ success: true, ...result });
  })
);

// Get algorithm by ID
router.get(
  '/:id',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const algorithm = await AlgorithmService.getAlgorithmById(id);
    res.json({ success: true, data: algorithm });
  })
);

// Get algorithm performance for a specific lottery
router.get(
  '/:id/performance/:lotteryId',
  asyncHandler(async (req: Request, res: Response) => {
    const { id, lotteryId } = req.params;
    const performance = await AlgorithmService.getAlgorithmPerformance(id, lotteryId);
    res.json({ success: true, data: performance });
  })
);

// Get top algorithms for a lottery
router.get(
  '/top/:lotteryId',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryId } = req.params;
    const { limit = 10 } = req.query;
    const topAlgorithms = await AlgorithmService.getTopAlgorithms(lotteryId, Number(limit));
    res.json({ success: true, data: topAlgorithms });
  })
);

// Sync algorithms from registry (admin endpoint)
router.post(
  '/sync',
  asyncHandler(async (req: Request, res: Response) => {
    const result = await AlgorithmService.syncAlgorithmsFromRegistry();
    res.json({ success: true, data: result });
  })
);

export default router;
