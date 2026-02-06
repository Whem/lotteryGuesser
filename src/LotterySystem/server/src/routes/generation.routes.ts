import { Router, Request, Response } from 'express';
import { body } from 'express-validator';
import { asyncHandler } from '../middleware/errorHandler';
import { optionalAuth, authenticate, AuthRequest } from '../middleware/auth';
import { GenerationService } from '../services/generation.service';

const router = Router();

// Generate numbers with specific algorithms
router.post(
  '/',
  optionalAuth,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { lotteryId, algorithmIds } = req.body;

    const results = await GenerationService.generateNumbers(
      lotteryId,
      algorithmIds,
      req.user?.id
    );

    res.json({ success: true, data: results });
  })
);

// Generate with top performing algorithms
router.post(
  '/top',
  optionalAuth,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { lotteryId, count = 10 } = req.body;

    const results = await GenerationService.generateWithTopAlgorithms(
      lotteryId,
      count,
      req.user?.id
    );

    res.json({ success: true, data: results });
  })
);

// Quick generate (single algorithm, fastest response)
router.get(
  '/quick/:lotteryId',
  optionalAuth,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { lotteryId } = req.params;
    const { algorithm = 'frequency_analysis' } = req.query;

    // Get algorithm ID by name
    const { prisma } = require('../config/database');
    const algo = await prisma.algorithm.findFirst({
      where: { name: String(algorithm), isActive: true },
    });

    if (!algo) {
      return res.status(404).json({
        success: false,
        error: { code: 'ALGORITHM_NOT_FOUND', message: 'Algorithm not found' },
      });
    }

    const results = await GenerationService.generateNumbers(
      lotteryId,
      [algo.id],
      req.user?.id
    );

    res.json({ success: true, data: results[0] });
  })
);

// Evaluate predictions against actual results
router.post(
  '/evaluate',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const {
      lotteryId,
      predictedNumbers,
      predictedAdditional,
      actualNumbers,
      actualAdditional,
      algorithmId,
    } = req.body;

    const result = await GenerationService.evaluatePrediction(
      lotteryId,
      predictedNumbers,
      predictedAdditional,
      actualNumbers,
      actualAdditional,
      algorithmId
    );

    res.json({ success: true, data: result });
  })
);

export default router;
