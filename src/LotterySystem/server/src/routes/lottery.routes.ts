import { Router, Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { optionalAuth, AuthRequest } from '../middleware/auth';
import { LotteryService } from '../services/lottery.service';

const router = Router();

// Get all lotteries
router.get(
  '/',
  asyncHandler(async (req: Request, res: Response) => {
    const { page = 1, limit = 10, sortBy = 'name', sortOrder = 'asc' } = req.query;
    
    const result = await LotteryService.getAllLotteries({
      page: Number(page),
      limit: Number(limit),
      sortBy: String(sortBy),
      sortOrder: sortOrder as 'asc' | 'desc',
    });

    res.json({ success: true, ...result });
  })
);

// Get lottery by ID
router.get(
  '/:id',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const lottery = await LotteryService.getLotteryById(id);
    res.json({ success: true, data: lottery });
  })
);

// Get winning numbers for a lottery
router.get(
  '/:id/winning-numbers',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const { page = 1, limit = 20, year, sortOrder = 'desc' } = req.query;

    const result = await LotteryService.getWinningNumbers(id, {
      page: Number(page),
      limit: Number(limit),
      year: year ? Number(year) : undefined,
      sortOrder: sortOrder as 'asc' | 'desc',
    });

    res.json({ success: true, ...result });
  })
);

// Get historical statistics
router.get(
  '/:id/stats',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const stats = await LotteryService.getHistoricalStats(id);
    res.json({ success: true, data: stats });
  })
);

// Add winning numbers (admin only - for now just protected)
router.post(
  '/:id/winning-numbers',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const { numbers, additionalNumbers, drawDate, jackpotAmount, winnersCount } = req.body;

    const result = await LotteryService.addWinningNumbers(id, {
      numbers,
      additionalNumbers,
      drawDate: new Date(drawDate),
      jackpotAmount: jackpotAmount ? BigInt(jackpotAmount) : undefined,
      winnersCount,
    });

    res.status(201).json({ success: true, data: result });
  })
);

export default router;
