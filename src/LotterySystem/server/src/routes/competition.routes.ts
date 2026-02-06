import { Router, Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { authenticate, AuthRequest } from '../middleware/auth';
import { prisma } from '../config/database';
import { GenerationService } from '../services/generation.service';

const router = Router();

// Get all competitions
router.get(
  '/',
  asyncHandler(async (req: Request, res: Response) => {
    const { page = 1, limit = 10, lotteryId, isActive } = req.query;
    const skip = (Number(page) - 1) * Number(limit);

    const where = {
      ...(lotteryId && { lotteryTypeId: String(lotteryId) }),
      ...(isActive !== undefined && { isActive: isActive === 'true' }),
    };

    const [competitions, total] = await Promise.all([
      prisma.competition.findMany({
        where,
        skip,
        take: Number(limit),
        orderBy: { startDate: 'desc' },
        include: {
          _count: { select: { entries: true } },
        },
      }),
      prisma.competition.count({ where }),
    ]);

    res.json({
      success: true,
      data: competitions,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        total,
        totalPages: Math.ceil(total / Number(limit)),
      },
    });
  })
);

// Get competition by ID
router.get(
  '/:id',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;

    const competition = await prisma.competition.findUnique({
      where: { id },
      include: {
        entries: {
          orderBy: { score: 'desc' },
          include: {
            // Include algorithm name
          },
        },
      },
    });

    if (!competition) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Competition not found' },
      });
    }

    res.json({ success: true, data: competition });
  })
);

// Start a new competition
router.post(
  '/',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryId } = req.body;

    const result = await GenerationService.runCompetition(lotteryId);

    res.status(201).json({ success: true, data: result });
  })
);

// Get competition results/leaderboard
router.get(
  '/:id/leaderboard',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;

    const entries = await prisma.competitionEntry.findMany({
      where: { competitionId: id },
      orderBy: [{ score: 'desc' }, { matchedMain: 'desc' }],
    });

    // Get algorithm names
    const algorithmIds = entries.map(e => e.algorithmId);
    const algorithms = await prisma.algorithm.findMany({
      where: { id: { in: algorithmIds } },
      select: { id: true, displayName: true },
    });

    const algorithmMap = new Map(algorithms.map(a => [a.id, a.displayName]));

    const leaderboard = entries.map((entry, index) => ({
      rank: index + 1,
      algorithmId: entry.algorithmId,
      algorithmName: algorithmMap.get(entry.algorithmId) || 'Unknown',
      predictedNumbers: entry.predictedNumbers,
      additionalNumbers: entry.additionalNumbers,
      matchedMain: entry.matchedMain,
      matchedAdditional: entry.matchedAdditional,
      score: entry.score,
    }));

    res.json({ success: true, data: leaderboard });
  })
);

// Finalize competition with actual results
router.post(
  '/:id/finalize',
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const { actualNumbers, actualAdditional } = req.body;

    const competition = await prisma.competition.findUnique({
      where: { id },
      include: { entries: true },
    });

    if (!competition) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Competition not found' },
      });
    }

    // Calculate scores for each entry
    const updates = competition.entries.map(entry => {
      const matchedMain = entry.predictedNumbers.filter(
        n => actualNumbers.includes(n)
      ).length;

      const matchedAdditional = entry.additionalNumbers && actualAdditional
        ? entry.additionalNumbers.filter(n => actualAdditional.includes(n)).length
        : 0;

      // Score calculation: main matches are weighted more heavily
      const score = matchedMain * 10 + matchedAdditional * 3;

      return prisma.competitionEntry.update({
        where: { id: entry.id },
        data: { matchedMain, matchedAdditional, score },
      });
    });

    await Promise.all(updates);

    // Assign ranks
    const rankedEntries = await prisma.competitionEntry.findMany({
      where: { competitionId: id },
      orderBy: [{ score: 'desc' }, { matchedMain: 'desc' }],
    });

    await Promise.all(
      rankedEntries.map((entry, index) =>
        prisma.competitionEntry.update({
          where: { id: entry.id },
          data: { rank: index + 1 },
        })
      )
    );

    // Close competition
    await prisma.competition.update({
      where: { id },
      data: { isActive: false, endDate: new Date() },
    });

    res.json({ success: true, message: 'Competition finalized' });
  })
);

export default router;
