import { Router, Response } from 'express';
import { body } from 'express-validator';
import { asyncHandler } from '../middleware/errorHandler';
import { authenticate, AuthRequest } from '../middleware/auth';
import { prisma } from '../config/database';

const router = Router();

// Get user profile
router.get(
  '/profile',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const user = await prisma.user.findUnique({
      where: { id: req.user!.id },
      select: {
        id: true,
        email: true,
        displayName: true,
        avatarUrl: true,
        isPremium: true,
        premiumExpiresAt: true,
        createdAt: true,
        lastLoginAt: true,
        settings: true,
        _count: {
          select: {
            savedTickets: true,
            generationHistory: true,
            favoriteAlgorithms: true,
          },
        },
      },
    });

    res.json({ success: true, data: user });
  })
);

// Update profile
router.patch(
  '/profile',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { displayName, avatarUrl } = req.body;

    const user = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        ...(displayName && { displayName }),
        ...(avatarUrl && { avatarUrl }),
      },
      select: {
        id: true,
        email: true,
        displayName: true,
        avatarUrl: true,
      },
    });

    res.json({ success: true, data: user });
  })
);

// Get user settings
router.get(
  '/settings',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    let settings = await prisma.userSettings.findUnique({
      where: { userId: req.user!.id },
    });

    // Create default settings if not exist
    if (!settings) {
      settings = await prisma.userSettings.create({
        data: { userId: req.user!.id },
      });
    }

    res.json({ success: true, data: settings });
  })
);

// Update settings
router.patch(
  '/settings',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const {
      defaultLotteryId,
      notificationsEnabled,
      darkMode,
      language,
      autoSaveTickets,
      showAlgorithmDetails,
    } = req.body;

    const settings = await prisma.userSettings.upsert({
      where: { userId: req.user!.id },
      create: {
        userId: req.user!.id,
        defaultLotteryId,
        notificationsEnabled,
        darkMode,
        language,
        autoSaveTickets,
        showAlgorithmDetails,
      },
      update: {
        ...(defaultLotteryId !== undefined && { defaultLotteryId }),
        ...(notificationsEnabled !== undefined && { notificationsEnabled }),
        ...(darkMode !== undefined && { darkMode }),
        ...(language !== undefined && { language }),
        ...(autoSaveTickets !== undefined && { autoSaveTickets }),
        ...(showAlgorithmDetails !== undefined && { showAlgorithmDetails }),
      },
    });

    res.json({ success: true, data: settings });
  })
);

// Get favorite algorithms
router.get(
  '/favorites',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const favorites = await prisma.userFavoriteAlgorithm.findMany({
      where: { userId: req.user!.id },
      include: {
        algorithm: {
          select: {
            id: true,
            name: true,
            displayName: true,
            category: true,
            complexity: true,
            globalScore: true,
          },
        },
      },
      orderBy: { priority: 'asc' },
    });

    res.json({ success: true, data: favorites });
  })
);

// Add favorite algorithm
router.post(
  '/favorites/:algorithmId',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { algorithmId } = req.params;

    const favorite = await prisma.userFavoriteAlgorithm.upsert({
      where: {
        userId_algorithmId: {
          userId: req.user!.id,
          algorithmId,
        },
      },
      create: {
        userId: req.user!.id,
        algorithmId,
      },
      update: {},
      include: { algorithm: true },
    });

    res.json({ success: true, data: favorite });
  })
);

// Remove favorite algorithm
router.delete(
  '/favorites/:algorithmId',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { algorithmId } = req.params;

    await prisma.userFavoriteAlgorithm.deleteMany({
      where: {
        userId: req.user!.id,
        algorithmId,
      },
    });

    res.json({ success: true, message: 'Favorite removed' });
  })
);

// Get generation history
router.get(
  '/history',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { page = 1, limit = 20 } = req.query;
    const skip = (Number(page) - 1) * Number(limit);

    const [history, total] = await Promise.all([
      prisma.generationHistory.findMany({
        where: { userId: req.user!.id },
        skip,
        take: Number(limit),
        orderBy: { createdAt: 'desc' },
        include: {
          lotteryType: {
            select: { name: true, displayName: true },
          },
          algorithm: {
            select: { name: true, displayName: true },
          },
        },
      }),
      prisma.generationHistory.count({ where: { userId: req.user!.id } }),
    ]);

    res.json({
      success: true,
      data: history,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        total,
        totalPages: Math.ceil(total / Number(limit)),
      },
    });
  })
);

export default router;
