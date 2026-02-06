import { Router, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { authenticate, AuthRequest } from '../middleware/auth';
import { prisma } from '../config/database';

const router = Router();

// Get all saved tickets
router.get(
  '/',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { page = 1, limit = 20, lotteryId, isPlayed } = req.query;
    const skip = (Number(page) - 1) * Number(limit);

    const where = {
      userId: req.user!.id,
      ...(lotteryId && { lotteryTypeId: String(lotteryId) }),
      ...(isPlayed !== undefined && { isPlayed: isPlayed === 'true' }),
    };

    const [tickets, total] = await Promise.all([
      prisma.savedTicket.findMany({
        where,
        skip,
        take: Number(limit),
        orderBy: { createdAt: 'desc' },
        include: {
          lotteryType: {
            select: { name: true, displayName: true, imageUrl: true },
          },
        },
      }),
      prisma.savedTicket.count({ where }),
    ]);

    res.json({
      success: true,
      data: tickets.map(t => ({
        ...t,
        prizeWon: t.prizeWon?.toString(),
      })),
      pagination: {
        page: Number(page),
        limit: Number(limit),
        total,
        totalPages: Math.ceil(total / Number(limit)),
      },
    });
  })
);

// Get ticket by ID
router.get(
  '/:id',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { id } = req.params;

    const ticket = await prisma.savedTicket.findFirst({
      where: { id, userId: req.user!.id },
      include: {
        lotteryType: true,
      },
    });

    if (!ticket) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Ticket not found' },
      });
    }

    res.json({
      success: true,
      data: {
        ...ticket,
        prizeWon: ticket.prizeWon?.toString(),
      },
    });
  })
);

// Save a new ticket
router.post(
  '/',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const {
      lotteryId,
      numbers,
      additionalNumbers,
      algorithmIds,
      nickname,
      targetDrawDate,
    } = req.body;

    const ticket = await prisma.savedTicket.create({
      data: {
        userId: req.user!.id,
        lotteryTypeId: lotteryId,
        numbers,
        additionalNumbers,
        algorithmIds: algorithmIds || [],
        nickname,
        targetDrawDate: targetDrawDate ? new Date(targetDrawDate) : null,
      },
      include: {
        lotteryType: {
          select: { name: true, displayName: true },
        },
      },
    });

    res.status(201).json({ success: true, data: ticket });
  })
);

// Update ticket
router.patch(
  '/:id',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { id } = req.params;
    const { nickname, isPlayed, targetDrawDate } = req.body;

    // Verify ownership
    const existing = await prisma.savedTicket.findFirst({
      where: { id, userId: req.user!.id },
    });

    if (!existing) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Ticket not found' },
      });
    }

    const ticket = await prisma.savedTicket.update({
      where: { id },
      data: {
        ...(nickname !== undefined && { nickname }),
        ...(isPlayed !== undefined && { isPlayed }),
        ...(targetDrawDate !== undefined && {
          targetDrawDate: targetDrawDate ? new Date(targetDrawDate) : null,
        }),
      },
    });

    res.json({ success: true, data: ticket });
  })
);

// Delete ticket
router.delete(
  '/:id',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { id } = req.params;

    // Verify ownership
    const existing = await prisma.savedTicket.findFirst({
      where: { id, userId: req.user!.id },
    });

    if (!existing) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Ticket not found' },
      });
    }

    await prisma.savedTicket.delete({ where: { id } });

    res.json({ success: true, message: 'Ticket deleted' });
  })
);

// Check ticket against actual results
router.post(
  '/:id/check',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { id } = req.params;
    const { actualNumbers, actualAdditional, prizeWon } = req.body;

    // Verify ownership
    const ticket = await prisma.savedTicket.findFirst({
      where: { id, userId: req.user!.id },
    });

    if (!ticket) {
      return res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: 'Ticket not found' },
      });
    }

    // Calculate matches
    const matchedNumbers = ticket.numbers.filter(n => actualNumbers.includes(n)).length;
    const matchedAdditional = ticket.additionalNumbers && actualAdditional
      ? ticket.additionalNumbers.filter(n => actualAdditional.includes(n)).length
      : 0;

    // Update ticket with results
    const updatedTicket = await prisma.savedTicket.update({
      where: { id },
      data: {
        isPlayed: true,
        matchedNumbers,
        matchedAdditional,
        prizeWon: prizeWon ? BigInt(prizeWon) : null,
      },
    });

    res.json({
      success: true,
      data: {
        ...updatedTicket,
        prizeWon: updatedTicket.prizeWon?.toString(),
        matchedNumbers,
        matchedAdditional,
      },
    });
  })
);

export default router;
