import { prisma } from '../config/database';
import { AppError } from '../middleware/errorHandler';
import { PaginationParams, PaginatedResponse } from '../types';

export class LotteryService {
  static async getAllLotteries(params: PaginationParams) {
    const { page = 1, limit = 10, sortBy = 'name', sortOrder = 'asc' } = params;
    const skip = (page - 1) * limit;

    const [lotteries, total] = await Promise.all([
      prisma.lotteryType.findMany({
        where: { isActive: true },
        skip,
        take: limit,
        orderBy: { [sortBy]: sortOrder },
        select: {
          id: true,
          name: true,
          displayName: true,
          country: true,
          description: true,
          imageUrl: true,
          minNumber: true,
          maxNumber: true,
          numbersCount: true,
          hasAdditionalNumbers: true,
          additionalMinNumber: true,
          additionalMaxNumber: true,
          additionalNumbersCount: true,
          drawDays: true,
          drawTime: true,
          currency: true,
          estimatedJackpot: true,
        },
      }),
      prisma.lotteryType.count({ where: { isActive: true } }),
    ]);

    return {
      data: lotteries.map(l => ({
        ...l,
        estimatedJackpot: l.estimatedJackpot?.toString(),
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  static async getLotteryById(id: string) {
    const lottery = await prisma.lotteryType.findUnique({
      where: { id },
      include: {
        _count: {
          select: { winningNumbers: true },
        },
      },
    });

    if (!lottery) {
      throw new AppError('Lottery not found', 404, 'LOTTERY_NOT_FOUND');
    }

    return {
      ...lottery,
      estimatedJackpot: lottery.estimatedJackpot?.toString(),
      totalDraws: lottery._count.winningNumbers,
    };
  }

  static async getWinningNumbers(
    lotteryId: string,
    params: PaginationParams & { year?: number }
  ) {
    const { page = 1, limit = 20, year, sortOrder = 'desc' } = params;
    const skip = (page - 1) * limit;

    const where = {
      lotteryTypeId: lotteryId,
      ...(year && { drawYear: year }),
    };

    const [numbers, total] = await Promise.all([
      prisma.winningNumber.findMany({
        where,
        skip,
        take: limit,
        orderBy: { drawDate: sortOrder },
        select: {
          id: true,
          numbers: true,
          additionalNumbers: true,
          drawDate: true,
          drawYear: true,
          drawWeek: true,
          jackpotAmount: true,
          winnersCount: true,
        },
      }),
      prisma.winningNumber.count({ where }),
    ]);

    return {
      data: numbers.map(n => ({
        ...n,
        jackpotAmount: n.jackpotAmount?.toString(),
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  static async getHistoricalStats(lotteryId: string) {
    const winningNumbers = await prisma.winningNumber.findMany({
      where: { lotteryTypeId: lotteryId },
      select: { numbers: true, additionalNumbers: true },
    });

    if (winningNumbers.length === 0) {
      throw new AppError('No historical data available', 404, 'NO_DATA');
    }

    // Calculate frequency statistics
    const mainFrequency: Record<number, number> = {};
    const additionalFrequency: Record<number, number> = {};

    winningNumbers.forEach(draw => {
      draw.numbers.forEach(num => {
        mainFrequency[num] = (mainFrequency[num] || 0) + 1;
      });
      draw.additionalNumbers?.forEach(num => {
        additionalFrequency[num] = (additionalFrequency[num] || 0) + 1;
      });
    });

    // Sort by frequency
    const sortedMain = Object.entries(mainFrequency)
      .sort((a, b) => b[1] - a[1])
      .map(([num, freq]) => ({ number: parseInt(num), frequency: freq }));

    const sortedAdditional = Object.entries(additionalFrequency)
      .sort((a, b) => b[1] - a[1])
      .map(([num, freq]) => ({ number: parseInt(num), frequency: freq }));

    // Hot and cold numbers
    const hotNumbers = sortedMain.slice(0, 10);
    const coldNumbers = sortedMain.slice(-10).reverse();

    return {
      totalDraws: winningNumbers.length,
      mainNumberStats: {
        frequency: sortedMain,
        hotNumbers,
        coldNumbers,
      },
      additionalNumberStats: sortedAdditional.length > 0 ? {
        frequency: sortedAdditional,
        hotNumbers: sortedAdditional.slice(0, 5),
        coldNumbers: sortedAdditional.slice(-5).reverse(),
      } : null,
    };
  }

  static async addWinningNumbers(
    lotteryId: string,
    data: {
      numbers: number[];
      additionalNumbers?: number[];
      drawDate: Date;
      jackpotAmount?: bigint;
      winnersCount?: number;
    }
  ) {
    const lottery = await prisma.lotteryType.findUnique({ where: { id: lotteryId } });
    if (!lottery) {
      throw new AppError('Lottery not found', 404, 'LOTTERY_NOT_FOUND');
    }

    // Validate numbers
    if (data.numbers.length !== lottery.numbersCount) {
      throw new AppError(
        `Expected ${lottery.numbersCount} main numbers`,
        400,
        'INVALID_NUMBERS_COUNT'
      );
    }

    // Calculate year and week
    const drawDate = new Date(data.drawDate);
    const startOfYear = new Date(drawDate.getFullYear(), 0, 1);
    const days = Math.floor((drawDate.getTime() - startOfYear.getTime()) / (24 * 60 * 60 * 1000));
    const drawWeek = Math.ceil((days + startOfYear.getDay() + 1) / 7);
    const drawYear = drawDate.getFullYear();

    return prisma.winningNumber.create({
      data: {
        lotteryTypeId: lotteryId,
        numbers: data.numbers.sort((a, b) => a - b),
        additionalNumbers: data.additionalNumbers?.sort((a, b) => a - b),
        drawDate,
        drawYear,
        drawWeek,
        jackpotAmount: data.jackpotAmount,
        winnersCount: data.winnersCount,
      },
    });
  }
}
