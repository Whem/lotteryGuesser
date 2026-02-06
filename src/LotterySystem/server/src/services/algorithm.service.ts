import { prisma } from '../config/database';
import { AppError } from '../middleware/errorHandler';
import { PaginationParams } from '../types';
import { algorithmRegistry } from '../algorithms';

export class AlgorithmService {
  static async getAllAlgorithms(params: PaginationParams & { category?: string; isPremium?: boolean }) {
    const { page = 1, limit = 20, sortBy = 'globalScore', sortOrder = 'desc', category, isPremium } = params;
    const skip = (page - 1) * limit;

    const where = {
      isActive: true,
      ...(category && { category: category as any }),
      ...(isPremium !== undefined && { isPremium }),
    };

    const [algorithms, total] = await Promise.all([
      prisma.algorithm.findMany({
        where,
        skip,
        take: limit,
        orderBy: { [sortBy]: sortOrder },
        select: {
          id: true,
          name: true,
          displayName: true,
          description: true,
          category: true,
          complexity: true,
          isPremium: true,
          globalScore: true,
          totalPredictions: true,
          avgExecutionTime: true,
        },
      }),
      prisma.algorithm.count({ where }),
    ]);

    return {
      data: algorithms,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  static async getAlgorithmById(id: string) {
    const algorithm = await prisma.algorithm.findUnique({
      where: { id },
      include: {
        performances: {
          include: {
            lotteryType: {
              select: { name: true, displayName: true },
            },
          },
        },
      },
    });

    if (!algorithm) {
      throw new AppError('Algorithm not found', 404, 'ALGORITHM_NOT_FOUND');
    }

    return algorithm;
  }

  static async getAlgorithmPerformance(algorithmId: string, lotteryId: string) {
    const performance = await prisma.algorithmPerformance.findUnique({
      where: {
        algorithmId_lotteryTypeId: {
          algorithmId,
          lotteryTypeId: lotteryId,
        },
      },
    });

    if (!performance) {
      return {
        score: 0,
        totalPredictions: 0,
        matchRates: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
        avgExecutionTime: 0,
      };
    }

    return {
      score: performance.score,
      totalPredictions: performance.totalPredictions,
      matchRates: {
        1: performance.matchRate1,
        2: performance.matchRate2,
        3: performance.matchRate3,
        4: performance.matchRate4,
        5: performance.matchRate5,
      },
      avgExecutionTime: performance.avgExecutionTime,
      perfectMatches: performance.perfectMatches,
    };
  }

  static async getTopAlgorithms(lotteryId: string, limit: number = 10) {
    const performances = await prisma.algorithmPerformance.findMany({
      where: {
        lotteryTypeId: lotteryId,
        totalPredictions: { gte: 5 }, // Minimum 5 predictions to be ranked
      },
      orderBy: { score: 'desc' },
      take: limit,
      include: {
        algorithm: {
          select: {
            id: true,
            name: true,
            displayName: true,
            category: true,
            complexity: true,
            isPremium: true,
          },
        },
      },
    });

    return performances.map((p, index) => ({
      rank: index + 1,
      algorithm: p.algorithm,
      score: p.score,
      totalPredictions: p.totalPredictions,
      matchRates: {
        1: p.matchRate1,
        2: p.matchRate2,
        3: p.matchRate3,
        4: p.matchRate4,
        5: p.matchRate5,
      },
    }));
  }

  static async updatePerformance(
    algorithmId: string,
    lotteryId: string,
    matchCount: number,
    additionalMatchCount: number,
    executionTime: number
  ) {
    // Calculate score (weighted by match count)
    const scoreWeights: Record<number, number> = {
      0: 0,
      1: 1,
      2: 3,
      3: 10,
      4: 50,
      5: 500,
    };
    const score = (scoreWeights[matchCount] || 0) + (additionalMatchCount * 2);

    // Upsert performance record
    const existing = await prisma.algorithmPerformance.findUnique({
      where: {
        algorithmId_lotteryTypeId: { algorithmId, lotteryTypeId: lotteryId },
      },
    });

    if (existing) {
      const newTotal = existing.totalPredictions + 1;
      const newAvgTime = (existing.avgExecutionTime * existing.totalPredictions + executionTime) / newTotal;
      const newScore = (existing.score * existing.totalPredictions + score) / newTotal;

      // Update match rates
      const matchRateKey = `matchRate${Math.min(matchCount, 5)}` as keyof typeof existing;
      const currentRate = existing[matchRateKey] as number || 0;
      const newMatchRate = ((currentRate * existing.totalPredictions) + (matchCount > 0 ? 1 : 0)) / newTotal;

      await prisma.algorithmPerformance.update({
        where: { id: existing.id },
        data: {
          score: newScore,
          totalPredictions: newTotal,
          avgExecutionTime: newAvgTime,
          [`matchRate${Math.min(matchCount, 5)}`]: newMatchRate,
          perfectMatches: matchCount >= 5 ? existing.perfectMatches + 1 : existing.perfectMatches,
          lastPrediction: new Date(),
        },
      });
    } else {
      await prisma.algorithmPerformance.create({
        data: {
          algorithmId,
          lotteryTypeId: lotteryId,
          score,
          totalPredictions: 1,
          avgExecutionTime: executionTime,
          [`matchRate${Math.min(matchCount, 5)}`]: 1,
          perfectMatches: matchCount >= 5 ? 1 : 0,
          lastPrediction: new Date(),
        },
      });
    }

    // Update global algorithm score
    const allPerformances = await prisma.algorithmPerformance.findMany({
      where: { algorithmId },
    });

    const globalScore = allPerformances.reduce((sum, p) => sum + p.score, 0) / allPerformances.length;
    const totalPredictions = allPerformances.reduce((sum, p) => sum + p.totalPredictions, 0);
    const avgExecutionTime = allPerformances.reduce((sum, p) => sum + p.avgExecutionTime, 0) / allPerformances.length;

    await prisma.algorithm.update({
      where: { id: algorithmId },
      data: {
        globalScore,
        totalPredictions,
        avgExecutionTime,
      },
    });
  }

  static async syncAlgorithmsFromRegistry() {
    const algorithms = algorithmRegistry.getAll();
    
    for (const algo of algorithms) {
      await prisma.algorithm.upsert({
        where: { name: algo.name },
        create: {
          name: algo.name,
          displayName: algo.displayName,
          description: algo.description,
          category: algo.category,
          complexity: algo.complexity,
          isPremium: algo.isPremium,
        },
        update: {
          displayName: algo.displayName,
          description: algo.description,
          category: algo.category,
          complexity: algo.complexity,
          isPremium: algo.isPremium,
        },
      });
    }

    return { synced: algorithms.length };
  }
}
