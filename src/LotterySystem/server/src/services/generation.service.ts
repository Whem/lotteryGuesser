import { prisma } from '../config/database';
import { AppError } from '../middleware/errorHandler';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw, AlgorithmResult } from '../types';
import { algorithmRegistry } from '../algorithms';
import { AlgorithmService } from './algorithm.service';

export class GenerationService {
  static async generateNumbers(
    lotteryId: string,
    algorithmIds: string[],
    userId?: string
  ): Promise<AlgorithmResult[]> {
    // Get lottery configuration
    const lottery = await prisma.lotteryType.findUnique({
      where: { id: lotteryId },
    });

    if (!lottery) {
      throw new AppError('Lottery not found', 404, 'LOTTERY_NOT_FOUND');
    }

    const config: LotteryConfig = {
      minNumber: lottery.minNumber,
      maxNumber: lottery.maxNumber,
      numbersCount: lottery.numbersCount,
      hasAdditionalNumbers: lottery.hasAdditionalNumbers,
      additionalMinNumber: lottery.additionalMinNumber || undefined,
      additionalMaxNumber: lottery.additionalMaxNumber || undefined,
      additionalNumbersCount: lottery.additionalNumbersCount || undefined,
    };

    // Get historical data
    const history = await this.getHistoricalData(lotteryId);

    // Get algorithms
    const algorithms = await prisma.algorithm.findMany({
      where: {
        id: { in: algorithmIds },
        isActive: true,
      },
    });

    if (algorithms.length === 0) {
      throw new AppError('No valid algorithms found', 400, 'NO_ALGORITHMS');
    }

    // Check premium access
    if (userId) {
      const user = await prisma.user.findUnique({
        where: { id: userId },
        select: { isPremium: true },
      });

      const premiumAlgorithms = algorithms.filter(a => a.isPremium);
      if (premiumAlgorithms.length > 0 && !user?.isPremium) {
        throw new AppError(
          `Premium algorithms selected: ${premiumAlgorithms.map(a => a.displayName).join(', ')}`,
          403,
          'PREMIUM_REQUIRED'
        );
      }
    }

    // Generate numbers from each algorithm
    const results: AlgorithmResult[] = [];

    for (const algo of algorithms) {
      const implementation = algorithmRegistry.get(algo.name);
      
      if (!implementation) {
        continue; // Skip if implementation not found
      }

      const startTime = performance.now();
      
      try {
        const numbers = await implementation.predict(config, history);
        const executionTime = performance.now() - startTime;

        results.push({
          algorithmId: algo.id,
          algorithmName: algo.displayName,
          numbers: {
            ...numbers,
            executionTime,
          },
        });

        // Save to history if user is logged in
        if (userId) {
          await prisma.generationHistory.create({
            data: {
              userId,
              lotteryTypeId: lotteryId,
              algorithmId: algo.id,
              generatedNumbers: numbers.main,
              additionalNumbers: numbers.additional,
              executionTime,
              confidence: numbers.confidence,
            },
          });
        }
      } catch (error) {
        // Log error but continue with other algorithms
        console.error(`Algorithm ${algo.name} failed:`, error);
      }
    }

    if (results.length === 0) {
      throw new AppError('All algorithms failed to generate numbers', 500, 'GENERATION_FAILED');
    }

    return results;
  }

  static async generateWithTopAlgorithms(
    lotteryId: string,
    count: number = 10,
    userId?: string
  ): Promise<AlgorithmResult[]> {
    // Get top performing algorithms for this lottery
    const topAlgorithms = await AlgorithmService.getTopAlgorithms(lotteryId, count);

    // If not enough ranked algorithms, fill with random ones
    let algorithmIds = topAlgorithms.map(a => a.algorithm.id);

    if (algorithmIds.length < count) {
      const additionalAlgorithms = await prisma.algorithm.findMany({
        where: {
          isActive: true,
          id: { notIn: algorithmIds },
        },
        take: count - algorithmIds.length,
        orderBy: { globalScore: 'desc' },
      });
      algorithmIds = [...algorithmIds, ...additionalAlgorithms.map(a => a.id)];
    }

    return this.generateNumbers(lotteryId, algorithmIds, userId);
  }

  static async getHistoricalData(lotteryId: string, limit: number = 500): Promise<HistoricalDraw[]> {
    const winningNumbers = await prisma.winningNumber.findMany({
      where: { lotteryTypeId: lotteryId },
      orderBy: { drawDate: 'desc' },
      take: limit,
      select: {
        numbers: true,
        additionalNumbers: true,
        drawDate: true,
        drawYear: true,
        drawWeek: true,
      },
    });

    return winningNumbers.map(wn => ({
      numbers: wn.numbers,
      additionalNumbers: wn.additionalNumbers || undefined,
      drawDate: wn.drawDate,
      year: wn.drawYear,
      week: wn.drawWeek,
    }));
  }

  static async evaluatePrediction(
    lotteryId: string,
    predictedNumbers: number[],
    predictedAdditional: number[] | undefined,
    actualNumbers: number[],
    actualAdditional: number[] | undefined,
    algorithmId: string
  ) {
    // Count matches
    const mainMatches = predictedNumbers.filter(n => actualNumbers.includes(n)).length;
    const additionalMatches = predictedAdditional && actualAdditional
      ? predictedAdditional.filter(n => actualAdditional.includes(n)).length
      : 0;

    // Update algorithm performance
    await AlgorithmService.updatePerformance(
      algorithmId,
      lotteryId,
      mainMatches,
      additionalMatches,
      0 // Execution time not available for historical evaluation
    );

    return {
      mainMatches,
      additionalMatches,
      totalMatches: mainMatches + additionalMatches,
    };
  }

  static async runCompetition(lotteryId: string) {
    // Get all active algorithms
    const algorithms = await prisma.algorithm.findMany({
      where: { isActive: true },
    });

    // Get lottery config and history
    const lottery = await prisma.lotteryType.findUnique({
      where: { id: lotteryId },
    });

    if (!lottery) {
      throw new AppError('Lottery not found', 404, 'LOTTERY_NOT_FOUND');
    }

    const config: LotteryConfig = {
      minNumber: lottery.minNumber,
      maxNumber: lottery.maxNumber,
      numbersCount: lottery.numbersCount,
      hasAdditionalNumbers: lottery.hasAdditionalNumbers,
      additionalMinNumber: lottery.additionalMinNumber || undefined,
      additionalMaxNumber: lottery.additionalMaxNumber || undefined,
      additionalNumbersCount: lottery.additionalNumbersCount || undefined,
    };

    const history = await this.getHistoricalData(lotteryId);

    // Create competition
    const competition = await prisma.competition.create({
      data: {
        name: `Competition ${new Date().toISOString()}`,
        lotteryTypeId: lotteryId,
        startDate: new Date(),
        isActive: true,
      },
    });

    // Generate predictions from all algorithms
    const entries: { algorithmId: string; numbers: number[]; additional?: number[] }[] = [];

    for (const algo of algorithms) {
      const implementation = algorithmRegistry.get(algo.name);
      if (!implementation) continue;

      try {
        const result = await implementation.predict(config, history);
        entries.push({
          algorithmId: algo.id,
          numbers: result.main,
          additional: result.additional,
        });
      } catch (error) {
        console.error(`Competition: Algorithm ${algo.name} failed:`, error);
      }
    }

    // Save competition entries
    await prisma.competitionEntry.createMany({
      data: entries.map(e => ({
        competitionId: competition.id,
        algorithmId: e.algorithmId,
        predictedNumbers: e.numbers,
        additionalNumbers: e.additional,
      })),
    });

    return {
      competitionId: competition.id,
      entriesCount: entries.length,
    };
  }
}
