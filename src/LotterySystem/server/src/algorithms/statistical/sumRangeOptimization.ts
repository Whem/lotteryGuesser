import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class SumRangeOptimization extends BaseAlgorithm {
  name = 'sum_range_optimization';
  displayName = 'Sum Range Optimizer';
  description = 'Generates numbers whose sum falls within the most common historical range';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate sum distribution
    const sums = history.map(draw => draw.numbers.reduce((a, b) => a + b, 0));
    const avgSum = sums.reduce((a, b) => a + b, 0) / sums.length;
    const stdDev = Math.sqrt(
      sums.reduce((acc, sum) => acc + Math.pow(sum - avgSum, 2), 0) / sums.length
    );
    
    // Target sum range (within 1 standard deviation)
    const targetMin = avgSum - stdDev;
    const targetMax = avgSum + stdDev;
    
    // Generate numbers that fall within target sum range
    const mainNumbers = this.generateWithTargetSum(
      config.minNumber,
      config.maxNumber,
      config.numbersCount,
      targetMin,
      targetMax
    );

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalSums = history
        .filter(d => d.additionalNumbers)
        .map(d => d.additionalNumbers!.reduce((a, b) => a + b, 0));
      
      if (additionalSums.length > 0) {
        const avgAdditionalSum = additionalSums.reduce((a, b) => a + b, 0) / additionalSums.length;
        const additionalStdDev = Math.sqrt(
          additionalSums.reduce((acc, sum) => acc + Math.pow(sum - avgAdditionalSum, 2), 0) / additionalSums.length
        );
        
        additional = this.generateWithTargetSum(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount,
          avgAdditionalSum - additionalStdDev,
          avgAdditionalSum + additionalStdDev
        );
      } else {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount
        );
      }
    }

    return {
      main: mainNumbers,
      additional,
      confidence: 0.72,
      executionTime: performance.now() - startTime,
    };
  }

  private generateWithTargetSum(
    min: number,
    max: number,
    count: number,
    targetMin: number,
    targetMax: number,
    maxAttempts: number = 1000
  ): number[] {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const numbers = this.generateRandomNumbers(min, max, count);
      const sum = numbers.reduce((a, b) => a + b, 0);
      
      if (sum >= targetMin && sum <= targetMax) {
        return numbers;
      }
    }
    
    // Fallback: try to adjust numbers to get closer to target
    let numbers = this.generateRandomNumbers(min, max, count);
    const targetSum = (targetMin + targetMax) / 2;
    
    for (let i = 0; i < 100; i++) {
      const currentSum = numbers.reduce((a, b) => a + b, 0);
      if (currentSum >= targetMin && currentSum <= targetMax) break;
      
      // Find a number to swap
      const indexToChange = Math.floor(Math.random() * count);
      const currentNum = numbers[indexToChange];
      
      if (currentSum < targetMin) {
        // Need higher sum, try a larger number
        const candidates = [];
        for (let n = currentNum + 1; n <= max; n++) {
          if (!numbers.includes(n)) candidates.push(n);
        }
        if (candidates.length > 0) {
          numbers[indexToChange] = candidates[Math.floor(Math.random() * candidates.length)];
        }
      } else {
        // Need lower sum, try a smaller number
        const candidates = [];
        for (let n = min; n < currentNum; n++) {
          if (!numbers.includes(n)) candidates.push(n);
        }
        if (candidates.length > 0) {
          numbers[indexToChange] = candidates[Math.floor(Math.random() * candidates.length)];
        }
      }
    }
    
    return numbers.sort((a, b) => a - b);
  }
}
