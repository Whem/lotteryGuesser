import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class GapAnalysis extends BaseAlgorithm {
  name = 'gap_analysis';
  displayName = 'Gap Analysis';
  description = 'Selects numbers that are "due" based on how long since they last appeared';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate gaps (how many draws since each number appeared)
    const gaps = this.calculateGaps(history, config.maxNumber);
    
    // Calculate average gap for each number
    const avgGaps = this.calculateAverageGaps(history, config.maxNumber);
    
    // Score numbers: higher score for numbers that exceeded their average gap
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const currentGap = gaps.get(num) || 0;
      const avgGap = avgGaps.get(num) || history.length;
      
      // Score is how much the current gap exceeds the average
      const score = avgGap > 0 ? currentGap / avgGap : currentGap;
      scores.set(num, score);
    }
    
    // Select numbers with highest "due" score
    const mainNumbers = this.selectTopN(scores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalGaps = this.calculateAdditionalGaps(history, config.additionalMaxNumber!);
      additional = this.selectTopN(additionalGaps, config.additionalNumbersCount);
      
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount
        );
      }
    }

    return {
      main: finalMain,
      additional,
      confidence: 0.6,
      executionTime: performance.now() - startTime,
    };
  }

  private calculateAverageGaps(history: HistoricalDraw[], maxNumber: number): Map<number, number> {
    const gaps = new Map<number, number[]>();
    const lastSeen = new Map<number, number>();
    
    // Initialize
    for (let i = 1; i <= maxNumber; i++) {
      gaps.set(i, []);
    }
    
    // Track gaps between appearances
    history.forEach((draw, index) => {
      draw.numbers.forEach(num => {
        if (lastSeen.has(num)) {
          const gap = index - lastSeen.get(num)!;
          gaps.get(num)!.push(gap);
        }
        lastSeen.set(num, index);
      });
    });
    
    // Calculate averages
    const avgGaps = new Map<number, number>();
    gaps.forEach((gapList, num) => {
      if (gapList.length > 0) {
        avgGaps.set(num, gapList.reduce((a, b) => a + b, 0) / gapList.length);
      } else {
        avgGaps.set(num, history.length);
      }
    });
    
    return avgGaps;
  }

  private calculateAdditionalGaps(history: HistoricalDraw[], maxNumber: number): Map<number, number> {
    const gaps = new Map<number, number>();
    
    for (let i = 1; i <= maxNumber; i++) {
      gaps.set(i, history.length);
    }
    
    history.forEach((draw, index) => {
      draw.additionalNumbers?.forEach(num => {
        if (!gaps.has(num) || gaps.get(num) === history.length) {
          gaps.set(num, index);
        }
      });
    });
    
    return gaps;
  }
}
