import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class CyclicPattern extends BaseAlgorithm {
  name = 'cyclic_pattern';
  displayName = 'Cyclic Pattern Detection';
  description = 'Detects repeating cycles in number appearances';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze cycles for each number
    const cycleScores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const score = this.analyzeNumberCycle(num, history);
      cycleScores.set(num, score);
    }
    
    // Select numbers with highest cycle scores (most "due")
    const mainNumbers = this.selectTopN(cycleScores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalCycleScores = new Map<number, number>();
      
      for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
        const score = this.analyzeAdditionalNumberCycle(num, history);
        additionalCycleScores.set(num, score);
      }
      
      additional = this.selectTopN(additionalCycleScores, config.additionalNumbersCount);
      
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
      confidence: 0.64,
      executionTime: performance.now() - startTime,
    };
  }

  private analyzeNumberCycle(num: number, history: HistoricalDraw[]): number {
    // Find all appearances of this number
    const appearances: number[] = [];
    
    history.forEach((draw, index) => {
      if (draw.numbers.includes(num)) {
        appearances.push(index);
      }
    });
    
    if (appearances.length < 3) {
      // Not enough data, use gap from last appearance
      return appearances.length === 0 ? history.length : appearances[0];
    }
    
    // Calculate average cycle length
    const cycles: number[] = [];
    for (let i = 1; i < appearances.length; i++) {
      cycles.push(appearances[i] - appearances[i - 1]);
    }
    
    const avgCycle = cycles.reduce((a, b) => a + b, 0) / cycles.length;
    const currentGap = appearances[0]; // Gap from most recent appearance
    
    // Score: how close is current gap to the average cycle?
    // Higher score if current gap is close to or exceeds average cycle
    const score = currentGap / avgCycle;
    
    return score;
  }

  private analyzeAdditionalNumberCycle(num: number, history: HistoricalDraw[]): number {
    const appearances: number[] = [];
    
    history.forEach((draw, index) => {
      if (draw.additionalNumbers?.includes(num)) {
        appearances.push(index);
      }
    });
    
    if (appearances.length < 2) {
      return appearances.length === 0 ? history.length : appearances[0];
    }
    
    const cycles: number[] = [];
    for (let i = 1; i < appearances.length; i++) {
      cycles.push(appearances[i] - appearances[i - 1]);
    }
    
    const avgCycle = cycles.reduce((a, b) => a + b, 0) / cycles.length;
    const currentGap = appearances[0];
    
    return currentGap / avgCycle;
  }
}
