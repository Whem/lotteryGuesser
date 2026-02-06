import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class EqualIntervalSpacing extends BaseAlgorithm {
  name = 'equal_interval_spacing';
  displayName = 'Equal Interval Spacing';
  description = 'Generates numbers with optimal equal spacing based on historical patterns';
  category = 'EXPERIMENTAL' as const;
  complexity = 'SIMPLE' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate optimal interval from historical data
    const intervals: number[] = [];
    
    for (const draw of history.slice(0, 30)) {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      for (let i = 1; i < sorted.length; i++) {
        intervals.push(sorted[i] - sorted[i - 1]);
      }
    }
    
    // Find most common interval
    const intervalCounts = new Map<number, number>();
    for (const interval of intervals) {
      intervalCounts.set(interval, (intervalCounts.get(interval) || 0) + 1);
    }
    
    let optimalInterval = Math.floor((config.maxNumber - config.minNumber) / (config.numbersCount + 1));
    let maxCount = 0;
    for (const [interval, count] of intervalCounts) {
      if (count > maxCount && interval > 0) {
        maxCount = count;
        optimalInterval = interval;
      }
    }
    
    // Try multiple starting points
    const frequency = this.calculateFrequency(history);
    let bestSelection: number[] = [];
    let bestScore = -1;
    
    const maxStart = config.maxNumber - (config.numbersCount - 1) * optimalInterval;
    for (let start = config.minNumber; start <= Math.min(maxStart, config.minNumber + 20); start++) {
      const selection: number[] = [];
      let current = start;
      
      for (let i = 0; i < config.numbersCount && current <= config.maxNumber; i++) {
        selection.push(current);
        current += optimalInterval;
      }
      
      if (selection.length === config.numbersCount) {
        const score = selection.reduce((sum, num) => sum + (frequency.get(num) || 0), 0);
        if (score > bestScore) {
          bestScore = score;
          bestSelection = selection;
        }
      }
    }
    
    // Fallback to random if no valid selection
    if (bestSelection.length < config.numbersCount) {
      bestSelection = this.generateRandomNumbers(config.minNumber, config.maxNumber, config.numbersCount);
    }
    
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }
    
    return {
      main: bestSelection.sort((a, b) => a - b),
      additional,
      confidence: 0.60,
      executionTime: performance.now() - startTime,
    };
  }
}

export const equalIntervalSpacing = new EqualIntervalSpacing();
