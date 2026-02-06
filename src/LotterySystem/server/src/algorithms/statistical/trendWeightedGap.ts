import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class TrendWeightedGap extends BaseAlgorithm {
  name = 'trend_weighted_gap';
  displayName = 'Trend Weighted Gap';
  description = 'Combines gap analysis with trend momentum for optimal selection';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const gaps = this.computeGaps(history, config);
    const recentFreq = this.calculateFrequency(history.slice(0, 10));
    const olderFreq = this.calculateFrequency(history.slice(10, 30));
    
    // Calculate trend momentum
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const gap = gaps.get(num) || 0;
      const recent = recentFreq.get(num) || 0;
      const older = olderFreq.get(num) || 0;
      
      // Momentum: positive if trending up
      const momentum = recent - older;
      
      // Gap score: higher for longer gaps
      const avgGap = history.length / Math.max(1, recentFreq.get(num) || 1);
      const gapScore = gap > avgGap ? (gap / avgGap) : 0.5;
      
      // Combined score: balance gap reversion with momentum
      const score = gapScore * 0.6 + (momentum + 1) * 0.4;
      scores.set(num, score);
    }
    
    // Select top scoring numbers
    const sorted = Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([num]) => num);
    
    const main = sorted.slice(0, config.numbersCount).sort((a, b) => a - b);
    let additional: number[] | undefined;
    
    if (config.hasadditional && config.additionalCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalCount
      );
    }
    
    return {
      main,
      additional,
      confidence: 0.70,
      executionTime: performance.now() - startTime,
    };
  }

  private computeGaps(history: HistoricalDraw[], config: LotteryConfig): Map<number, number> {
    const gaps = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let gap = 0;
      for (const draw of history) {
        if (draw.numbers.includes(num)) break;
        gap++;
      }
      gaps.set(num, gap);
    }
    
    return gaps;
  }
}

export const trendWeightedGap = new TrendWeightedGap();
