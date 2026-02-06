import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class HierarchicalRegime extends BaseAlgorithm {
  name = 'hierarchical_regime';
  displayName = 'Hierarchical Regime Mixture';
  description = 'Multi-level regime analysis for adaptive number selection';
  category = 'EXPERIMENTAL' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Level 1: Short-term regime (last 5 draws)
    const shortTermFreq = this.calculateFrequency(history.slice(0, 5));
    
    // Level 2: Medium-term regime (last 20 draws)
    const mediumTermFreq = this.calculateFrequency(history.slice(0, 20));
    
    // Level 3: Long-term regime (all history)
    const longTermFreq = this.calculateFrequency(history);
    
    // Detect current regime
    const shortTermAvg = Array.from(shortTermFreq.values()).reduce((a, b) => a + b, 0) / shortTermFreq.size || 1;
    const mediumTermAvg = Array.from(mediumTermFreq.values()).reduce((a, b) => a + b, 0) / mediumTermFreq.size || 1;
    
    // Regime: 'hot' if short-term > medium-term, else 'cold'
    const isHotRegime = shortTermAvg > mediumTermAvg * 0.9;
    
    // Calculate hierarchical scores
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const shortScore = (shortTermFreq.get(num) || 0) / Math.max(1, shortTermAvg);
      const mediumScore = (mediumTermFreq.get(num) || 0) / Math.max(1, mediumTermAvg);
      const longScore = (longTermFreq.get(num) || 0) / Math.max(1, history.length * 0.05);
      
      // Combine scores based on regime
      let score: number;
      if (isHotRegime) {
        // In hot regime, favor short-term trends
        score = shortScore * 0.5 + mediumScore * 0.3 + longScore * 0.2;
      } else {
        // In cold regime, favor mean reversion
        score = (1 - shortScore) * 0.3 + mediumScore * 0.3 + longScore * 0.4;
      }
      
      scores.set(num, score);
    }
    
    // Select top scoring numbers
    const sorted = Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([num]) => num);
    
    const main = sorted.slice(0, config.numbersCount).sort((a, b) => a - b);
    let additional: number[] | undefined;
    
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }
    
    return { main, additional, confidence: 0.74, executionTime: performance.now() - startTime };
  }
}

export const hierarchicalRegime = new HierarchicalRegime();
