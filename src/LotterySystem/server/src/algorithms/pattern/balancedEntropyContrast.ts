import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class BalancedEntropyContrast extends BaseAlgorithm {
  name = 'balanced_entropy_contrast';
  displayName = 'Balanced Entropy Contrast';
  description = 'Uses entropy-based contrast to balance predictable and rare numbers';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const frequency = this.calculateFrequency(history);
    const totalDraws = history.length;
    
    // Calculate entropy contribution for each number
    const entropy = new Map<number, number>();
    let totalEntropy = 0;
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const freq = frequency.get(num) || 0;
      const p = freq / Math.max(1, totalDraws * config.numbersCount);
      
      if (p > 0) {
        const e = -p * Math.log2(p);
        entropy.set(num, e);
        totalEntropy += e;
      } else {
        entropy.set(num, 0);
      }
    }
    
    // Normalize and calculate contrast score
    const avgEntropy = totalEntropy / (config.maxNumber - config.minNumber + 1);
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const e = entropy.get(num) || 0;
      const freq = frequency.get(num) || 0;
      
      // High entropy = unpredictable, contrast with frequency
      const entropyScore = e / Math.max(0.001, avgEntropy);
      const freqScore = freq / Math.max(1, totalDraws);
      
      // Balance: some high entropy (rare/unpredictable) + some stable (frequent)
      const contrast = Math.abs(entropyScore - 1) * freqScore;
      scores.set(num, contrast + entropyScore * 0.5);
    }
    
    // Select mix of high and medium scorers
    const sorted = Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1]);
    
    const halfCount = Math.ceil(config.numbersCount / 2);
    const selected: number[] = [];
    
    // Top scorers
    for (let i = 0; i < halfCount && i < sorted.length; i++) {
      selected.push(sorted[i][0]);
    }
    
    // Mid scorers for balance
    const midStart = Math.floor(sorted.length / 3);
    for (let i = midStart; selected.length < config.numbersCount && i < sorted.length; i++) {
      if (!selected.includes(sorted[i][0])) {
        selected.push(sorted[i][0]);
      }
    }
    
    const main = selected.sort((a, b) => a - b).slice(0, config.numbersCount);
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
      confidence: 0.71,
      executionTime: performance.now() - startTime,
    };
  }
}

export const balancedEntropyContrast = new BalancedEntropyContrast();
