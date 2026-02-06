import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class SlidingWindowEntropy extends BaseAlgorithm {
  name = 'sliding_window_entropy';
  displayName = 'Sliding Window Entropy';
  description = 'Analyzes entropy changes across sliding windows to predict numbers';
  category = 'MACHINE_LEARNING' as const;
  complexity = 'MODERATE' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const windowSizes = [5, 10, 20];
    const entropyScores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let totalEntropyChange = 0;
      
      for (const windowSize of windowSizes) {
        // Calculate entropy in current window
        const currentWindow = history.slice(0, windowSize);
        const currentCount = currentWindow.filter(d => d.numbers.includes(num)).length;
        const currentP = currentCount / Math.max(1, windowSize);
        const currentEntropy = currentP > 0 ? -currentP * Math.log2(currentP) : 0;
        
        // Calculate entropy in previous window
        const prevWindow = history.slice(windowSize, windowSize * 2);
        const prevCount = prevWindow.filter(d => d.numbers.includes(num)).length;
        const prevP = prevCount / Math.max(1, prevWindow.length);
        const prevEntropy = prevP > 0 ? -prevP * Math.log2(prevP) : 0;
        
        // Entropy change indicates pattern shift
        const entropyChange = currentEntropy - prevEntropy;
        totalEntropyChange += entropyChange * (1 / windowSize); // Weight smaller windows more
      }
      
      // Numbers with increasing entropy (becoming more unpredictable) might be due
      // Numbers with stable high entropy are good candidates
      const baseFreq = this.calculateFrequency(history.slice(0, 30)).get(num) || 0;
      const score = baseFreq * 0.6 + (totalEntropyChange + 1) * 0.4;
      
      entropyScores.set(num, score);
    }
    
    // Select top scoring numbers
    const sorted = Array.from(entropyScores.entries())
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
    
    return { main, additional, confidence: 0.66, executionTime: performance.now() - startTime };
  }
}

export const slidingWindowEntropy = new SlidingWindowEntropy();
