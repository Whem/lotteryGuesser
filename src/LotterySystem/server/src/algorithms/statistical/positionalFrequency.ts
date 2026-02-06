import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class PositionalFrequency extends BaseAlgorithm {
  name = 'positional_frequency';
  displayName = 'Positional Frequency';
  description = 'Analyzes which numbers appear most often at each position when sorted';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Track frequency at each position
    const positionFrequencies: Map<number, number>[] = [];
    for (let i = 0; i < config.numbersCount; i++) {
      positionFrequencies.push(new Map<number, number>());
    }
    
    // Analyze sorted numbers at each position
    history.forEach(draw => {
      const sortedNumbers = [...draw.numbers].sort((a, b) => a - b);
      sortedNumbers.forEach((num, position) => {
        const freq = positionFrequencies[position];
        freq.set(num, (freq.get(num) || 0) + 1);
      });
    });
    
    // Select most frequent number for each position
    const mainNumbers: number[] = [];
    const usedNumbers = new Set<number>();
    
    positionFrequencies.forEach(freqMap => {
      const sorted = Array.from(freqMap.entries())
        .filter(([num]) => !usedNumbers.has(num))
        .sort((a, b) => b[1] - a[1]);
      
      if (sorted.length > 0) {
        const selected = sorted[0][0];
        mainNumbers.push(selected);
        usedNumbers.add(selected);
      }
    });
    
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalPositionFreq: Map<number, number>[] = [];
      for (let i = 0; i < config.additionalNumbersCount; i++) {
        additionalPositionFreq.push(new Map<number, number>());
      }
      
      history.forEach(draw => {
        const sorted = [...(draw.additionalNumbers || [])].sort((a, b) => a - b);
        sorted.forEach((num, position) => {
          if (position < additionalPositionFreq.length) {
            const freq = additionalPositionFreq[position];
            freq.set(num, (freq.get(num) || 0) + 1);
          }
        });
      });
      
      additional = [];
      const usedAdditional = new Set<number>();
      
      additionalPositionFreq.forEach(freqMap => {
        const sorted = Array.from(freqMap.entries())
          .filter(([num]) => !usedAdditional.has(num))
          .sort((a, b) => b[1] - a[1]);
        
        if (sorted.length > 0) {
          additional!.push(sorted[0][0]);
          usedAdditional.add(sorted[0][0]);
        }
      });
      
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount
        );
      }
    }

    return {
      main: finalMain.sort((a, b) => a - b),
      additional: additional?.sort((a, b) => a - b),
      confidence: 0.68,
      executionTime: performance.now() - startTime,
    };
  }
}
