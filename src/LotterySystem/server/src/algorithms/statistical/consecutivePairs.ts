import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class ConsecutivePairs extends BaseAlgorithm {
  name = 'consecutive_pairs';
  displayName = 'Consecutive Pairs Analysis';
  description = 'Identifies and uses frequently occurring number pairs';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Build pair frequency map
    const pairFrequency = new Map<string, number>();
    
    history.forEach(draw => {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      
      // Check all pairs
      for (let i = 0; i < sorted.length; i++) {
        for (let j = i + 1; j < sorted.length; j++) {
          const pairKey = `${sorted[i]}-${sorted[j]}`;
          pairFrequency.set(pairKey, (pairFrequency.get(pairKey) || 0) + 1);
        }
      }
    });
    
    // Sort pairs by frequency
    const sortedPairs = Array.from(pairFrequency.entries())
      .sort((a, b) => b[1] - a[1]);
    
    // Select numbers from top pairs
    const selectedNumbers = new Set<number>();
    
    for (const [pairKey] of sortedPairs) {
      if (selectedNumbers.size >= config.numbersCount) break;
      
      const [num1, num2] = pairKey.split('-').map(Number);
      
      if (selectedNumbers.size < config.numbersCount && !selectedNumbers.has(num1)) {
        selectedNumbers.add(num1);
      }
      if (selectedNumbers.size < config.numbersCount && !selectedNumbers.has(num2)) {
        selectedNumbers.add(num2);
      }
    }
    
    const mainNumbers = Array.from(selectedNumbers).sort((a, b) => a - b);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalPairFreq = new Map<string, number>();
      
      history.forEach(draw => {
        if (!draw.additionalNumbers || draw.additionalNumbers.length < 2) return;
        
        const sorted = [...draw.additionalNumbers].sort((a, b) => a - b);
        for (let i = 0; i < sorted.length; i++) {
          for (let j = i + 1; j < sorted.length; j++) {
            const pairKey = `${sorted[i]}-${sorted[j]}`;
            additionalPairFreq.set(pairKey, (additionalPairFreq.get(pairKey) || 0) + 1);
          }
        }
      });
      
      if (additionalPairFreq.size > 0) {
        const sortedAdditionalPairs = Array.from(additionalPairFreq.entries())
          .sort((a, b) => b[1] - a[1]);
        
        const selectedAdditional = new Set<number>();
        for (const [pairKey] of sortedAdditionalPairs) {
          if (selectedAdditional.size >= config.additionalNumbersCount) break;
          const [num1, num2] = pairKey.split('-').map(Number);
          if (!selectedAdditional.has(num1)) selectedAdditional.add(num1);
          if (selectedAdditional.size < config.additionalNumbersCount && !selectedAdditional.has(num2)) {
            selectedAdditional.add(num2);
          }
        }
        
        additional = Array.from(selectedAdditional).sort((a, b) => a - b);
      }
      
      if (!additional || additional.length < config.additionalNumbersCount) {
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
      confidence: 0.66,
      executionTime: performance.now() - startTime,
    };
  }
}
