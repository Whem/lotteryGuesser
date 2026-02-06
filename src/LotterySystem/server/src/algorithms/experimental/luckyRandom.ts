import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class LuckyRandom extends BaseAlgorithm {
  name = 'lucky_random';
  displayName = 'Lucky Random';
  description = 'Generates truly random numbers with slight frequency bias';
  category = 'EXPERIMENTAL' as const;
  complexity = 'SIMPLE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // 70% random, 30% frequency-biased
    const randomCount = Math.ceil(config.numbersCount * 0.7);
    const biasedCount = config.numbersCount - randomCount;
    
    // Pure random selection
    const randomNumbers = this.generateRandomNumbers(
      config.minNumber,
      config.maxNumber,
      randomCount
    );
    
    // Frequency-biased selection for remaining
    const frequency = this.calculateFrequency(history);
    const availableNumbers = Array.from(frequency.entries())
      .filter(([num]) => !randomNumbers.includes(num))
      .sort((a, b) => b[1] - a[1]);
    
    const biasedNumbers = availableNumbers
      .slice(0, biasedCount)
      .map(e => e[0]);
    
    // Fill if needed
    const mainNumbers = [...randomNumbers, ...biasedNumbers];
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers - pure random
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }

    return {
      main: finalMain,
      additional,
      confidence: 0.5, // Lower confidence for random
      executionTime: performance.now() - startTime,
    };
  }
}
