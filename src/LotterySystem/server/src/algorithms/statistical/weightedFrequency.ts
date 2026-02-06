import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class WeightedFrequency extends BaseAlgorithm {
  name = 'weighted_frequency';
  displayName = 'Weighted Frequency';
  description = 'Recent draws are weighted more heavily than older ones';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate weighted frequency (more recent = higher weight)
    const weights = new Map<number, number>();
    const decayFactor = 0.95; // Weight decays by 5% per draw
    
    history.forEach((draw, index) => {
      const weight = Math.pow(decayFactor, index); // Most recent has weight 1, decreasing
      
      draw.numbers.forEach(num => {
        weights.set(num, (weights.get(num) || 0) + weight);
      });
    });
    
    // Normalize weights
    const maxWeight = Math.max(...Array.from(weights.values()));
    weights.forEach((value, key) => {
      weights.set(key, value / maxWeight);
    });
    
    // Select using weighted random selection
    const mainNumbers = this.weightedRandomSelect(weights, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers with same approach
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalWeights = new Map<number, number>();
      
      history.forEach((draw, index) => {
        const weight = Math.pow(decayFactor, index);
        draw.additionalNumbers?.forEach(num => {
          additionalWeights.set(num, (additionalWeights.get(num) || 0) + weight);
        });
      });
      
      if (additionalWeights.size > 0) {
        additional = this.weightedRandomSelect(additionalWeights, config.additionalNumbersCount);
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
      confidence: 0.7,
      executionTime: performance.now() - startTime,
    };
  }
}
