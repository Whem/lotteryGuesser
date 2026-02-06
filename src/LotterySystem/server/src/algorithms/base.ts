import { LotteryConfig, GeneratedNumbers, HistoricalDraw, PredictionAlgorithm } from '../types';

export abstract class BaseAlgorithm implements PredictionAlgorithm {
  abstract name: string;
  abstract displayName: string;
  abstract description: string;
  abstract category: PredictionAlgorithm['category'];
  abstract complexity: PredictionAlgorithm['complexity'];
  isPremium: boolean = false;

  abstract predict(
    config: LotteryConfig,
    history: HistoricalDraw[],
    options?: Record<string, any>
  ): Promise<GeneratedNumbers>;

  // Utility methods for all algorithms
  protected generateRandomNumbers(
    min: number,
    max: number,
    count: number,
    exclude: number[] = []
  ): number[] {
    const available = [];
    for (let i = min; i <= max; i++) {
      if (!exclude.includes(i)) {
        available.push(i);
      }
    }
    
    const result: number[] = [];
    while (result.length < count && available.length > 0) {
      const randomIndex = Math.floor(Math.random() * available.length);
      result.push(available.splice(randomIndex, 1)[0]);
    }
    
    return result.sort((a, b) => a - b);
  }

  protected calculateFrequency(history: HistoricalDraw[]): Map<number, number> {
    const frequency = new Map<number, number>();
    
    history.forEach(draw => {
      draw.numbers.forEach(num => {
        frequency.set(num, (frequency.get(num) || 0) + 1);
      });
    });
    
    return frequency;
  }

  protected calculateAdditionalFrequency(history: HistoricalDraw[]): Map<number, number> {
    const frequency = new Map<number, number>();
    
    history.forEach(draw => {
      draw.additionalNumbers?.forEach(num => {
        frequency.set(num, (frequency.get(num) || 0) + 1);
      });
    });
    
    return frequency;
  }

  protected selectTopN(
    scores: Map<number, number>,
    count: number,
    ascending: boolean = false
  ): number[] {
    const entries = Array.from(scores.entries());
    entries.sort((a, b) => ascending ? a[1] - b[1] : b[1] - a[1]);
    return entries.slice(0, count).map(e => e[0]).sort((a, b) => a - b);
  }

  protected fillRemaining(
    selected: number[],
    config: LotteryConfig,
    targetCount: number
  ): number[] {
    const result = [...selected];
    
    while (result.length < targetCount) {
      const num = Math.floor(Math.random() * (config.maxNumber - config.minNumber + 1)) + config.minNumber;
      if (!result.includes(num)) {
        result.push(num);
      }
    }
    
    return result.sort((a, b) => a - b);
  }

  protected weightedRandomSelect(
    weights: Map<number, number>,
    count: number,
    exclude: number[] = []
  ): number[] {
    const result: number[] = [];
    const availableWeights = new Map(weights);
    
    // Remove excluded numbers
    exclude.forEach(num => availableWeights.delete(num));
    
    while (result.length < count && availableWeights.size > 0) {
      const totalWeight = Array.from(availableWeights.values()).reduce((a, b) => a + b, 0);
      let random = Math.random() * totalWeight;
      
      for (const [num, weight] of availableWeights) {
        random -= weight;
        if (random <= 0) {
          result.push(num);
          availableWeights.delete(num);
          break;
        }
      }
    }
    
    return result.sort((a, b) => a - b);
  }

  protected calculateGaps(history: HistoricalDraw[], maxNumber: number): Map<number, number> {
    const gaps = new Map<number, number>();
    const lastSeen = new Map<number, number>();
    
    // Initialize all numbers with max gap
    for (let i = 1; i <= maxNumber; i++) {
      gaps.set(i, history.length);
    }
    
    // Calculate actual gaps
    history.forEach((draw, index) => {
      draw.numbers.forEach(num => {
        if (!lastSeen.has(num)) {
          gaps.set(num, index);
        }
        lastSeen.set(num, index);
      });
    });
    
    return gaps;
  }
}
