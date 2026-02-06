import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class RandomWalkMomentum extends BaseAlgorithm {
  name = 'random_walk_momentum';
  displayName = 'Random Walk with Momentum';
  description = 'Simulates random walk from hot numbers with momentum adjustment';
  category = 'PROBABILITY' as const;
  complexity = 'MODERATE' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const frequency = this.calculateFrequency(history.slice(0, 30));
    const recentFreq = this.calculateFrequency(history.slice(0, 5));
    
    // Calculate momentum for each number
    const momentum = new Map<number, number>();
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const recent = recentFreq.get(num) || 0;
      const overall = frequency.get(num) || 0;
      const m = recent * 3 - overall; // Positive = trending up
      momentum.set(num, m);
    }
    
    // Start random walks from high momentum numbers
    const startPoints = Array.from(momentum.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, config.numbersCount * 2)
      .map(([num]) => num);
    
    const selected = new Set<number>();
    const range = config.maxNumber - config.minNumber + 1;
    
    // Perform random walks
    for (const start of startPoints) {
      if (selected.size >= config.numbersCount) break;
      
      let current = start;
      const walkLength = Math.floor(Math.random() * 5) + 1;
      
      for (let step = 0; step < walkLength; step++) {
        const m = momentum.get(current) || 0;
        const bias = m > 0 ? 0.6 : 0.4; // Bias towards momentum direction
        
        const direction = Math.random() < bias ? 1 : -1;
        const stepSize = Math.floor(Math.random() * 3) + 1;
        
        current = current + direction * stepSize;
        
        // Wrap around
        if (current < config.minNumber) current = config.maxNumber - (config.minNumber - current - 1);
        if (current > config.maxNumber) current = config.minNumber + (current - config.maxNumber - 1);
      }
      
      if (current >= config.minNumber && current <= config.maxNumber) {
        selected.add(current);
      }
    }
    
    // Fill remaining
    while (selected.size < config.numbersCount) {
      const num = config.minNumber + Math.floor(Math.random() * range);
      selected.add(num);
    }
    
    const main = Array.from(selected).sort((a, b) => a - b).slice(0, config.numbersCount);
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
      confidence: 0.65,
      executionTime: performance.now() - startTime,
    };
  }
}

export const randomWalkMomentum = new RandomWalkMomentum();
