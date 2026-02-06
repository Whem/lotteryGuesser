import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class MonteCarlo extends BaseAlgorithm {
  name = 'monte_carlo';
  displayName = 'Monte Carlo Simulation';
  description = 'Uses random sampling to simulate thousands of draws and find optimal numbers';
  category = 'PROBABILITY' as const;
  complexity = 'COMPLEX' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate weights based on historical frequency
    const frequency = this.calculateFrequency(history);
    const weights = this.normalizeWeights(frequency, config.minNumber, config.maxNumber);
    
    // Run Monte Carlo simulations
    const simulationCount = 10000;
    const numberCounts = new Map<number, number>();
    
    for (let i = 0; i < simulationCount; i++) {
      const simulatedDraw = this.simulateDraw(weights, config.numbersCount);
      simulatedDraw.forEach(num => {
        numberCounts.set(num, (numberCounts.get(num) || 0) + 1);
      });
    }
    
    // Select numbers that appeared most frequently in simulations
    const sortedBySimFreq = Array.from(numberCounts.entries())
      .sort((a, b) => b[1] - a[1]);
    
    const mainNumbers = sortedBySimFreq
      .slice(0, config.numbersCount)
      .map(e => e[0])
      .sort((a, b) => a - b);
    
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers with same approach
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFreq = this.calculateAdditionalFrequency(history);
      const additionalWeights = this.normalizeWeights(
        additionalFreq,
        config.additionalMinNumber!,
        config.additionalMaxNumber!
      );
      
      if (additionalWeights.size > 0) {
        const additionalCounts = new Map<number, number>();
        
        for (let i = 0; i < simulationCount; i++) {
          const simulatedAdditional = this.simulateDraw(additionalWeights, config.additionalNumbersCount);
          simulatedAdditional.forEach(num => {
            additionalCounts.set(num, (additionalCounts.get(num) || 0) + 1);
          });
        }
        
        additional = Array.from(additionalCounts.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, config.additionalNumbersCount)
          .map(e => e[0])
          .sort((a, b) => a - b);
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

  private normalizeWeights(
    frequency: Map<number, number>,
    min: number,
    max: number
  ): Map<number, number> {
    const weights = new Map<number, number>();
    
    // Add base weight for all numbers
    for (let i = min; i <= max; i++) {
      weights.set(i, (frequency.get(i) || 0) + 1); // +1 to ensure all numbers have some chance
    }
    
    // Normalize
    const total = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    weights.forEach((value, key) => {
      weights.set(key, value / total);
    });
    
    return weights;
  }

  private simulateDraw(weights: Map<number, number>, count: number): number[] {
    const result: number[] = [];
    const availableWeights = new Map(weights);
    
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
    
    return result;
  }
}
