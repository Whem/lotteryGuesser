import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class FibonacciSequence extends BaseAlgorithm {
  name = 'fibonacci_sequence';
  displayName = 'Fibonacci Sequence';
  description = 'Uses Fibonacci numbers and their relationships for prediction';
  category = 'EXPERIMENTAL' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Generate Fibonacci numbers within range
    const fibNumbers = this.generateFibonacciInRange(config.minNumber, config.maxNumber);
    
    // Score numbers based on Fibonacci proximity
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let score = 0;
      
      // Check if it's a Fibonacci number
      if (fibNumbers.includes(num)) {
        score += 3;
      }
      
      // Check proximity to Fibonacci numbers
      fibNumbers.forEach(fib => {
        const distance = Math.abs(num - fib);
        if (distance > 0 && distance <= 3) {
          score += (4 - distance) * 0.5;
        }
      });
      
      // Add historical frequency component
      const frequency = this.calculateFrequency(history);
      score += (frequency.get(num) || 0) / history.length;
      
      scores.set(num, score);
    }
    
    // Select based on scores
    const mainNumbers = this.selectTopN(scores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFib = this.generateFibonacciInRange(
        config.additionalMinNumber!,
        config.additionalMaxNumber!
      );
      
      if (additionalFib.length >= config.additionalNumbersCount) {
        additional = additionalFib.slice(0, config.additionalNumbersCount);
      } else {
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
      confidence: 0.55,
      executionTime: performance.now() - startTime,
    };
  }

  private generateFibonacciInRange(min: number, max: number): number[] {
    const fibNumbers: number[] = [];
    let a = 1, b = 1;
    
    while (a <= max) {
      if (a >= min) {
        fibNumbers.push(a);
      }
      const next = a + b;
      a = b;
      b = next;
    }
    
    return fibNumbers;
  }
}
