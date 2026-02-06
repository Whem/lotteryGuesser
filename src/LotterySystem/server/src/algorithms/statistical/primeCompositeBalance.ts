import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class PrimeCompositeBalance extends BaseAlgorithm {
  name = 'prime_composite_balance';
  displayName = 'Prime-Composite Balance';
  description = 'Balances prime and composite numbers based on historical ratios';
  category = 'STATISTICAL' as const;
  complexity = 'MODERATE' as const;
  isPremium = false;

  private isPrime(n: number): boolean {
    if (n < 2) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;
    for (let i = 3; i <= Math.sqrt(n); i += 2) {
      if (n % i === 0) return false;
    }
    return true;
  }

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze historical prime/composite ratio
    let totalPrimes = 0;
    let totalNumbers = 0;
    
    for (const draw of history.slice(0, 50)) {
      for (const num of draw.numbers) {
        if (this.isPrime(num)) totalPrimes++;
        totalNumbers++;
      }
    }
    
    const targetPrimeRatio = totalNumbers > 0 ? totalPrimes / totalNumbers : 0.4;
    const targetPrimes = Math.round(config.numbersCount * targetPrimeRatio);
    const targetComposites = config.numbersCount - targetPrimes;
    
    // Get prime and composite numbers in range
    const primes: number[] = [];
    const composites: number[] = [];
    
    for (let i = config.minNumber; i <= config.maxNumber; i++) {
      if (this.isPrime(i)) primes.push(i);
      else composites.push(i);
    }
    
    // Score by frequency
    const frequency = this.calculateFrequency(history);
    primes.sort((a, b) => (frequency.get(b) || 0) - (frequency.get(a) || 0));
    composites.sort((a, b) => (frequency.get(b) || 0) - (frequency.get(a) || 0));
    
    // Select balanced mix
    const selected: number[] = [];
    selected.push(...primes.slice(0, targetPrimes));
    selected.push(...composites.slice(0, targetComposites));
    
    // Fill if needed
    while (selected.length < config.numbersCount) {
      const remaining = [...primes, ...composites].filter(n => !selected.includes(n));
      if (remaining.length > 0) {
        selected.push(remaining[Math.floor(Math.random() * remaining.length)]);
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
      confidence: 0.68,
      executionTime: performance.now() - startTime,
    };
  }
}

export const primeCompositeBalance = new PrimeCompositeBalance();
