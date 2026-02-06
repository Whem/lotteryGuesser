import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class OddEvenBalance extends BaseAlgorithm {
  name = 'odd_even_balance';
  displayName = 'Odd/Even Balance';
  description = 'Maintains optimal odd/even ratio based on historical patterns';
  category = 'STATISTICAL' as const;
  complexity = 'SIMPLE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze historical odd/even ratios
    const ratios = history.map(draw => {
      const oddCount = draw.numbers.filter(n => n % 2 === 1).length;
      return oddCount / draw.numbers.length;
    });
    
    const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    const targetOddCount = Math.round(config.numbersCount * avgRatio);
    const targetEvenCount = config.numbersCount - targetOddCount;
    
    // Get odd and even numbers with their frequencies
    const oddFreq = new Map<number, number>();
    const evenFreq = new Map<number, number>();
    
    history.forEach(draw => {
      draw.numbers.forEach(num => {
        if (num % 2 === 1) {
          oddFreq.set(num, (oddFreq.get(num) || 0) + 1);
        } else {
          evenFreq.set(num, (evenFreq.get(num) || 0) + 1);
        }
      });
    });
    
    // Select top odd and even numbers
    const selectedOdd = this.selectTopN(oddFreq, targetOddCount);
    const selectedEven = this.selectTopN(evenFreq, targetEvenCount);
    
    // Fill if needed
    const allOdds = [];
    const allEvens = [];
    for (let i = config.minNumber; i <= config.maxNumber; i++) {
      if (i % 2 === 1) allOdds.push(i);
      else allEvens.push(i);
    }
    
    while (selectedOdd.length < targetOddCount) {
      const available = allOdds.filter(n => !selectedOdd.includes(n));
      if (available.length === 0) break;
      selectedOdd.push(available[Math.floor(Math.random() * available.length)]);
    }
    
    while (selectedEven.length < targetEvenCount) {
      const available = allEvens.filter(n => !selectedEven.includes(n));
      if (available.length === 0) break;
      selectedEven.push(available[Math.floor(Math.random() * available.length)]);
    }
    
    const mainNumbers = [...selectedOdd, ...selectedEven].sort((a, b) => a - b);

    // Additional numbers with same logic
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalRatios = history
        .filter(d => d.additionalNumbers)
        .map(d => {
          const oddCount = d.additionalNumbers!.filter(n => n % 2 === 1).length;
          return oddCount / d.additionalNumbers!.length;
        });
      
      if (additionalRatios.length > 0) {
        const avgAdditionalRatio = additionalRatios.reduce((a, b) => a + b, 0) / additionalRatios.length;
        const targetAdditionalOdd = Math.round(config.additionalNumbersCount * avgAdditionalRatio);
        
        const additionalOdds = [];
        const additionalEvens = [];
        for (let i = config.additionalMinNumber!; i <= config.additionalMaxNumber!; i++) {
          if (i % 2 === 1) additionalOdds.push(i);
          else additionalEvens.push(i);
        }
        
        additional = [
          ...this.shuffleArray(additionalOdds).slice(0, targetAdditionalOdd),
          ...this.shuffleArray(additionalEvens).slice(0, config.additionalNumbersCount - targetAdditionalOdd),
        ].sort((a, b) => a - b);
      } else {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount
        );
      }
    }

    return {
      main: mainNumbers,
      additional,
      confidence: 0.62,
      executionTime: performance.now() - startTime,
    };
  }

  private shuffleArray<T>(arr: T[]): T[] {
    return [...arr].sort(() => Math.random() - 0.5);
  }
}
