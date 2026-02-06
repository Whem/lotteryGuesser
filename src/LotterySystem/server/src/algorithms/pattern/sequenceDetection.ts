import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class SequenceDetection extends BaseAlgorithm {
  name = 'sequence_detection';
  displayName = 'Sequence Detection';
  description = 'Identifies arithmetic and other sequences in winning numbers';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze arithmetic sequences
    const arithmeticScores = this.analyzeArithmeticSequences(history, config);
    
    // Analyze geometric patterns (multiples)
    const multipleScores = this.analyzeMultiples(history, config);
    
    // Combine scores
    const combinedScores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const arithmeticScore = arithmeticScores.get(num) || 0;
      const multipleScore = multipleScores.get(num) || 0;
      combinedScores.set(num, arithmeticScore + multipleScore);
    }
    
    // Select numbers with best sequence patterns
    const mainNumbers = this.selectTopN(combinedScores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
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
      confidence: 0.62,
      executionTime: performance.now() - startTime,
    };
  }

  private analyzeArithmeticSequences(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    
    // Find common differences in sequences
    const differences = new Map<number, number>();
    
    history.forEach(draw => {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      
      // Check for arithmetic progressions of length 3+
      for (let i = 0; i < sorted.length - 2; i++) {
        const diff1 = sorted[i + 1] - sorted[i];
        const diff2 = sorted[i + 2] - sorted[i + 1];
        
        if (diff1 === diff2 && diff1 > 0) {
          differences.set(diff1, (differences.get(diff1) || 0) + 1);
        }
      }
    });
    
    // Get most common differences
    const topDifferences = Array.from(differences.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(e => e[0]);
    
    // Score numbers that could continue sequences
    const lastDraw = history[0]?.numbers.sort((a, b) => a - b) || [];
    
    lastDraw.forEach(num => {
      topDifferences.forEach(diff => {
        for (let i = 1; i <= 3; i++) {
          const nextNum = num + (diff * i);
          const prevNum = num - (diff * i);
          
          if (nextNum >= config.minNumber && nextNum <= config.maxNumber) {
            scores.set(nextNum, (scores.get(nextNum) || 0) + (3 - i + 1));
          }
          if (prevNum >= config.minNumber && prevNum <= config.maxNumber) {
            scores.set(prevNum, (scores.get(prevNum) || 0) + (3 - i + 1));
          }
        }
      });
    });
    
    return scores;
  }

  private analyzeMultiples(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    
    // Find common base numbers for multiples
    const baseCounts = new Map<number, number>();
    
    history.forEach(draw => {
      draw.numbers.forEach(num => {
        // Check which bases this number is a multiple of
        for (let base = 2; base <= 10; base++) {
          if (num % base === 0) {
            baseCounts.set(base, (baseCounts.get(base) || 0) + 1);
          }
        }
      });
    });
    
    // Get most common bases
    const topBases = Array.from(baseCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(e => e[0]);
    
    // Score multiples of common bases
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      topBases.forEach((base, index) => {
        if (num % base === 0) {
          scores.set(num, (scores.get(num) || 0) + (3 - index));
        }
      });
    }
    
    return scores;
  }
}
